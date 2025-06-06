import torch
import torch.distributed as dist

import gc
import time
import logging
from PIL import Image
import argparse
from datetime import datetime
import GPUtil
import json
import os

from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe
from para_attn.parallel_vae.diffusers_adapters import parallelize_vae

from diffusers import AutoencoderKLWan, WanImageToVideoPipeline,WanTransformer3DModel
from diffusers.utils import export_to_video, load_image
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from transformers import CLIPVisionModel, UMT5EncoderModel
from diffusers.video_processor import VideoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompilationConfig():
    attention_type: str = "sage"
    cache_type: str = "teacache"
    cache_threshold: float = 0.1
    compilation : bool = False
    model_id : str = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
    warmup : bool = True

class CompiledWanModel():
    
    def __init__(self,config: CompilationConfig):
        logger.info(f"Initializing CompiledWanModel pipeline with model")
        
        dist.init_process_group(backend='nccl', init_method='env://')
        rank = dist.get_rank()
        logger.info(f"Rank: {rank}")
        torch.cuda.set_device(rank)
        start_load_time = time.time()
        
        self.model_id = config.model_id
        self.attention_type = config.attention_type
        self.cache_type = config.cache_type
        self.cache_threshold = config.cache_threshold
        self.compilation = config.compilation
        self.warmup = config.warmup
        
        self.pipe = None
        
        if self.model_id == "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers":
            self.flow_shift = 3.0
        elif self.model_id == "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers":
            self.flow_shift = 5.0
        else:
            self.flow_shift = 3.0
        
        try:
            self.load_model()
            self.optimize_pipe()
            logger.info(f"Pipeline initialized {self.model_id} in {time.time() - start_load_time} seconds")
            if self.warmup:
                self.warmup_step()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise e
                
    def max_memory_usage(self):
        """Get maximum GPU memory usage in MB"""
        gpu = GPUtil.getGPUs()[0]
        return gpu.memoryTotal
    
    def get_gpu_memory_usage(self):
        """Get current GPU memory usage in MB"""
        gpu = GPUtil.getGPUs()[0]
        return gpu.memoryUsed
    
    def log_gpu_memory_usage(self, message=""):
        """Log current GPU memory usage"""
        memory_used = self.get_gpu_memory_usage()
        memory_total = self.max_memory_usage()
        memory_percent = (memory_used / memory_total) * 100
        logger.info(f"GPU Memory Usage {message}: {memory_used:.2f}MB / {memory_total:.2f}MB ({memory_percent:.2f}%)")
        return memory_used
                
    def load_model(self):
        if self.pipe is None:
            self.text_encoder = UMT5EncoderModel.from_pretrained(self.model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16)
            self.vae = AutoencoderKLWan.from_pretrained(self.model_id, subfolder="vae", torch_dtype=torch.float32)
            self.image_encoder = CLIPVisionModel.from_pretrained(self.model_id, subfolder="image_encoder", torch_dtype=torch.float32)
            self.transformer = WanTransformer3DModel.from_pretrained(self.model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
            self.transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
            self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4
            self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
            self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
            
            self.pipe = WanImageToVideoPipeline.from_pretrained(
                self.model_id,
                vae=self.vae,
                transformer=self.transformer,
                text_encoder=self.text_encoder,
                image_encoder=self.image_encoder,
                torch_dtype=torch.bfloat16
            )
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config, flow_shift=self.flow_shift)
            self.pipe.to("cuda")
            
            self.pipe
            
    def apply_cache(self):
        if self.cache_type == "fbcache":
            from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe
            apply_cache_on_pipe(self.pipe , residual_diff_threshold=self.cache_threshold)
            logger.info(f"Cache applied with fbcache and residual_diff_threshold = {self.cache_threshold}")
        elif self.cache_type == "teacache":
            from cachify import teacache_forward
            import types
            # Properly bind the teacache_forward as a method
            self.pipe.transformer.forward = types.MethodType(teacache_forward, self.pipe.transformer)
            self.pipe.transformer.__class__.enable_teacache = True
            self.pipe.transformer.__class__.cnt = 0
            self.pipe.transformer.__class__.num_steps = 30
            self.pipe.transformer.__class__.rel_l1_thresh = self.cache_threshold 
            self.pipe.transformer.__class__.accumulated_rel_l1_distance = 0
            self.pipe.transformer.__class__.previous_modulated_input = None
            self.pipe.transformer.__class__.previous_residual = None
            return
        else:
            raise ValueError(f"Invalid cache type: {self.cache_type}")
            
    def apply_sage_attention(self):
        from sageattention import sageattn
        from sage_attn import set_sage_attn_wan
        set_sage_attn_wan(self.pipe.transformer, sageattn)   
        
    def optimize_pipe(self):
        parallelize_vae(self.vae)
        parallelize_pipe( 
            self.pipe,
            mesh=init_context_parallel_mesh(
                self.pipe.device.type,
            ),
        )
        self.apply_cache()
        
        self.apply_sage_attention()
        
        if self.compilation:
            torch._inductor.config.reorder_for_compute_comm_overlap = True
            self.pipe.transformer = torch.compile(self.pipe.transformer, mode="max-autotune-no-cudagraphs")
    
    def warmup_step(self):
        logger.info("Running Warm Up!")
        prompt = "A car driving on a road"
        negative_prompt = "blurry, low quality, dark"
        image = load_image("https://storage.googleapis.com/falserverless/gallery/car_720p.png")
        start_time = time.time()
        self.log_gpu_memory_usage("before warmup")
        with torch.inference_mode():
            self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            num_inference_steps=30,
            height=576,
            width=1024,
            num_frames=81,
                guidance_scale=5.0
            )
        self.get_matrix(start_time,time.time(),576,1024)
        logger.info("Warm Up Completed!")
    
    def get_matrix(self,start_time : int,end_time : int,height : int,width : int):
        logger.info("-"*40)
        logger.info(f"Order_ID : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"inference time : {end_time - start_time}")
        logger.info(f"height : {height}")
        logger.info(f"width : {width}")
        logger.info("-"*40)
        logger.info("\n")
        
    def generate_video(self,prompt : str,negative_prompt : str,image : Image.Image,num_frames : int = 81,guidance_scale : float = 5.0,num_inference_steps : int = 30,height : int = 576,width : int = 1024,fps : int = 16):
        with torch.inference_mode():
            output = self.pipe(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                output_type="pil",
            ).frames[0]
            
        if dist.get_rank() == 0:
            import uuid 
            export_to_video(output, f"outputs/compiled_{uuid.uuid4()}.mp4", fps=fps)
            return output
        
    def shutdown(self):
        dist.destroy_process_group()
        logger.info("Pipeline shutdown completed")
        
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers")
    parser.add_argument("--attention_type", type=str, default="sage")
    parser.add_argument("--cache_type", type=str, default="teacache")
    parser.add_argument("--cache_threshold", type=float, default=0.1)
    parser.add_argument("--compilation", type=bool, default=False)
    
    os.environ["MASTER_ADDR"] = "localhost"  # or the IP of the master node
    os.environ["MASTER_PORT"] = "29500"      # any free port                # rank of this process
    os.environ["WORLD_SIZE"] = str(8)
    
    config = CompilationConfig()
    config.model_id = parser.parse_args().model_id
    config.attention_type = parser.parse_args().attention_type
    config.cache_type = parser.parse_args().cache_type
    config.cache_threshold = parser.parse_args().cache_threshold
    config.compilation = parser.parse_args().compilation
    
    model = CompiledWanModel(config)
    
    with open("inputs.json", "r") as f:
            inputs = json.load(f)
            
    for i in range(0,len(inputs)):
        prompt = inputs[str(i+1)]["prompt"]
        negative_prompt = inputs[str(i+1)]["negative_prompt"]
        image = load_image(inputs[str(i+1)]["image"])
        image = image.resize((1280,720))
        start_time = time.time()
        latents = model.generate_video(prompt=prompt,negative_prompt=negative_prompt,image=image,height=720,width=1280,num_frames=81,guidance_scale=5.0,num_inference_steps=30,fps=16)
        end_time = time.time()
        model.get_matrix(start_time,end_time,1280,720)
    
    model.shutdown()