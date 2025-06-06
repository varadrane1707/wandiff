import torch


from diffusers import AutoencoderKLWan, WanImageToVideoPipeline,WanTransformer3DModel
from diffusers.utils import export_to_video, load_image
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

from transformers import CLIPVisionModel, UMT5EncoderModel

# from para_attn.context_parallel import init_context_parallel_mesh
# from para_attn.context_parallel.diffusers_adapters import parallelize_pipe

from diffusers.video_processor import VideoProcessor

import gc
import time
import logging
from PIL import Image
import argparse
from datetime import datetime
import GPUtil
import json
import os
         
from para_attn.parallel_vae.diffusers_adapters import parallelize_vae



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WanI2V():
    """
    Wan Image to Video Pipeline
    Supports 14B and 720P models (Wan-AI/Wan2.1-I2V-14B-480P-Diffusers, Wan-AI/Wan2.1-I2V-14B-720P-Diffusers)
    """
    
    def __init__(self,model_id,apply_cache=True,cache_threshold=0.1,quantization_tf=False,do_warmup=False):
        logger.info(f"Initializing WanI2V pipeline with model {model_id}")
        start_load_time = time.time()
        self.pipe = None
        self.model_id = model_id
        self.apply_cache = apply_cache
        self.cache_threshold = cache_threshold
        self.quantization_tf = quantization_tf
        self.max_memory_used = 0  # Track maximum memory usage
        
        if self.model_id == "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers":
            self.flow_shift = 3.0
        elif self.model_id == "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers":
            self.flow_shift = 5.0
        
        try:
            self.load_model()
            self.optimize_pipe()
            logger.info(f"Pipeline initialized {model_id} in {time.time() - start_load_time} seconds")
            if do_warmup:
                self.warmup()
        except Exception as e:
            logger.error(f"Error initializing {model_id}: {e}")
            raise e
    
    def max_memory_usage(self):
       return GPUtil.getGPUs()[0].memoryTotal

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
        
        self.log_gpu_memory_usage("before loading models")
        self.text_encoder = UMT5EncoderModel.from_pretrained(self.model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16)
        self.log_gpu_memory_usage("after loading text_encoder")
        
        self.vae = AutoencoderKLWan.from_pretrained(self.model_id, subfolder="vae", torch_dtype=torch.float32)
        self.log_gpu_memory_usage("after loading vae")
        
        if self.quantization_tf:
            ckpt_path="wan2.1_i2v_720p_14B_fp8_e4m3fn.safetensors"
            self.transformer = WanTransformer3DModel.from_single_file(ckpt_path,torch_dtype=torch.float8_e4m3fn)
        else:
            self.transformer = WanTransformer3DModel.from_pretrained(self.model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
        self.log_gpu_memory_usage("after loading transformer")
        
        self.transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)

        
        self.image_encoder = CLIPVisionModel.from_pretrained(self.model_id, subfolder="image_encoder", torch_dtype=torch.float32)
        self.log_gpu_memory_usage("after loading image_encoder")
        
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
        self.log_gpu_memory_usage("after creating pipeline")
        
    def optimize_pipe(self):
        self.log_gpu_memory_usage("before optimization")
        # self.pipe.transformer.enable_gradient_checkpointing()  # Enable gradient checkpointing
        # self.pipe.enable_attention_slicing(slice_size="auto")  
        # parallelize_pipe( 
        #     self.pipe,
        #     mesh=init_context_parallel_mesh(
        #         self.pipe.device.type,
        #     ),
        # )
        if self.apply_cache:
            from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe
            apply_cache_on_pipe(self.pipe , residual_diff_threshold=self.cache_threshold)
        self.log_gpu_memory_usage("after optimization")
            
    def clear_memory(self):
        self.log_gpu_memory_usage("before clearing memory")
        torch.cuda.empty_cache()
        gc.collect()
        self.log_gpu_memory_usage("after clearing memory")
        
    def generate_video(self,prompt : str,negative_prompt : str,image : Image.Image,num_frames : int = 81,guidance_scale : float = 5.0,num_inference_steps : int = 30,height : int = 576,width : int = 1024,fps : int = 16):
        self.log_gpu_memory_usage("before video generation")
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
                output_type="latent",
            ).frames[0]
            
        self.clear_memory()
        # if isinstance(output[0], torch.Tensor):
        #     output = [frame.cpu() if frame.device.type == 'cuda' else frame for frame in output]
        # export_to_video(output, "wan-i2v.mp4", fps=fps)
        self.log_gpu_memory_usage("after latent generation")
        
        output = self.decode_video(output)
        return output
        
    def decode_video(self,latents,output_type="pil"):
        
        #get size of latents in GB
        size = latents.element_size() * latents.numel() / (1024 ** 3)
        logger.info(f"Size of latents: {size:.2f}GB")
        
        #convert vae to bfloat16
        self.vae.to(torch.bfloat16)
        latents = latents.to(self.vae.dtype)
        
        #get size of latents in GB
        size = latents.element_size() * latents.numel() / (1024 ** 3)
        logger.info(f"Size of latents: {size:.2f}GB")
        
        latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)
        return video
    
    def warmup(self):
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
        self.clear_memory()
        self.log_gpu_memory_usage("after warmup")
        
    def shutdown(self):
        self.log_gpu_memory_usage("before shutdown")
        self.log_gpu_memory_usage("after shutdown")
        
    def get_matrix(self,start_time : int,end_time : int,height : int,width : int):
        logger.info("-"*40)
        logger.info(f"Order_ID : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"inference time : {end_time - start_time}")
        logger.info(f"height : {height}")
        logger.info(f"width : {width}")
        logger.info(f"max GPU memory used: {self.max_memory_used:.2f}MB")
        logger.info("-"*40)
        logger.info("\n")
        
        
        
if __name__ == "__main__":
    
    RESOLUTION_CONFIG = {
        "Horizontal" : {
            "height" : 576,
            "width" : 1024
        },
        "Vertical" : {
            "height" : 1024,
            "width" : 576
        },
        "Square" : {
            "height" : 768,
            "width" : 768
        },
        "1280*720" : {
            "height" : 720,
            "width" : 1280
        }
        
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str,default="inputs.json")
    parser.add_argument("--resolution", type=str,default="Horizontal")
    parser.add_argument("--apply_cache", type=bool,default=True)
    parser.add_argument("--cache_threshold", type=float,default=0.1)
    parser.add_argument("--quantization_tf", type=bool,default=False)
    parser.add_argument("--world_size", type=int,default=1) 
    parser.add_argument("--num_frames", type=int,default=81)
    parser.add_argument("--do_warmup", type=bool,default=False)
    args = parser.parse_args()  
    resolution = args.resolution
    apply_cache = args.apply_cache
    cache_threshold = args.cache_threshold
    quantization_tf = args.quantization_tf
    world_size = args.world_size
    num_frames = args.num_frames
    do_warmup = args.do_warmup
    os.environ["MASTER_ADDR"] = "localhost"  # or the IP of the master node
    os.environ["MASTER_PORT"] = "29500"      # any free port                # rank of this process
    os.environ["WORLD_SIZE"] = str(world_size) 
    if args.input_json:
        with open(args.input_json, "r") as f:
            inputs = json.load(f)
    else:
        inputs = {
            "1" : {
                "prompt" : "A car driving on a road",
                "negative_prompt" : "blurry, low quality, dark",
                "image" : "https://storage.googleapis.com/falserverless/gallery/car_720p.png"
            }
        }
    
    WanModel = WanI2V("Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",apply_cache=apply_cache,cache_threshold=cache_threshold,quantization_tf=quantization_tf,do_warmup=do_warmup)
    WanModel.log_gpu_memory_usage("at script start")
    
    for i in range(0,len(inputs)):
        prompt = inputs[str(i+1)]["prompt"]
        negative_prompt = inputs[str(i+1)]["negative_prompt"]
        image = load_image(inputs[str(i+1)]["image"])
        image = image.resize((RESOLUTION_CONFIG[resolution]["width"],RESOLUTION_CONFIG[resolution]["height"]))
        start_time = time.time()
        WanModel.generate_video(prompt=prompt,negative_prompt=negative_prompt,image=image,height=RESOLUTION_CONFIG[resolution]["height"],width=RESOLUTION_CONFIG[resolution]["width"],num_frames=num_frames,guidance_scale=5.0,num_inference_steps=30,fps=16)
        end_time = time.time()
        WanModel.get_matrix(start_time,end_time,RESOLUTION_CONFIG[resolution]["height"],RESOLUTION_CONFIG[resolution]["width"])
    
    WanModel.log_gpu_memory_usage("at script end")
    WanModel.shutdown()