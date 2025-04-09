import torch
import torch.distributed as dist

from diffusers import AutoencoderKLWan, WanImageToVideoPipeline,WanTransformer3DModel
from diffusers.utils import export_to_video, load_image
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

from transformers import CLIPVisionModel, UMT5EncoderModel

from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe

import gc
import time
import logging
from PIL import Image
import argparse
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WanI2V():
    """
    Wan Image to Video Pipeline
    Supports 14B and 720P models (Wan-AI/Wan2.1-I2V-14B-480P-Diffusers, Wan-AI/Wan2.1-I2V-14B-720P-Diffusers)
    """
    
    def __init__(self,model_id,apply_cache=True,cache_threshold=0.1):
        logger.info(f"Initializing WanI2V pipeline with model {model_id}")
        dist.init_process_group()
        torch.cuda.set_device(dist.get_rank())
        start_load_time = time.time()
        self.pipe = None
        self.model_id = model_id
        self.apply_cache = apply_cache
        self.cache_threshold = cache_threshold
        if self.model_id == "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers":
            self.flow_shift = 3.0
        elif self.model_id == "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers":
            self.flow_shift = 5.0
        
        try:
            self.load_model()
            self.optimize_pipe()
            logger.info(f"Pipeline initialized {model_id} in {time.time() - start_load_time} seconds")
            self.warmup()
        except Exception as e:
            logger.error(f"Error initializing {model_id}: {e}")
            raise e
                
    def load_model(self):
        self.text_encoder = UMT5EncoderModel.from_pretrained(self.model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16)
        self.vae = AutoencoderKLWan.from_pretrained(self.model_id, subfolder="vae", torch_dtype=torch.float32)
        self.transformer = WanTransformer3DModel.from_pretrained(self.model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
        self.image_encoder = CLIPVisionModel.from_pretrained(self.model_id, subfolder="image_encoder", torch_dtype=torch.float32)
        
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
        
    def optimize_pipe(self):
        self.pipe.transformer.enable_gradient_checkpointing()  # Enable gradient checkpointing
        self.pipe.enable_attention_slicing(slice_size="auto")  
        parallelize_pipe( 
            self.pipe,
            mesh=init_context_parallel_mesh(
                self.pipe.device.type,
            ),
        )
        if self.apply_cache:
            from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe
            apply_cache_on_pipe(self.pipe , residual_diff_threshold=self.cache_threshold)
            
    def clear_memory(self):
        torch.cuda.empty_cache()
        gc.collect()
        
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
                output_type="pil" if dist.get_rank() == 0 else "pt",
            ).frames[0]
            
        if dist.get_rank() == 0:
            self.clear_memory()
            if isinstance(output[0], torch.Tensor):
                output = [frame.cpu() if frame.device.type == 'cuda' else frame for frame in output]
            export_to_video(output, "wan-i2v.mp4", fps=fps)
    
    def warmup(self):
        logger.info("Running Warm Up!")
        prompt = "A car driving on a road"
        negative_prompt = "blurry, low quality, dark"
        image = load_image("https://storage.googleapis.com/falserverless/gallery/car_720p.png")
        start_time = time.time()
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
        
    def shutdown(self):
        dist.destroy_process_group()
        
    def get_matrix(self,start_time : int,end_time : int,height : int,width : int):
        with open("matrix.txt", "a") as f:
            f.write("-"*40)
            f.write(f"Order_ID : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"inference time : {end_time - start_time}\n")
            f.write(f"height : {height}\n")
            f.write(f"width : {width}\n")
            f.write("-"*40)
            f.write("\n")
        
        
        
if __name__ == "__main__":
    inputs = {
    "1" : {
        "prompt" : "Cars racing in slow motion",
        "negative_prompt" : "bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards",
        "image" : load_image("https://storage.googleapis.com/falserverless/gallery/car_720p.png")
    },
    "2" : {
        "prompt" : "A cat in a car",
        "negative_prompt" : "bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards",
            "image" : load_image("https://fancypawscatclinic.com/uploads/SiteAssets/426/images/services/payment-options-cat-720px.jpg")
        }
    }
    
    # RESOLUTION_CONFIG = {
    #     "Horizontal" : {
    #         "height" : 576,
    #         "width" : 1024
    #     },
    #     "Vertical" : {
    #         "height" : 1024,
    #         "width" : 576
    #     },
    #     "Square" : {
    #         "height" : 768,
    #         "width" : 768
    #     }
    # }
    parser = argparse.ArgumentParser()
    parser.add_argument("height", type=int)
    parser.add_argument("width", type=int)
    parser.add_argument("apply_cache", type=bool,default=True)
    parser.add_argument("cache_threshold", type=float,default=0.1)
    args = parser.parse_args()
    height = args.height
    width = args.width
    apply_cache = args.apply_cache
    cache_threshold = args.cache_threshold
    WanModel = WanI2V("Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",apply_cache=apply_cache,cache_threshold=cache_threshold)
    
    for i in range(0,len(inputs)):
            prompt = inputs[str(i+1)]["prompt"]
            negative_prompt = inputs[str(i+1)]["negative_prompt"]
            image = inputs[str(i+1)]["image"]
            start_time = time.time()
            WanModel.generate_video(prompt=prompt,negative_prompt=negative_prompt,image=image,height=height,width=width,num_frames=81,guidance_scale=5.0,num_inference_steps=30,fps=16)
            end_time = time.time()
            WanModel.get_matrix(start_time,end_time,height,width)
    WanModel.shutdown()