import torch
import torch.distributed as dist
import time
from dataclasses import dataclass
from typing import Union
from pathlib import Path

from diffusers import (
    AutoencoderKLWan,
    WanImageToVideoPipeline,
    WanTransformer3DModel,
    UniPCMultistepScheduler
)
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel, UMT5EncoderModel
from PIL import Image

@dataclass
class WanPipelineConfig:
    model_id: str = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
    data_type: torch.dtype = torch.bfloat16
    device: str = "cuda"
    width: int = 1024
    height: int = 576
    num_frames: int = 81
    guidance_scale: float = 5.0
    num_inference_steps: int = 30
    fps: int = 16

class WanI2VPipeline:
    def __init__(self, config: WanPipelineConfig):
        self.config = config
        self.pipe = None
        self.setup_distributed()
        
    def setup_distributed(self):
        """Initialize distributed training setup"""
        if not dist.is_initialized():
            dist.init_process_group()
        torch.cuda.set_device(dist.get_rank())
        
    def load_models(self):
        """Load and initialize all required models"""
        try:
            print("Loading models...")
            start_time = time.time()
            
            # Load all model components
            image_encoder = CLIPVisionModel.from_pretrained(
                self.config.model_id,
                subfolder="image_encoder",
                torch_dtype=torch.float32
            )
            
            text_encoder = UMT5EncoderModel.from_pretrained(
                self.config.model_id,
                subfolder="text_encoder",
                torch_dtype=self.config.data_type
            )
            
            vae = AutoencoderKLWan.from_pretrained(
                self.config.model_id,
                subfolder="vae",
                torch_dtype=torch.float32
            )
            
            transformer = WanTransformer3DModel.from_pretrained(
                self.config.model_id,
                subfolder="transformer",
                torch_dtype=self.config.data_type
            )
            
            # Initialize pipeline
            self.pipe = WanImageToVideoPipeline.from_pretrained(
                self.config.model_id,
                vae=vae,
                transformer=transformer,
                text_encoder=text_encoder,
                image_encoder=image_encoder,
                torch_dtype=self.config.data_type
            )
            
            # Configure scheduler and move to device
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(
                self.pipe.scheduler.config,
                flow_shift=5.0
            )
            self.pipe.to(self.config.device)
            
            # Apply optimizations
            self._apply_optimizations()
            
            print(f"Models loaded in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load models: {str(e)}")
            
    def _apply_optimizations(self):
        """Apply various pipeline optimizations"""
        from para_attn.context_parallel import init_context_parallel_mesh
        from para_attn.context_parallel.diffusers_adapters import parallelize_pipe
        from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe
        
        # Apply parallel attention
        parallelize_pipe(
            self.pipe,
            mesh=init_context_parallel_mesh(self.pipe.device.type)
        )
        
        # Apply caching
        apply_cache_on_pipe(self.pipe, residual_diff_threshold=0.1)
        
    def generate_video(
        self,
        image_path: Union[str, Path],
        prompt: str,
        negative_prompt: str,
        output_path: str = "output.mp4"
    ) -> None:
        """Generate video from input image"""
        try:
            # Load and preprocess image
            image = self._prepare_image(image_path)
            
            # Generate video frames
            print("Generating video...")
            start_time = time.time()
            
            output = self.pipe(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=self.config.height,
                width=self.config.width,
                num_frames=self.config.num_frames,
                guidance_scale=self.config.guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
                output_type="pil" if dist.get_rank() == 0 else "pt",
            ).frames[0]
            
            # Save video if primary process
            if dist.get_rank() == 0:
                self._save_video(output, output_path)
                self._print_statistics(start_time)
                
        except Exception as e:
            raise RuntimeError(f"Video generation failed: {str(e)}")
        finally:
            self._cleanup()
            
    def _prepare_image(self, image_path: Union[str, Path]) -> Image.Image:
        """Load and preprocess input image"""
        image = load_image(image_path)
        return image.resize((self.config.width, self.config.height))
        
    def _save_video(self, frames, output_path: str):
        """Save generated frames as video"""
        if isinstance(frames[0], torch.Tensor):
            frames = [frame.cpu() if frame.device.type == 'cuda' else frame for frame in frames]
        export_to_video(frames, output_path, fps=self.config.fps)
        print(f"Video saved to {output_path}")
        
    def _print_statistics(self, start_time: float):
        """Print generation statistics"""
        print(f"{'=' * 50}")
        print(f"Device: {torch.cuda.get_device_name()}")
        print(f"Number of GPUs: {dist.get_world_size()}")
        print(f"Resolution: {self.config.width}x{self.config.height}")
        print(f"Generation Time: {time.time() - start_time:.2f} seconds")
        print(f"{'=' * 50}")
        
    def _cleanup(self):
        """Cleanup resources"""
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        
    def __del__(self):
        """Cleanup on destruction"""
        if dist.is_initialized():
            dist.destroy_process_group()

# Example usage:
if __name__ == "__main__":
    config = WanPipelineConfig()
    pipeline = WanI2VPipeline(config)
    pipeline.load_models()
    
    prompt = "Cars racing in slow motion"
    negative_prompt = (
        "bright colors, overexposed, static, blurred details, subtitles, "
        "style, artwork, painting, picture, still, overall gray, worst quality, "
        "low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
        "poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, "
        "fused fingers, still picture, cluttered background, three legs, "
        "many people in the background, walking backwards"
    )
    
    pipeline.generate_video(
        image_path="car_720p.png",
        prompt=prompt,
        negative_prompt=negative_prompt,
        output_path="wan-i2v.mp4"
    )