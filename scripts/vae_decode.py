from diffusers import AutoencoderKLWan
import torch
import logging
import GPUtil
from diffusers.video_processor import VideoProcessor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def max_memory_usage():
       return GPUtil.getGPUs()[0].memoryTotal

def get_gpu_memory_usage():
    """Get current GPU memory usage in MB"""
    gpu = GPUtil.getGPUs()[0]
    return gpu.memoryUsed

def log_gpu_memory_usage(message=""):
    """Log current GPU memory usage"""
    memory_used = get_gpu_memory_usage()
    memory_total = max_memory_usage()
    memory_percent = (memory_used / memory_total) * 100
    logger.info(f"GPU Memory Usage {message}: {memory_used:.2f}MB / {memory_total:.2f}MB ({memory_percent:.2f}%)")
    return memory_used

def load_latents(latents_path):
    latents = torch.load(latents_path)
    print(f"Latent shape: {latents.shape}")
    return latents

def decode_latents(latents, vae,video_processor,output_type="pil"):
    torch.cuda.empty_cache()
    log_gpu_memory_usage("before decoding")
    size = latents.element_size() * latents.numel() / (1024 ** 3)
    logger.info(f"Size of latents: {size:.2f}GB")
    latents = latents.to(vae.dtype)
    latents = latents.to(vae.device)
    print(latents.device)
    print(latents.dtype)
    log_gpu_memory_usage("after converting to vae dtype")
    size = latents.element_size() * latents.numel() / (1024 ** 3)
    logger.info(f"Size of latents: {size:.2f}GB")
    latents_mean = (
            torch.tensor(vae.config.latents_mean)
            .view(1, vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
    latents = latents / latents_std + latents_mean
    log_gpu_memory_usage("after adding latents mean and std")
    with torch.inference_mode():
        video = vae.decode(latents, return_dict=False)[0]
    log_gpu_memory_usage("after decoding")
    video = video_processor.postprocess_video(video, output_type=output_type)
    log_gpu_memory_usage("after postprocessing")
    return video


if __name__ == "__main__":
    import time
    time_start = time.time()
    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.1-I2V-14B-720P-Diffusers", subfolder="vae", torch_dtype=torch.float16)
    video_processor = VideoProcessor(vae_scale_factor=8)
    vae.to("cuda")
    latents = load_latents("latents.pt")
    decoded_video = decode_latents(latents, vae,video_processor)
    decoded_video[0].save("decoded_video.mp4", fps=16)
    time_end = time.time()
    logger.info(f"Time taken: {time_end - time_start:.2f} seconds")
    # import os
    # import time
    # from tile_vae import WanVAE
    # from diffusers.video_processor import VideoProcessor
    # video_processor = VideoProcessor(vae_scale_factor=8)
    # vae_path = "Wan2.1_VAE.pth"
    # vae = WanVAE(
    #         vae_pth=vae_path,
    #         device="cuda")
    # time_start = time.time()
    # VAE_tile_size = 0
    # video = vae.decode(latents, VAE_tile_size, any_end_frame= True)[0]
    # video = video[:,  :-1] 
    
    # Fix the dimensions issue by reshaping the tensor
    # VideoProcessor expects [batch, channels, frames, height, width]
    # Check current shape
    # print(f"Video shape before reshape: {video.shape}")
    
    # # If video shape is [channels, frames, height, width], add batch dimension
    # if len(video.shape) == 4:
    #     video = video.unsqueeze(0)  # Add batch dimension
    #     print(f"Video shape after reshape: {video.shape}")
    
    # # Process and save the video
    # video = video_processor.postprocess_video(video, output_type="pil")
    # from diffusers.utils import export_to_video
    # export_to_video(video[0], "wan-i2v-test.mp4", fps=16)
    time_end = time.time()
    logger.info(f"Time taken: {time_end - time_start:.2f} seconds")
    
    