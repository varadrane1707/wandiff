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
    video = vae.decode(latents, return_dict=False)[0]
    log_gpu_memory_usage("after decoding")
    video = video_processor.postprocess_video(video, output_type=output_type)
    log_gpu_memory_usage("after postprocessing")
    return video


if __name__ == "__main__":
    vae = AutoencoderKLWan.from_pretrained("Wan-AI/Wan2.1-I2V-14B-720P-Diffusers", subfolder="vae", torch_dtype=torch.float16).to("cuda")
    video_processor = VideoProcessor(vae_scale_factor=8)
    latents = load_latents("latents.pt")
    # print(f"Latent shape: {latents.shape}")
    decoded_video = decode_latents(latents, vae,video_processor)
    decoded_video.save("decoded_video.mp4")
    
    