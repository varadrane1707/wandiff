import torch
import numpy as np
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanTransformer3DModel
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel,UMT5EncoderModel
from diffusers.hooks.group_offloading import apply_group_offloading
import time

# Available models: Wan-AI/Wan2.1-I2V-14B-480P-Diffusers, Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
model_id = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
image_encoder = CLIPVisionModel.from_pretrained(
    model_id, subfolder="image_encoder", torch_dtype=torch.float32
)

text_encoder = UMT5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16)
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
transformer = WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16)

onload_device = torch.device("cuda")
offload_device = torch.device("cpu")

apply_group_offloading(text_encoder,
    onload_device=onload_device,
    offload_device=offload_device,
    offload_type="block_level",
    num_blocks_per_group=4
)

transformer.enable_group_offload(
    onload_device=onload_device,
    offload_device=offload_device,
    offload_type="block_level",
    num_blocks_per_group=4,
)
pipe = WanImageToVideoPipeline.from_pretrained(
    model_id,
    vae=vae,
    transformer=transformer,
    text_encoder=text_encoder,
    image_encoder=image_encoder,
    torch_dtype=torch.bfloat16
)
# replace this with pipe.to("cuda") if you have sufficient VRAM
# pipe.enable_model_cpu_offload()
pipe.to("cuda")
# from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe
# apply_cache_on_pipe(pipe)

image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
)

max_area = 1280 * 720
aspect_ratio = image.height / image.width
mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
image = image.resize((width, height))

prompt = (
    "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in "
    "the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
)
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

num_frames = 81

for i in range(0,2):
    print(f"Generating frame {i} of {num_frames}")
    start_time = time.time()
    output = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=30,
        guidance_scale=5.0,
    ).frames[i]
    end_time = time.time()
    print(f"Time taken to generate frame {i}: {end_time - start_time} seconds")
    export_to_video(output, f"wan-i2v-test-{i}.mp4", fps=16)