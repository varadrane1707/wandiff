import torch
import torch.distributed as dist
import numpy as np
import time
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel
from PIL import Image
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
import io

dist.init_process_group()

torch.cuda.set_device(dist.get_rank())

start_loading = time.time()

data_type = torch.bfloat16

# Available models: Wan-AI/Wan2.1-I2V-14B-480P-Diffusers, Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
model_id = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"

image_encoder = CLIPVisionModel.from_pretrained(
    model_id, subfolder="image_encoder", torch_dtype=data_type
)
vae = AutoencoderKLWan.from_pretrained(
    model_id, subfolder="vae", torch_dtype=torch.float32
)
pipe = WanImageToVideoPipeline.from_pretrained(
    model_id, vae=vae, image_encoder=image_encoder, torch_dtype=data_type
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)
pipe.to("cuda")

# Import and apply parallel attention
from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe

parallelize_pipe(
    pipe,
    mesh=init_context_parallel_mesh(
        pipe.device.type,
    ),
)
end_loading = time.time()

from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe
apply_cache_on_pipe(pipe , residual_diff_threshold=0.05)

# pipe.enable_model_cpu_offload(gpu_id=dist.get_rank())
# pipe.enable_vae_tiling()

# torch._inductor.config.reorder_for_compute_comm_overlap = True
# pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")
print("Pipeline Loaded.....")
loading_time = end_loading - start_loading

prompt = (
    "Cars racing in slow motion"
)
negative_prompt = (
    "bright colors, overexposed, static, blurred details, subtitles, style, artwork, painting, picture, still, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still picture, cluttered background, three legs, many people in the background, walking backwards"
)

image = load_image(
    "https://storage.googleapis.com/falserverless/gallery/car_720p.png"
)

# max_area = 1024 * 1024
# aspect_ratio = image.height / image.width
# mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
# print(f"MOD VALUE :{mod_value}")
# height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
# width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
width , height = image.size
# image = image.resize((width, height))

# Start measuring inference time
start_inference = time.time()

# Run the pipeline
output = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    num_frames=81,
    guidance_scale=5.0,
    num_inference_steps=30,
    output_type="pil" if dist.get_rank() == 0 else "pt",
).frames[0]

# End of inference time measurement
end_inference = time.time()
inference_time = end_inference - start_inference

# Save output and print timing info
if dist.get_rank() == 0:
    print(f"{'=' * 50}")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Number of GPUs: {dist.get_world_size()}")
    print(f"Pipeline Loading Time: {loading_time:.2f} seconds")
    print(f"Pipeline Inference Time: {inference_time:.2f} seconds")
    print(f"Resolution: {width}x{height}")
    print(f"{'=' * 50}")
    print("Saving video to wan-i2v.mp4")
    export_to_video(output, "wan-i2v.mp4", fps=16)

dist.destroy_process_group() 