import torch
import torch.distributed as dist
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

import os 
os.environ["MASTER_ADDR"] = "localhost"  # or the IP of the master node
os.environ["MASTER_PORT"] = "29500"      # any free port                # rank of this process
os.environ["WORLD_SIZE"] = "8"
dist.init_process_group(backend='nccl', init_method='env://')
rank = dist.get_rank()
print(f"Rank: {rank}")
torch.cuda.set_device(rank)

pipe = WanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.1-I2V-14B-720P-Diffusers", torch_dtype=torch.bfloat16)

pipe.to(torch.device("cuda"))

from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe

parallelize_pipe(
    pipe,
    mesh=init_context_parallel_mesh(
        pipe.device.type,
    ),
)

from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

apply_cache_on_pipe(pipe,residual_diff_threshold=0.15)

with open("inputs.json", "r") as f:
    import json
    inputs = json.load(f)

import torch
import torch.distributed as dist



def generate_video(pipe,prompt,negative_prompt,image):
    output = pipe(
                image=image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=720,
                width=1280,
                num_frames=81,
                guidance_scale=5.0,
                num_inference_steps=30,
                output_type="pil",
            ).frames[0]
    if dist.get_rank() == 0:
        import uuid 
        export_to_video(output, f"outputs/output_{uuid.uuid4()}.mp4", fps=16)
    return True
import time
for i,input in enumerate(inputs):
    time_start = time.time()
    prompt = inputs[str(i+1)]["prompt"]
    negative_prompt = inputs[str(i+1)]["negative_prompt"]
    image = load_image(inputs[str(i+1)]["image"])
    generate_video(pipe,prompt,negative_prompt,image)
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")
    
for i,input in enumerate(inputs):
    time_start = time.time()
    prompt = inputs[str(i+1)]["prompt"]
    negative_prompt = inputs[str(i+1)]["negative_prompt"]
    image = load_image(inputs[str(i+1)]["image"])
    generate_video(pipe,prompt,negative_prompt,image)
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")
    
dist.destroy_process_group()