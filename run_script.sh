wget https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_720p_14B_fp8_e4m3fn.safetensors
torchrun --nproc-per-node=8 wan_i2v.py --input_json inputs.json --resolution 1280*720 --apply_cache True --cache_threshold 0.1 --world_size 8 --num_frames 61