import torch
import sys

def check_cuda_info():
    print("\n=== CUDA Device Information ===")
    print(f"CUDA is available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Number of CUDA devices: {device_count}")
        
        print("\n=== Device Details ===")
        for i in range(device_count):
            print(f"\nDevice {i}:")
            print(f"  Name: {torch.cuda.get_device_name(i)}")
            print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
            
        current_device = torch.cuda.current_device()
        print(f"\nCurrent device: {current_device}")
        print(f"Current device name: {torch.cuda.get_device_name(current_device)}")
        
        # Try to get memory info for current device
        try:
            print("\n=== Memory Information ===")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        except Exception as e:
            print(f"Could not get memory information: {e}")
    else:
        print("No CUDA devices available")

if __name__ == "__main__":
    check_cuda_info() 