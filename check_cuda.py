import torch
import os

print(f"--- PyTorch Diagnostics ---")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA_VISIBLE_DEVICES Environment Variable: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    try:
        print(f"Attempting to allocate a tensor on GPU {torch.cuda.current_device()}...")
        a = torch.tensor([1.0, 2.0]).cuda()
        print(f"Successfully allocated tensor: {a} on device {a.device}")
    except Exception as e:
        print(f"Error during CUDA tensor allocation test: {e}")
else:
    print("CUDA is NOT available according to this PyTorch installation.")

print(f"--- End Diagnostics ---")
