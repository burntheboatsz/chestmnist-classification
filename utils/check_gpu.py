import torch

print("=" * 60)
print("GPU STATUS CHECK")
print("=" * 60)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")

if torch.cuda.is_available():
    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print("=" * 60)
    print("STATUS: GPU ACTIVE")
else:
    print("=" * 60)
    print("STATUS: GPU NOT DETECTED!")
print("=" * 60)
