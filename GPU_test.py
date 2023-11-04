import torch

# Check if a GPU is available
if torch.cuda.is_available():
    # Get the name of the GPU
    gpu_name = torch.cuda.get_device_name(0)  # 0 refers to the first GPU, change if using multiple GPUs
    print(f"GPU ({gpu_name}) is available for computation.")
else:
    print("No GPU available. Using CPU for computation.")
