import torch

print(f"PyTorch version: {torch.__version__}")
# Máy bạn sẽ trả về False vì không có card NVIDIA, điều này là bình thường
print(f"Is CUDA available? {torch.cuda.is_available()}")