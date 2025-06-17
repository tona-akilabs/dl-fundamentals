from PIL import Image
import torchvision.transforms as transforms
import torch

# Define image transform (resize and convert to tensor)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ensure same size
    transforms.ToTensor(),          # convert to tensor [0, 1]
])

# Load images
img1 = Image.open("images/cat01.jpg").convert("RGB")
img2 = Image.open("images/fish01.jpg").convert("RGB")
img3 = Image.open("images/cat02.jpg").convert("RGB")
img4 = Image.open("images/cat01.jpg").convert("RGB")

# Transform images
tensor1 = transform(img1)
tensor2 = transform(img2)
tensor3 = transform(img3)
tensor4 = transform(img4)

diff = torch.norm(tensor1 - tensor2)  # scalar
print(f"L2 distance: {diff.item()}")

l1_diff = torch.abs(tensor1 - tensor2).mean()
print(f"L1 distance: {l1_diff.item()}")

l3_diff = torch.abs(tensor1 - tensor3).mean()
print(f"L3 distance: {l3_diff.item()}")

#l4_same = torch.abs(tensor1 - tensor4).mean()
l4_same = torch.norm(tensor1 - tensor4).mean()
print(f"L4 distance: {l4_same.item()}")


cos = torch.nn.functional.cosine_similarity(
    tensor1.flatten(), tensor2.flatten(), dim=0
)
print(f"Cosine similarity: {cos.item()}")