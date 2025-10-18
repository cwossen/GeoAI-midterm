from contrastive_dataset import ContrastiveDataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = ContrastiveDataset(root_dir=r"C:\PythonProjects\VGGFace2\train", transform=transform)

# Inspect 3 samples
for i in range(3):
    a, p, n, label = dataset[i]
    print(f"Sample {i}: label={label}, anchor={a.shape}, positive={p.shape}, negative={n.shape}")
