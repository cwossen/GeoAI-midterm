import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from contrastive_dataset import ContrastiveDataset
from model import ContrastiveModel

def main():
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.GaussianBlur(kernel_size=11),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = ContrastiveDataset(
        root_dir=r"C:/PythonProjects/VGGFace2/train",  # ✅ Confirm this path
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,  # 💡 Set to 0 on Windows to avoid multiprocessing issues
        pin_memory=True
    )

    model = ContrastiveModel()
    model.eval()

    anchor, positive = next(iter(dataloader))
    with torch.no_grad():
        out1 = model(anchor)
        out2 = model(positive)

    print("Anchor output shape:", out1.shape)
    print("Positive output shape:", out2.shape)

if __name__ == "__main__":
    main()
