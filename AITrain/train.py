import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from facenet_pytorch import InceptionResnetV1
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from facenet_pytorch.models.utils import fixed_image_standardization
import random

# 데이터 변환
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    fixed_image_standardization
])

# 데이터 로드
dataset = ImageFolder('dataset', transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 모델 정의 (pretrained=False로 처음부터 학습 가능)
model = InceptionResnetV1(pretrained=False, classify=False, num_classes=None).to('cuda')

# Triplet Loss
criterion = nn.TripletMarginLoss(margin=1.0, p=2)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Triplet 샘플링 함수
def sample_triplet(dataset):
    classes = dataset.classes
    class_to_idx = dataset.class_to_idx

    anchor_class = random.choice(classes)
    anchor_imgs = [x[0] for x in dataset.samples if x[1] == class_to_idx[anchor_class]]

    # anchor와 positive 선택
    anchor_img = random.choice(anchor_imgs)
    positive_img = random.choice([img for img in anchor_imgs if img != anchor_img])

    # negative는 다른 클래스에서
    negative_class = random.choice([c for c in classes if c != anchor_class])
    negative_imgs = [x[0] for x in dataset.samples if x[1] == class_to_idx[negative_class]]
    negative_img = random.choice(negative_imgs)

    return anchor_img, positive_img, negative_img

# 학습 루프
epochs = 10

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for i in range(len(loader)):
        # Triplet 샘플링
        anchor_path, positive_path, negative_path = sample_triplet(dataset)

        anchor = transform(transforms.functional.pil_to_tensor(transforms.functional.to_pil_image(torchvision.io.read_image(anchor_path).float() / 255.0))).unsqueeze(0).to('cuda')
        positive = transform(transforms.functional.pil_to_tensor(transforms.functional.to_pil_image(torchvision.io.read_image(positive_path).float() / 255.0))).unsqueeze(0).to('cuda')
        negative = transform(transforms.functional.pil_to_tensor(transforms.functional.to_pil_image(torchvision.io.read_image(negative_path).float() / 255.0))).unsqueeze(0).to('cuda')

        optimizer.zero_grad()
        anchor_embed = model(anchor)
        positive_embed = model(positive)
        negative_embed = model(negative)

        loss = criterion(anchor_embed, positive_embed, negative_embed)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")
