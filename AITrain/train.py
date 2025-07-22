import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from facenet_pytorch import InceptionResnetV1
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import random
import torchvision
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
import os

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] 사용 디바이스: {device}")

# 표준화 함수 정의
def fixed_image_standardization_def(tensor):
    return (tensor - 0.5) / 0.5

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: fixed_image_standardization_def(x))  # Lambda로 감싸기
])

# 데이터 로드
dataset_dir = 'dataset'
print(f"[INFO] 데이터셋 경로: {dataset_dir}")

dataset = ImageFolder(dataset_dir, transform=transform)
print(f"[INFO] 클래스 수: {len(dataset.classes)}")
print(f"[INFO] 총 이미지 수: {len(dataset.samples)}")

# 모델 정의 (사전학습 X)

model = InceptionResnetV1(
    pretrained='vggface2',   # 얼굴 임베딩용으로 사용
    classify=False            # 분류기가 아니라 임베딩 학습용
).to(device)
print("[INFO] InceptionResnetV1 모델 생성 완료")

# 손실함수 및 옵티마이저
criterion = nn.TripletMarginLoss(margin=1.0, p=2)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Triplet 샘플링 함수
def sample_triplet(dataset):
    classes = dataset.classes
    class_to_idx = dataset.class_to_idx

    anchor_class = random.choice(classes)
    anchor_imgs = [x[0] for x in dataset.samples if x[1] == class_to_idx[anchor_class]]

    if len(anchor_imgs) < 2:
        return None  # Triplet을 구성할 수 없음

    anchor_img = random.choice(anchor_imgs)
    positive_img = random.choice([img for img in anchor_imgs if img != anchor_img])

    negative_class = random.choice([c for c in classes if c != anchor_class])
    negative_imgs = [x[0] for x in dataset.samples if x[1] == class_to_idx[negative_class]]
    negative_img = random.choice(negative_imgs)

    return anchor_img, positive_img, negative_img, anchor_class, negative_class

# 학습 루프
epochs = 5
triplet_per_epoch = 100

for epoch in range(epochs):
    model.train()
    total_loss = 0
    print(f"\n[Epoch {epoch+1}/{epochs}] 시작")

    for step in range(triplet_per_epoch):
        triplet = sample_triplet(dataset)

        if triplet is None:
            print("[WARN] Triplet 샘플링 실패 (데이터 부족)")
            continue

        anchor_path, positive_path, negative_path, anchor_class, negative_class = triplet

        # 이미지 로드 및 전처리
        anchor_img = transform(to_pil_image(read_image(anchor_path).float() / 255.0)).unsqueeze(0).to(device)
        positive_img = transform(to_pil_image(read_image(positive_path).float() / 255.0)).unsqueeze(0).to(device)
        negative_img = transform(to_pil_image(read_image(negative_path).float() / 255.0)).unsqueeze(0).to(device)

        # 임베딩 추출
        optimizer.zero_grad()
        anchor_embed = model(anchor_img)
        positive_embed = model(positive_img)
        negative_embed = model(negative_img)

        # Triplet Loss 계산
        loss = criterion(anchor_embed, positive_embed, negative_embed)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 상세 출력
        print(f"[Epoch {epoch+1}/{epochs}] Step {step+1}/{triplet_per_epoch} | "
              f"Anchor Class: {anchor_class} | Negative Class: {negative_class} | "
              f"Loss: {loss.item():.4f}")

    avg_loss = total_loss / triplet_per_epoch
    print(f"[Epoch {epoch+1}] 평균 손실: {avg_loss:.4f}")

print("\n[INFO] 학습 완료")
