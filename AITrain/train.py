import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from torchvision.datasets import ImageFolder
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
import random

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] 사용 디바이스: {device}")

# 표준화 함수 정의
def fixed_image_standardization_def(tensor):
    return (tensor - 0.5) / 0.5

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: fixed_image_standardization_def(x))
])

# 데이터 로드
dataset_dir = 'dataset'  # 데이터셋 경로 지정
print(f"[INFO] 데이터셋 경로: {dataset_dir}")

dataset = ImageFolder(dataset_dir, transform=transform)
print(f"[INFO] 클래스 수: {len(dataset.classes)}")
print(f"[INFO] 총 이미지 수: {len(dataset.samples)}")

# 모델 정의 (사전학습된 VGGFace2 임베딩 모델)
model = InceptionResnetV1(pretrained='vggface2', classify=False).to(device)
print("[INFO] InceptionResnetV1 모델 생성 완료")

# 손실 함수 및 옵티마이저
criterion = nn.TripletMarginLoss(margin=1.0, p=2)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Triplet 샘플링 함수 (단일 triplet 반환)
def sample_triplet(dataset):
    classes = dataset.classes
    class_to_idx = dataset.class_to_idx

    if len(classes) < 2:
        raise ValueError("데이터셋에 2개 이상의 클래스가 필요합니다.")

    anchor_class = random.choice(classes)
    anchor_imgs = [x[0] for x in dataset.samples if x[1] == class_to_idx[anchor_class]]

    if len(anchor_imgs) < 2:
        return None

    anchor_img = random.choice(anchor_imgs)
    positive_img = random.choice([img for img in anchor_imgs if img != anchor_img])

    negative_classes = [c for c in classes if c != anchor_class]
    if not negative_classes:
        return None

    negative_class = random.choice(negative_classes)
    negative_imgs = [x[0] for x in dataset.samples if x[1] == class_to_idx[negative_class]]

    if len(negative_imgs) == 0:
        return None

    negative_img = random.choice(negative_imgs)

    return anchor_img, positive_img, negative_img, anchor_class, negative_class

# 배치 단위로 Triplet 샘플을 모으는 함수
def sample_triplet_batch(dataset, batch_size):
    batch = []
    while len(batch) < batch_size:
        triplet = sample_triplet(dataset)
        if triplet is not None:
            batch.append(triplet)
    return batch

# 학습 변수
epochs = 5
triplet_per_epoch = 100
batch_size = 8  # 원하는 배치 크기 지정 (예: 8)

print(f"[INFO] 배치 크기: {batch_size}")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    print(f"\n[Epoch {epoch+1}/{epochs}] 시작")

    steps_per_epoch = triplet_per_epoch // batch_size

    for step in range(steps_per_epoch):
        batch = sample_triplet_batch(dataset, batch_size)

        anchor_imgs = []
        positive_imgs = []
        negative_imgs = []

        for triplet in batch:
            anchor_path, positive_path, negative_path, anchor_class, negative_class = triplet

            anchor_img = transform(to_pil_image(read_image(anchor_path).float() / 255.0))
            positive_img = transform(to_pil_image(read_image(positive_path).float() / 255.0))
            negative_img = transform(to_pil_image(read_image(negative_path).float() / 255.0))

            anchor_imgs.append(anchor_img.unsqueeze(0))
            positive_imgs.append(positive_img.unsqueeze(0))
            negative_imgs.append(negative_img.unsqueeze(0))

        anchor_imgs = torch.cat(anchor_imgs, dim=0).to(device)
        positive_imgs = torch.cat(positive_imgs, dim=0).to(device)
        negative_imgs = torch.cat(negative_imgs, dim=0).to(device)

        optimizer.zero_grad()

        anchor_embeds = model(anchor_imgs)
        positive_embeds = model(positive_imgs)
        negative_embeds = model(negative_imgs)

        loss = criterion(anchor_embeds, positive_embeds, negative_embeds)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        print(f"[Epoch {epoch+1}/{epochs}] Step {step+1}/{steps_per_epoch} | Loss: {loss.item():.4f}")

    avg_loss = total_loss / steps_per_epoch
    print(f"[Epoch {epoch+1}] 평균 손실: {avg_loss:.4f}")

print("\n[INFO] 학습 완료")

# 모델 저장
save_path = 'facenet_model.pth'
torch.save(model.state_dict(), save_path)
print(f"[INFO] 모델 가중치를 '{save_path}'에 저장했습니다.")
