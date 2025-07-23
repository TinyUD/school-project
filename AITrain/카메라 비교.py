import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
from sklearn.metrics.pairwise import cosine_similarity
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import warnings
import time
import cv2  # OpenCV 추가

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")
warnings.filterwarnings("ignore", category=FutureWarning, module="facenet_pytorch")

# --- 1. 디바이스 설정 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 2. FaceNet 모델 관련 설정 ---
def fixed_image_standardization_def(tensor):
    return (tensor - 0.5) / 0.5

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: fixed_image_standardization_def(x))
])

model = InceptionResnetV1(pretrained='vggface2', classify=False).to(device)
save_path = 'facenet_model.pth'
try:
    model.load_state_dict(torch.load(save_path, map_location=device))
    print(f"[INFO] 모델 가중치를 '{save_path}'에서 성공적으로 로드했습니다.")
except FileNotFoundError:
    print(f"[ERROR] '{save_path}' 파일을 찾을 수 없습니다. 학습된 모델이 저장되었는지 확인해주세요.")
    exit()
model.eval()

# --- 3. MTCNN 설정 ---
mtcnn = MTCNN(keep_all=False, margin=100, image_size=160, post_process=False, device=device)

# --- 4. 임베딩 추출 함수 ---
def get_face_embedding(image_path, debug_mode=False):
    try:
        image = Image.open(image_path).convert('RGB')
        face_cropped = mtcnn(image)

        if face_cropped is None:
            print(f"[DEBUG] 이미지 '{image_path}'에서 얼굴을 찾지 못했습니다.")
            return None

        if debug_mode:
            debug_output_dir = "debug_cropped_faces"
            os.makedirs(debug_output_dir, exist_ok=True)
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            debug_output_path = os.path.join(debug_output_dir, f"{name}_cropped_face{ext}")
            to_pil_image(face_cropped.cpu()).save(debug_output_path)
            print(f"[DEBUG] 크롭된 얼굴 이미지 저장: {debug_output_path}")

        processed_face = transform(to_pil_image(face_cropped.cpu()))
        processed_face = processed_face.unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model(processed_face)
        return embedding.squeeze(0).cpu()

    except Exception as e:
        print(f"[ERROR] 이미지 '{image_path}' 처리 중 오류 발생: {e}") 
        return None

# --- 5. 얼굴 검증 ---
def verify_faces(image_path1, image_path2, threshold=0.7, debug_mode=False):
    embedding1 = get_face_embedding(image_path1, debug_mode)
    embedding2 = get_face_embedding(image_path2, debug_mode)

    if embedding1 is None:
        return f"검증 실패: '{image_path1}'에서 유효한 얼굴을 찾을 수 없습니다."
    if embedding2 is None:
        return f"검증 실패: '{image_path2}'에서 유효한 얼굴을 찾을 수 없습니다."

    similarity = cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))[0][0]

    if similarity >= threshold:
        return f"동일 인물입니다! (유사도: {similarity:.4f})"
    else:
        return f"다른 인물입니다. (유사도: {similarity:.4f})"

# --- 7. 카메라에서 이미지 캡처 함수 ---
def capture_from_camera(temp_filename='temp_camera.jpg'):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 웹캠을 열 수 없습니다.")
        return None
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("[ERROR] 웹캠 이미지 캡처 실패.")
        return None

    # BGR → RGB 변환 및 저장
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame)
    image.save(temp_filename)
    return temp_filename

# --- 8. 실시간 비교 루프 함수 ---
def compare_with_live_camera(reference_image_path, interval_sec=5, threshold=0.7):
    print(f"\n[INFO] 기준 이미지: {reference_image_path}")
    print("[INFO] 실시간 얼굴 비교 시작... (Ctrl+C로 중단)")

    try:
        while True:
            captured_path = capture_from_camera()
            if captured_path is not None:
                result = verify_faces(captured_path, reference_image_path, threshold=threshold)
                print(f"[{time.strftime('%H:%M:%S')}] 결과: {result}")
            time.sleep(interval_sec)
    except KeyboardInterrupt:
        print("\n[INFO] 실시간 비교 중단됨.")

# --- 9. 실행 블록 ---
if __name__ == '__main__':
    reference_image = '비교/12.jpg'
    compare_with_live_camera(reference_image_path=reference_image, interval_sec=5, threshold=0.5)
