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

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")
warnings.filterwarnings("ignore", category=FutureWarning, module="facenet_pytorch")

# --- 1. 디바이스 설정 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

# --- 2. FaceNet 모델 관련 설정 ---
def fixed_image_standardization_def(tensor):
    return (tensor - 0.5) / 0.5

# get_face_embedding에서 PIL Image로 변환된 후 사용할 transform
# MTCNN에서 이미 (160, 160)으로 크롭되므로 Resize는 필요 없음
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

# --- 3. MTCNN 얼굴 검출 및 전처리 관련 설정 ---
# MTCNN 모델 인스턴스 (얼굴 크롭 및 정렬)
# get_face_embedding에서 이미지 로드 후 바로 사용할 수 있도록 keep_all=False로 설정하여 하나의 얼굴만 반환
mtcnn = MTCNN(keep_all=False, margin=100, image_size=160, post_process=False, device=device)

# --- 4. 임베딩 추출 함수 (MTCNN 전처리 적용) ---
def get_face_embedding(image_path, debug_mode=False): # debug_mode 인자 추가
    try:
        image = Image.open(image_path).convert('RGB')
        face_cropped = mtcnn(image)

        if face_cropped is None:
            print(f"[DEBUG] 이미지 '{image_path}'에서 얼굴을 찾지 못했습니다.")
            return None

        if debug_mode: # 디버그 모드일 경우 크롭된 얼굴 저장
            debug_output_dir = "debug_cropped_faces"
            os.makedirs(debug_output_dir, exist_ok=True)
            # 원본 파일명에 _cropped_face를 붙여 저장
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

# --- 5. 얼굴 검증 시스템 ---
def verify_faces(image_path1, image_path2, threshold=0.7, debug_mode=False): # debug_mode 인자 추가
    embedding1 = get_face_embedding(image_path1, debug_mode) # debug_mode 전달
    embedding2 = get_face_embedding(image_path2, debug_mode) # debug_mode 전달

    if embedding1 is None:
        return f"검증 실패: '{image_path1}'에서 유효한 얼굴을 찾을 수 없습니다."
    if embedding2 is None:
        return f"검증 실패: '{image_path2}'에서 유효한 얼굴을 찾을 수 없습니다."

    similarity = cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))[0][0]

    if similarity >= threshold:
        return f"동일 인물입니다! (유사도: {similarity:.4f})"
    else:
        return f"다른 인물입니다. (유사도: {similarity:.4f})"

# --- 6. MTCNN을 이용한 얼굴 크롭 및 저장 함수 (선택 사항 - 필요한 경우에만 실행) ---
# 이 함수는 얼굴 이미지를 별도로 저장하고 싶을 때 사용합니다.
# `get_face_embedding` 함수는 이미 MTCNN을 내부적으로 사용하므로, 
# 이 `crop_and_save_faces` 함수를 매번 실행할 필요는 없습니다.
def crop_and_save_faces_batch(input_dir, output_dir, margin=20, image_size=160):
    mtcnn_batch = MTCNN(keep_all=False, margin=margin, image_size=image_size, post_process=False, device=device)

    print(f"\n[INFO] '{input_dir}'에서 얼굴 크롭 및 '{output_dir}'에 저장 시작...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(input_dir):
        for file in tqdm(files, desc=f"처리 중: {root}"):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                input_path = os.path.join(root, file)
                
                relative_path = os.path.relpath(root, input_dir)
                output_folder = os.path.join(output_dir, relative_path)
                os.makedirs(output_folder, exist_ok=True)
                
                output_path = os.path.join(output_folder, file)
                
                try:
                    image = Image.open(input_path).convert('RGB')
                    face_cropped_tensor = mtcnn_batch(image) # MTCNN은 텐서를 반환

                    if face_cropped_tensor is not None:
                        # 텐서를 PIL Image로 변환하여 저장
                        face_image_pil = to_pil_image(face_cropped_tensor.cpu())
                        face_image_pil.save(output_path)
                    else:
                        # print(f"[WARNING] '{input_path}'에서 얼굴을 찾을 수 없어 저장하지 않습니다.")
                        pass # 얼굴 검출 실패 시
                except Exception as e:
                    print(f"[ERROR] 처리 중 오류 발생: {input_path}, 오류: {e}")
    print(f"[INFO] 얼굴 크롭 및 저장 완료.")

# --- 메인 실행 블록 ---
if __name__ == '__main__':
    image1_same_person = '비교/8.jpg'
    image2_same_person = '비교/12.jpg'

    print(f"\n[INFO] '{image1_same_person}'와 '{image2_same_person}' 동일 인물 검증 시도...")
    result_same = verify_faces(image1_same_person, image2_same_person, threshold=0.5)
    print(f"검증 결과: {result_same}")