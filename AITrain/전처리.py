import os
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import MTCNN
import numpy as np

def crop_and_save_faces(input_dir, output_dir, margin=20, image_size=160):
    mtcnn = MTCNN(keep_all=False, margin=margin, image_size=image_size, post_process=False)

    for root, dirs, files in os.walk(input_dir):
        for file in tqdm(files):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                input_path = os.path.join(root, file)
                
                relative_path = os.path.relpath(root, input_dir)
                output_folder = os.path.join(output_dir, relative_path)
                os.makedirs(output_folder, exist_ok=True)
                
                output_path = os.path.join(output_folder, file)
                
                try:
                    image = Image.open(input_path).convert('RGB')
                    face = mtcnn(image)
                    
                    if face is not None:
                        face_np = face.permute(1, 2, 0).byte().cpu().numpy()
                        face_image = Image.fromarray(face_np)
                        face_image.save(output_path)
                    else:
                        pass  # 얼굴 검출 실패 시
                except Exception as e:
                    print(f"처리 중 오류 발생: {input_path}, 오류: {e}")




if __name__ == "__main__":
    input_directory = "dataset"    # 원본 이미지 폴더
    output_directory = "dataset_after"  # 전처리 후 저장 폴더

    crop_and_save_faces(input_directory, output_directory)
