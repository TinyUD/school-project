"""
CNN 기반 얼굴 인식 보안 프로그램
---------------------------------------
1) 이미지 학습을 위한 데이터
: Original Images 폴더의 사람별 이미지 학습
- Original Images 폴더 안에는 인물의 이름별로 폴더가 존재
- 해당 폴더 안에 인물 얼굴 학습을 위한 여러 이미지 파일이 존재
- 여기에 특정한 다른 인물의 새로운 폴더를 만들어 학습하도록 세팅할 수 있음.
(!!!) Original Images에는 학습하고자 하는 사람의 얼굴은 반드시 들어가야 함 (!!!)   

2) 학습 원리
: Haar Cascade를 이용해 얼굴 ROI(관심 영역)를 자동으로 잘라서 CNN 학습
- 일반 사진에서 얼굴 부분을 찾아내는 데 활용되는 Haar Cascade 알고리즘을 이용
  -> 일단은 학습할 사진에서 얼굴부분을 먼저 찾아냄.
- 찾은 얼굴 부분을 인공지능 알고리즘 중 이미지 학습에 특히 적합한
  딥러닝 알고리즘 구현을 위해 합성곱 신경망(CNN)을 이용해서 학습


3) 학습된 모델로 테스트
: 학습된 모델을 즉시 Faces 폴더나 카메라로 테스트 가능
- 이미 사람들의 얼굴을 학습했다면, 그 사람의 다른 사진으로 테스트를 했을 때
  인식이 제대로 되어야 한다. 이를 위해서 테스트를 위한 이미지가 Faces 폴더에 있음.
  : 먼저 이 Faces 모델에 있는 이미지들을 기존에 학습한 사람들의 이미지와 대조해서,
    어떤 사람에 해당하는지 혹은 해당하지 않는지를 판별할 수 있다.

- 프로젝트 실제 참여자의 얼굴을 학습하려면, Original Images 폴더 안에 프로젝트
  참여자의 이미지를 충분히 넣어 먼저 학습을 진행해야 한다.
  : 학습이 완료된 후에는 카메라를 통한 인식을 수행해서 실시간 얼굴 인식을
    구현하고자 함.
  : Faces 폴더에 학습한 사람의 얼굴 이미지를 넣으면 사진을 통한 테스트 가능
  : Faces 폴더에 별도의 테스트 이미지를 넣지 않더라도 실시간 비디오를 통해서
    얼굴 인식이 별도로 가능하게 세팅하고자 함.
"""
"""
CNN 기반 얼굴 인식 보안 프로그램 (개선 버전)
---------------------------------------
- Original Images 폴더의 사람별 이미지 학습
- Haar Cascade 및 DNN을 이용한 얼굴 추출
- MobileNetV2 기반 CNN으로 얼굴 분류
- Faces 폴더 또는 웹캠을 통해 실시간 테스트 가능
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import PIL.Image
from tensorflow.keras import layers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from joblib import dump, load

# ----------------------------------------
# 0) DNN 얼굴 검출 모델 로드
# ----------------------------------------
try:
    face_net = cv2.dnn.readNetFromCaffe(
        "deploy.prototxt",
        "res10_300x300_ssd_iter_140000.caffemodel"
    )
    layer_names = face_net.getLayerNames()
    if len(layer_names) == 0:
        raise ValueError("레이어가 0개입니다. 모델이 손상되었을 수 있음.")
    print(f"[OK] DNN 모델 로드 성공 / 레이어 수: {len(layer_names)}")
except cv2.error as e:
    print(f"[ERROR] DNN 모델 로드 실패: {e}")
    sys.exit(1)

aligned_dir = "faces_aligned"

# ----------------------------------------
# 얼굴 학습 함수
# ----------------------------------------
def train_faces():
    src_dir = "Original Images"
    need_crop = False

    # faces_aligned 폴더 확인
    if os.path.exists(aligned_dir):
        has_images = any(
            os.path.isfile(os.path.join(root, file))
            for root, _, files in os.walk(aligned_dir) for file in files
        )
        if not has_images:
            print("[WARN] faces_aligned 폴더는 있지만 파일이 없으므로 다시 crop 시작합니다.")
            need_crop = True
        else:
            print("[INFO] faces_aligned 폴더와 이미지가 존재, 그대로 사용합니다.")
    else:
        os.makedirs(aligned_dir)
        need_crop = True

    # 얼굴 Crop 작업
    if need_crop:
        try:
            for person in os.listdir(src_dir):
                person_dir = os.path.join(src_dir, person)
                save_dir = os.path.join(aligned_dir, person)
                os.makedirs(save_dir, exist_ok=True)

                for file in os.listdir(person_dir):
                    img_path = os.path.join(person_dir, file)
                    img = cv2.imread(img_path)

                    if img is None:
                        print(f"[ERROR] 이미지 로드 실패: {img_path}")
                        raise RuntimeError("faces_aligned 생성 중 이미지 로드 실패")

                    if len(img.shape) != 3:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    elif img.shape[2] == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                    if img.shape[0] < 50 or img.shape[1] < 50:
                        print(f"[WARN] 너무 작은 이미지 {img.shape}, skip")
                        continue

                    h, w = img.shape[:2]
                    blob = cv2.dnn.blobFromImage(img, 1.0, (300,300),
                                                 (104.0, 177.0, 123.0))

                    if blob is None or blob.size == 0:
                        print(f"[ERROR] blob empty {img_path}")
                        raise RuntimeError("faces_aligned 생성 중 blob empty")

                    print(f"[DEBUG] blob.shape={blob.shape}, min={blob.min()}, max={blob.max()}, mean={blob.mean()}")
                    face_net.setInput(blob)

                    try:
                        detections = face_net.forward()
                    except cv2.error as e:
                        print(f"[ERROR] forward 실패 {img_path}, 이유: {e}")
                        raise RuntimeError("faces_aligned 생성 중 forward 실패")

                    for i in range(detections.shape[2]):
                        confidence = detections[0,0,i,2]
                        if confidence < 0.5:
                            continue
                        box = detections[0,0,i,3:7] * np.array([w,h,w,h])
                        (x1,y1,x2,y2) = box.astype("int")
                        x1, y1 = max(0,x1), max(0,y1)
                        x2, y2 = min(w-1,x2), min(h-1,y2)

                        face = img[y1:y2, x1:x2]
                        if face.shape[0] < 50 or face.shape[1] < 50:
                            continue

                        face_resized = cv2.resize(face, (128,128))
                        save_path = os.path.join(save_dir,
                                                 f"{file.split('.')[0]}_{i}.jpg")
                        cv2.imwrite(save_path, face_resized)
            print("[INFO] faces_aligned 얼굴 crop 완료")
        except RuntimeError as e:
            print(f"[FATAL] {e}")
            sys.exit(1)

    # 데이터 증강 (완화된 설정)
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True
    )

    train_gen = datagen.flow_from_directory(
        aligned_dir,
        target_size=(128,128),
        batch_size=16,
        class_mode='categorical',
        subset='training'
    )
    val_gen = datagen.flow_from_directory(
        aligned_dir,
        target_size=(128,128),
        batch_size=16,
        class_mode='categorical',
        subset='validation'
    )

    # MobileNetV2 기반 모델 정의 (미세조정 포함)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(128,128,3),
        include_top=False,
        weights='imagenet'
    )
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(train_gen.num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # 학습 수행 (EarlyStopping 포함)
    es = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(train_gen, validation_data=val_gen, epochs=30, callbacks=[es])
    model.save("face_cnn_model.h5")
    print("[INFO] face_cnn_model.h5 저장 완료")

    dump(train_gen.class_indices, "label_indices.joblib")
    print("[INFO] label_indices.joblib 저장 완료")

# 메인 메뉴
if __name__ == "__main__":
    print("==== CNN 얼굴 보안 시스템 ====")
    print("1. 학습")
    print("2. 얼굴 판별 (Faces 폴더)")
    print("3. 얼굴 판별 (카메라)")
    choice = input("선택 (1/2/3): ")
    if choice == '1':
        train_faces()
    elif choice == '2':
        print("[TODO] recognize_from_faces_folder ")
    elif choice == '3':
        print("[TODO] recognize_from_camera 함수 미완성")
    else:
        print("잘못된 선택입니다.")
