import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import time
import threading
import numpy as np
import warnings

# FaceNet 관련 라이브러리
import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from facenet_pytorch import InceptionResnetV1, MTCNN

# 경고 메시지 무시
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")
warnings.filterwarnings("ignore", category=FutureWarning, module="facenet_pytorch")

# --- 1. FaceNet 모델 및 설정 (기존 코드 기반) ---
class FaceComparer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] 사용 디바이스: {self.device}")

        # MTCNN 설정 (얼굴 검출)
        self.mtcnn = MTCNN(keep_all=False, margin=40, image_size=160, post_process=False, device=self.device)

        # FaceNet ResNet 모델 설정 (임베딩 추출)
        self.model = InceptionResnetV1(pretrained='vggface2', classify=False).to(self.device)
        save_path = 'facenet_model.pth'
        try:
            self.model.load_state_dict(torch.load(save_path, map_location=self.device))
            print(f"[INFO] 모델 가중치를 '{save_path}'에서 성공적으로 로드했습니다.")
        except FileNotFoundError:
            print(f"[ERROR] '{save_path}' 파일을 찾을 수 없습니다. 프로그램 실행에 문제가 발생할 수 있습니다.")
            # GUI 앱에서는 exit() 대신 사용자에게 알림
            # messagebox.showerror("오류", f"모델 파일 '{save_path}'를 찾을 수 없습니다.")
            # raise FileNotFoundError(f"모델 파일 '{save_path}'를 찾을 수 없습니다.")
        self.model.eval()

        # 이미지 전처리 설정
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            lambda x: (x - 0.5) / 0.5
        ])

    def get_face_embedding(self, image):
        """ PIL Image를 입력받아 얼굴 임베딩을 반환 """
        if image.mode != 'RGB':
            image = image.convert('RGB')

        face_cropped = self.mtcnn(image)
        if face_cropped is None:
            return None, None

        # 얼굴 부분만 잘라내기 (좌표값 필요)
        boxes, _ = self.mtcnn.detect(image)
        if boxes is None or len(boxes) == 0:
            return None, None
        box = boxes[0]

        processed_face = self.transform(to_pil_image(face_cropped.cpu()))
        processed_face = processed_face.unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(processed_face)
        return embedding.squeeze(0).cpu(), box

    def compare_embeddings(self, emb1, emb2):
        """ 두 임베딩 간의 코사인 유사도 계산 """
        if emb1 is None or emb2 is None:
            return 0.0
        return torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

# --- 2. GUI 애플리케이션 ---
class FaceComparisonApp:
    def __init__(self, root, title):
        self.root = root
        self.root.title(title)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 모델 로더 초기화
        try:
            self.comparer = FaceComparer()
        except Exception as e:
            messagebox.showerror("초기화 오류", f"모델 로딩 중 심각한 오류 발생: {e}")
            self.root.destroy()
            return

        # 상태 변수
        self.reference_image = None
        self.reference_embedding = None
        self.is_running = False
        self.video_thread = None

        self.reference_image_folder = "레퍼런스 이미지"
        self.reference_image_list = self.get_reference_image_list()

        # UI 구성
        self.create_widgets()

        # 웹캠 초기화
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("웹캠 오류", "웹캠을 열 수 없습니다. 연결 상태를 확인하세요.")
            self.is_running = False

    def get_reference_image_list(self):
        if not os.path.isdir(self.reference_image_folder):
            messagebox.showwarning("폴더 없음", f"'{self.reference_image_folder}' 폴더를 찾을 수 없습니다.")
            return ["폴더 없음"]
        
        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
        images = [f for f in os.listdir(self.reference_image_folder) if f.lower().endswith(supported_formats)]
        return images if images else ["이미지 없음"]

    def create_widgets(self):
        # --- 메인 프레임 ---
        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # --- 왼쪽 프레임 (기준 이미지) ---
        left_frame = tk.Frame(main_frame, bd=2, relief=tk.GROOVE)
        left_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.BOTH, expand=True)

        tk.Label(left_frame, text="기준 이미지", font=("Helvetica", 14)).pack(pady=5)
        self.ref_image_label = tk.Label(left_frame, text="아래에서 이미지를 선택하세요", bg="lightgrey")
        self.ref_image_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # 드롭다운 메뉴 추가
        tk.Label(left_frame, text="이미지 선택:").pack(pady=(10, 0))
        self.selected_image_var = tk.StringVar(self.root)
        if self.reference_image_list:
            self.selected_image_var.set(self.reference_image_list[0]) # 기본값 설정
        
        self.dropdown = tk.OptionMenu(left_frame, self.selected_image_var, *self.reference_image_list, command=self.on_reference_image_selected)
        self.dropdown.pack(pady=5)


        # --- 오른쪽 프레임 (웹캠) ---
        right_frame = tk.Frame(main_frame, bd=2, relief=tk.GROOVE)
        right_frame.pack(side=tk.RIGHT, padx=5, pady=5, fill=tk.BOTH, expand=True)

        tk.Label(right_frame, text="실시간 웹캠", font=("Helvetica", 14)).pack(pady=5)
        self.webcam_label = tk.Label(right_frame, text="비교 시작을 누르세요", bg="lightgrey")
        self.webcam_label.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.result_label = tk.Label(right_frame, text="결과: 대기 중", font=("Helvetica", 12, "bold"))
        self.result_label.pack(pady=10)

        # --- 하단 프레임 (컨트롤) ---
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(padx=10, pady=5, fill=tk.X)

        self.btn_toggle_run = tk.Button(bottom_frame, text="비교 시작", command=self.toggle_run, width=15, height=2)
        self.btn_toggle_run.pack(side=tk.LEFT, expand=True, padx=5)

        self.btn_quit = tk.Button(bottom_frame, text="종료", command=self.on_closing, width=15, height=2)
        self.btn_quit.pack(side=tk.RIGHT, expand=True, padx=5)

    def on_reference_image_selected(self, selected_image_name):
        if selected_image_name in ["폴더 없음", "이미지 없음"]:
            self.reference_image = None
            self.reference_embedding = None
            self.ref_image_label.config(image='', text="이미지를 선택하세요")
            return

        file_path = os.path.join(self.reference_image_folder, selected_image_name)
        
        try:
            # 이미지 로드 및 임베딩 추출
            self.reference_image = Image.open(file_path)
            embedding, _ = self.comparer.get_face_embedding(self.reference_image)

            if embedding is None:
                messagebox.showwarning("얼굴 검출 실패", "기준 이미지에서 얼굴을 찾을 수 없습니다. 다른 이미지를 선택해주세요.")
                self.reference_image = None
                self.reference_embedding = None
                self.ref_image_label.config(image='', text="얼굴 검출 실패")
                return

            self.reference_embedding = embedding
            print(f"[INFO] 기준 이미지 '{selected_image_name}' 로드 및 임베딩 생성 완료.")

            # GUI에 이미지 표시
            self.display_image(self.reference_image.copy(), self.ref_image_label)

        except Exception as e:
            messagebox.showerror("오류", f"이미지 처리 중 오류 발생: {e}")
            self.reference_image = None
            self.reference_embedding = None
            self.ref_image_label.config(image='', text="오류 발생")

    def display_image(self, img, label_widget, max_size=(400, 400)):
        img.thumbnail(max_size)
        photo = ImageTk.PhotoImage(image=img)
        label_widget.config(image=photo, text="")
        label_widget.image = photo # 참조 유지

    def toggle_run(self):
        if self.is_running:
            self.is_running = False
            self.btn_toggle_run.config(text="비교 시작")
            if self.video_thread:
                self.video_thread.join() # 스레드가 끝날 때까지 기다림
            self.webcam_label.config(text="비교 시작을 누르세요", bg="lightgrey", image='')

        else:
            if self.reference_embedding is None:
                messagebox.showwarning("경고", "먼저 기준 이미지를 불러와주세요.")
                return
            if not self.cap.isOpened():
                messagebox.showerror("웹캠 오류", "웹캠이 연결되어 있지 않습니다.")
                return

            self.is_running = True
            self.btn_toggle_run.config(text="비교 중지")
            self.video_thread = threading.Thread(target=self.video_loop)
            self.video_thread.daemon = True
            self.video_thread.start()

    def video_loop(self):
        threshold = 0.6 # 유사도 임계값
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print("[ERROR] 웹캠 프레임 읽기 실패")
                time.sleep(0.1)
                continue

            # BGR -> RGB 변환
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # 얼굴 검출 및 임베딩 추출
            webcam_embedding, box = self.comparer.get_face_embedding(pil_image)

            result_text = "얼굴 감지 안됨"
            color = (0, 0, 255) # 빨간색 (기본)

            if webcam_embedding is not None and self.reference_embedding is not None:
                similarity = self.comparer.compare_embeddings(self.reference_embedding, webcam_embedding)
                if similarity > threshold:
                    result_text = f"동일 인물 (유사도: {similarity:.2f})"
                    color = (0, 255, 0) # 초록색
                else:
                    result_text = f"다른 인물 (유사도: {similarity:.2f})"
                    color = (255, 0, 0) # 파란색

                # 감지된 얼굴에 사각형 그리기
                if box is not None:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 결과 텍스트 업데이트
            self.result_label.config(text=f"결과: {result_text}")

            # Tkinter 레이블에 웹캠 프레임 표시
            photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.webcam_label.config(image=photo)
            self.webcam_label.image = photo

            time.sleep(0.03) # 루프 지연

    def on_closing(self):
        if messagebox.askokcancel("종료", "프로그램을 종료하시겠습니까?"):
            self.is_running = False
            if self.video_thread and self.video_thread.is_alive():
                self.video_thread.join(timeout=1) # 스레드가 종료될 시간을 줌
            if self.cap.isOpened():
                self.cap.release()
            self.root.destroy()

# --- 3. 애플리케이션 실행 ---
if __name__ == '__main__':
    from PIL import Image, ImageTk
    # Tkinter의 PhotoImage가 PIL을 필요로 하므로, PIL이 설치되지 않았다면 에러가 발생할 수 있습니다.
    # 이 부분은 사용자가 PIL을 설치했다고 가정합니다.
    
    root = tk.Tk()
    app = FaceComparisonApp(root, "실시간 얼굴 비교 프로그램")
    root.mainloop()
