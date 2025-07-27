import os
import cv2
import numpy as np
from PIL import Image
import torch
import sys

sys.path.append("sam2")

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ===== 설정 =====
checkpoint_path = "./checkpoints/sam2.1_hiera_large.pt"
config_path     = "configs/sam2.1/sam2.1_hiera_l.yaml"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam_model = build_sam2(config_path, checkpoint_path).to(device)
predictor = SAM2ImagePredictor(sam_model)

# ===== 비디오 경로 =====
video_path = "/home/fick17/Desktop/JY/ycb/co-tracker/assets/video_front_no_occ.mp4"
output_dir = "/home/fick17/Desktop/JY/ycb/sam2/mask"
os.makedirs(output_dir, exist_ok=True)

# ===== 비디오에서 첫 프레임 가져오기 =====
cap = cv2.VideoCapture(video_path)
success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("❌ 비디오에서 첫 프레임을 읽을 수 없습니다.")

# ===== 마우스 클릭으로 포인트 선택 =====
clicked_points = []

def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append([x, y])
        cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Click points (ENTER=OK)", param)

print("🖱️ 객체를 클릭하고 ENTER를 누르세요.")
display_img = frame.copy()
cv2.imshow("Click points (ENTER=OK)", display_img)
cv2.setMouseCallback("Click points (ENTER=OK)", mouse_callback, param=display_img)

while True:
    key = cv2.waitKey(1)
    if key == 13 or key == 10:  # Enter
        break
cv2.destroyAllWindows()

if len(clicked_points) == 0:
    print("❗ 클릭된 포인트가 없습니다.")
    exit()

# ===== SAM2 예측 =====
point_coords = np.array(clicked_points)
point_labels = np.ones(len(point_coords))  # all positive

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(frame)
    masks, _, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels, multimask_output=False)

if masks.shape[0] == 0:
    print("❌ 마스크가 생성되지 않았습니다.")
    exit()

mask = masks[0].astype(np.uint8) * 255
out_path = os.path.join(output_dir, "first_frame_mask.png")
cv2.imwrite(out_path, mask)
print(f"✅ 마스크 저장 완료: {out_path}")
