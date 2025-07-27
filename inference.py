import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import sys
sys.path.append("sam2")  # sam2 디렉토리 내부 모듈 import

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ===== 설정 =====
checkpoint_path = "./checkpoints/sam2.1_hiera_large.pt"
config_path     = "configs/sam2.1/sam2.1_hiera_l.yaml"
# config_path = os.path.abspath("sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml")
# checkpoint_path = os.path.abspath("sam2/checkpoints/sam2.1_hiera_large.pt")
# checkpoint_path = "/home/fick17/Desktop/JY/FoundationPose/object_image/sam2/sam2/checkpoints/sam2.1_hiera_large.pt"
# config_path = "/home/fick17/Desktop/JY/FoundationPose/object_image/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"

device = "cuda" if torch.cuda.is_available() else "cpu"
sam_model = build_sam2(config_path, checkpoint_path).to(device)
predictor = SAM2ImagePredictor(sam_model)

# ===== 경로 설정 =====
input_dir = "/home/fick17/Desktop/JY/ycb/co-tracker/assets/video_front_video.mp4"
output_dir = "/home/fick17/Desktop/JY/ycb/sam2/mask"
os.makedirs(output_dir, exist_ok=True)
# ===== resize 설정 =====
target_width = 1024
target_height = 1024
# ===== 이미지 파일 로딩 =====
img_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

# ===== 마우스 포인트 저장 =====
clicked_points = []
def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append([x, y])
        cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Click multiple points (ENTER=OK, ESC=skip)", param)

# ===== 메인 루프 =====
img_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

for fname in img_files:
    img_path = os.path.join(input_dir, fname)
    image_pil = Image.open(img_path).convert("RGB")
    image_np_orig = np.array(image_pil)
    orig_h, orig_w = image_np_orig.shape[:2]

    # resize for viewing & mask prediction
    # image_np = cv2.resize(image_np_orig, (target_width, target_height), interpolation=cv2.INTER_AREA)
    image_np = image_np_orig.copy()
    display_img = image_np.copy()

    scale_x = orig_w / target_width
    scale_y = orig_h / target_height

    clicked_points = []

    # 이미지 띄우고 클릭
    cv2.imshow("Click multiple points (ENTER=OK, ESC=skip)", display_img)
    cv2.setMouseCallback("Click multiple points (ENTER=OK, ESC=skip)", mouse_callback, param=display_img)

    while True:
        key = cv2.waitKey(1)
        if key == 13:  # Enter
            break
        elif key == 27:  # ESC
            print(f"⏩ 건너뜀: {fname}")
            clicked_points = []
            break

    cv2.destroyAllWindows()
    if len(clicked_points) == 0:
        continue

    # 포인트 변환 (resize back to model input size)
    scaled_points = [[int(x), int(y)] for x, y in clicked_points]
    point_coords = np.array(scaled_points)
    point_labels = np.ones(len(point_coords))  # all positive points

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image_np)
        masks, _, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels, multimask_output=False)

        if masks.shape[0] == 0:
            print(f"❌ 마스크 없음: {fname}")
            continue

        mask_resized = cv2.resize(masks[0].astype(np.uint8) * 255, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        out_path = os.path.join(output_dir, fname.replace(".jpg", ".png"))
        cv2.imwrite(out_path, mask_resized)
        print(f"✅ 마스크 저장됨: {out_path}")