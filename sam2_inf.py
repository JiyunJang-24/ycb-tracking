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
# checkpoint_path = "sam2/checkpoints/sam2.1_hiera_large.pt"
# config_path     = "sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
config_path = os.path.abspath("sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml")
checkpoint_path = os.path.abspath("sam2/checkpoints/sam2.1_hiera_large.pt")
# checkpoint_path = "/home/fick17/Desktop/JY/FoundationPose/object_image/sam2/sam2/checkpoints/sam2.1_hiera_large.pt"
# config_path = "/home/fick17/Desktop/JY/FoundationPose/object_image/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"

device = "cuda" if torch.cuda.is_available() else "cpu"
sam_model = build_sam2(config_path, checkpoint_path).to(device)
predictor = SAM2ImagePredictor(sam_model)

# ===== 경로 설정 =====
input_dir = "./"
output_dir = "./mask"
os.makedirs(output_dir, exist_ok=True)

# ===== 이미지 파일 로딩 =====
img_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

# ===== 마스크 생성 =====
for fname in tqdm(img_files):
    img_path = os.path.join(input_dir, fname)
    image = Image.open(img_path).convert("RGB")
    image_np = np.array(image)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image_np)
        h, w = image_np.shape[:2]
        center_point = [[w // 2, h // 2]]  # 중앙 기준 포인트
        input_prompts = {"point_coords": center_point, "point_labels": [1]}

        masks, _, _ = predictor.predict(input_prompts)
        if masks.shape[0] == 0:
            print(f"❌ 마스크 없음: {fname}")
            continue

        mask = masks[0].astype(np.uint8) * 255
        out_path = os.path.join(output_dir, fname.replace(".jpg", ".png"))
        cv2.imwrite(out_path, mask)

print("✅ SAM2 마스크 저장 완료")
