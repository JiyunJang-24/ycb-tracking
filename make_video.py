import os
import cv2

def images_to_video(image_dir, output_path, fps=15, ext='.jpg'):
    # 1) 폴더 내 이미지 파일 목록 정렬
    imgs = sorted([f for f in os.listdir(image_dir) if f.endswith(ext)])
    if not imgs:
        raise ValueError(f"No images with extension {ext} in {image_dir}")

    # 2) 첫 번째 이미지로 프레임 크기 가져오기
    first = cv2.imread(os.path.join(image_dir, imgs[0]))
    h, w = first.shape[:2]

    # 3) VideoWriter 초기화 (H.264 인코딩)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # 4) 이미지 순서대로 쓰기
    for fn in imgs:
        img = cv2.imread(os.path.join(image_dir, fn))
        writer.write(img)
    writer.release()
    print(f"Saved video: {output_path}")

if __name__ == "__main__":
    BASE = "/home/fick17/Desktop/JY/ycb/data"
    SEQS = sorted([d for d in os.listdir(BASE) if os.path.isdir(os.path.join(BASE, d))])
    # 예: 0001 시퀀스만 동영상으로 만들려면 리스트를 ['0001'] 로 바꾸세요.
    for seq in SEQS:
    # seq = SEQS[0]
        img_dir = os.path.join(BASE, seq)
        out_file = os.path.join(BASE, f"{seq}.mp4")
        images_to_video(img_dir, out_file, fps=15, ext='.jpg')
