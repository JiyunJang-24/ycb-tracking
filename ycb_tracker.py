import cv2
import torch
import imageio
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import os
import matplotlib.pyplot as plt

from base64 import b64encode
# from cotracker.utils.visualizer import Visualizer, read_video_from_path
from IPython.display import HTML

def select_points_interactively(img):
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow('Image', img)

    print("▶ 마우스로 포인트를 클릭하세요. 다 선택했으면 Enter 키를 누르세요.")
    cv2.imshow('Image', img)
    cv2.setMouseCallback('Image', click_event)

    while True:
        key = cv2.waitKey(0)
        if key == 13 or key == 10:  # Enter key: Windows(13), Linux/Mac(10)
            break

    cv2.destroyAllWindows()
    print(f"선택된 포인트: {points}")
    return np.array(points)

def show_video(video_path):
    video_file = open(video_path, "r+b").read()
    video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
    return HTML(f"""<video width="640" height="480" autoplay loop controls><source src="{video_url}"></video>""")

def track_with_cotracker(frames, points):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    points_tensor = torch.tensor(points)[None, None].float().to(device)
    # pred_tracks, _ = cotracker(video, query=points_tensor)
    pred_tracks, pred_visibility = cotracker(video, grid_size=30)
    return pred_tracks[0].cpu().numpy()  # (T, N, 2)


def convert_to_3d(points_2d, depth, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    points_3d = []
    for (x, y) in points_2d:
        x, y = int(round(x)), int(round(y))
        if y >= depth.shape[0] or x >= depth.shape[1]:
            points_3d.append([np.nan]*3)
            continue
        z = depth[y, x] / 1000.0
        if z == 0 or np.isnan(z):
            points_3d.append([np.nan]*3)
        else:
            X = (x - cx) * z / fx
            Y = (y - cy) * z / fy
            points_3d.append([X, Y, z])
    return np.array(points_3d)


def estimate_orientations(tracks_3d):
    Rs = []
    ref = tracks_3d[0]
    for t in range(1, len(tracks_3d)):
        tgt = tracks_3d[t]
        if np.isnan(tgt).any():
            Rs.append(np.eye(3))
            continue
        mu_ref, mu_tgt = ref.mean(0), tgt.mean(0)
        H = (ref - mu_ref).T @ (tgt - mu_tgt)
        U, _, Vt = np.linalg.svd(H)
        R_t = Vt.T @ U.T
        if np.linalg.det(R_t) < 0:
            Vt[2] *= -1
            R_t = Vt.T @ U.T
        Rs.append(R_t)
    return Rs


def draw_axis(img, K, R, t, length=0.05):
    axis_pts = np.array([[0,0,0],[length,0,0],[0,length,0],[0,0,length]])
    pts = R @ axis_pts.T + t[:, None]
    pts = K @ pts
    pts = pts[:2] / pts[2:]
    pts = pts.T.astype(int)
    origin = tuple(pts[0])
    img = cv2.line(img, origin, tuple(pts[1]), (0,0,255), 2)
    img = cv2.line(img, origin, tuple(pts[2]), (0,255,0), 2)
    img = cv2.line(img, origin, tuple(pts[3]), (255,0,0), 2)
    return img


def main():
    # 1. Load YCB sequence
    seq_dir = Path('~/Desktop/JY/ycb/data/0001').expanduser()
    frame_ids = sorted([int(f.name[:6]) for f in seq_dir.glob('*-color.jpg')])
    frames = [cv2.imread(str(seq_dir / f"{fid:06d}-color.jpg")) for fid in frame_ids][:10]
    depths = [cv2.imread(str(seq_dir / f"{fid:06d}-depth.png"), -1) for fid in frame_ids][:10]
    # bboxs = [np.loadtxt(str(seq_dir / f"{fid:06d}-box.txt")) for fid in frame_ids][:10]
    # 2. Load intrinsics
    K = np.array([[1066.778, 0, 312.9869],
                  [0, 1067.487, 241.3109],
                  [0, 0, 1]])

    # 3. Select points from first frame
    import pdb; pdb.set_trace()
    init_points = select_points_interactively(frames[0].copy())
    print("start tracking with COTRACKER")
    # 4. Track
    tracks_2d = track_with_cotracker(frames, init_points)  # (T, N, 2)

    # 5. Convert to 3D
    tracks_3d = []
    for t in range(len(frames)):
        pts3d = convert_to_3d(tracks_2d[t], depths[t], K)
        tracks_3d.append(pts3d)
    tracks_3d = np.array(tracks_3d)  # (T, N, 3)

    # 6. Estimate orientations
    Rs = estimate_orientations(tracks_3d)

    # 7. Visualize
    out = cv2.VideoWriter('pose_track_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (frames[0].shape[1], frames[0].shape[0]))
    for i in range(1, len(frames)):
        img = frames[i].copy()
        img = draw_axis(img, K, Rs[i-1], np.zeros(3))
        out.write(img)
    out.release()
    print("Saved to pose_track_output.mp4")


def track_with_cotracker_pairwise(frame0, frame1, init_points):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    video = torch.tensor([frame0, frame1]).permute(0, 3, 1, 2)[None].float().to(device)
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    points_tensor = torch.tensor(init_points)[None, None].float().to(device)
    # pred_tracks, _ = cotracker(video, query=points_tensor)
    pred_tracks, _ = cotracker(video, grid_size=30)
    return pred_tracks[0].cpu().numpy()  # shape: (2, N, 2)

def main_pairwise():
    # 1. Load YCB sequence
    seq_dir = Path('~/Desktop/JY/ycb/data/0001').expanduser()
    frame_ids = sorted([int(f.name[:6]) for f in seq_dir.glob('*-color.jpg')])
    frames = [cv2.imread(str(seq_dir / f"{fid:06d}-color.jpg")) for fid in frame_ids][:10]
    depths = [cv2.imread(str(seq_dir / f"{fid:06d}-depth.png"), -1) for fid in frame_ids][:10]

    # 2. Load intrinsics
    K = np.array([[1066.778, 0, 312.9869],
                  [0, 1067.487, 241.3109],
                  [0, 0, 1]])

    # 3. Select points from first frame
    init_points = select_points_interactively(frames[0].copy())

    tracks_3d = []
    Rs = []

    # 4. Pairwise track and visualize
    out = cv2.VideoWriter('pose_track_pairwise.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5, (frames[0].shape[1], frames[0].shape[0]))

    curr_points = init_points
    for i in range(len(frames) - 1):
        f0, f1 = frames[i], frames[i + 1]
        d0, d1 = depths[i], depths[i + 1]

        # Track from f0 → f1
        tracked = track_with_cotracker_pairwise(f0, f1, curr_points)
        curr_points = tracked[1]  # update tracked points

        # Convert to 3D
        p3d_0 = convert_to_3d(tracked[0], d0, K)
        p3d_1 = convert_to_3d(tracked[1], d1, K)

        if np.isnan(p3d_1).any():
            R_t = np.eye(3)
        else:
            mu0, mu1 = p3d_0.mean(0), p3d_1.mean(0)
            H = (p3d_0 - mu0).T @ (p3d_1 - mu1)
            U, _, Vt = np.linalg.svd(H)
            R_t = Vt.T @ U.T
            if np.linalg.det(R_t) < 0:
                Vt[2] *= -1
                R_t = Vt.T @ U.T

        Rs.append(R_t)

        # Draw axis
        img_with_axis = draw_axis(f1.copy(), K, R_t, np.zeros(3))
        out.write(img_with_axis)

        # Optional: show intermediate results
        cv2.imshow("Tracked Frame", img_with_axis)
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break

    out.release()
    cv2.destroyAllWindows()
    print("Saved to pose_track_pairwise.mp4")


if __name__ == "__main__":
    main_pairwise()
