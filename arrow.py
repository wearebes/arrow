
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import os
from fractions import Fraction
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, periodogram, resample_poly, sosfiltfilt


VIDEO_PATH: Optional[str] = None  
FIXED_RADIUS_MM: float = 2.9
FILTER_BAND_HZ: Tuple[float, float] = (4.0, 6.0)
FILTER_ORDER: int = 4
TARGET_FS_HZ: float = 200.0 


CANNY_LOW: int = 30
CANNY_HIGH: int = 100
VOTE_THRESHOLD: int = 15

try:
    import tkinter as tk
    from tkinter import filedialog
    _TK_OK = True
except Exception:
    _TK_OK = False


def pick_video_path() -> str:
    if VIDEO_PATH and os.path.exists(VIDEO_PATH):
        return VIDEO_PATH

    if not _TK_OK:
        raise RuntimeError("tkinter 不可用，且 VIDEO_PATH 未设置。请在脚本顶部设置 VIDEO_PATH。")

    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select input video",
        filetypes=[("Video files", "*.avi *.mp4 *.mov *.mkv *.m4v"), ("All files", "*.*")]
    )
    root.destroy()
    if not path:
        raise RuntimeError("No video selected.")
    return path
# ROI 
def select_circle_roi(frame: np.ndarray) -> Tuple[Tuple[int, int], int]:
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    radius = 0
    mode: str = "new"  # "new" or "move"
    dragging = False

    def near_center(x: int, y: int, c: Tuple[int, int], tol: int = 12) -> bool:
        return (x - c[0])**2 + (y - c[1])**2 <= tol**2

    def mouse_callback(event, x, y, flags, param):
        nonlocal center, radius, mode, dragging
        if event == cv2.EVENT_LBUTTONDOWN:
            if radius > 0 and near_center(x, y, center):
                mode = "move"
                dragging = True
            else:
                mode = "new"
                dragging = True
                center = (x, y)
                radius = 0
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            if mode == "move":
                center = (x, y)
            else:
                radius = int(np.hypot(x - center[0], y - center[1]))
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
            if mode == "new":
                radius = int(np.hypot(x - center[0], y - center[1]))

    win = "Select Circular ROI"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        cv2.resizeWindow(win, w, h)
    except Exception:
        pass
    cv2.setMouseCallback(win, mouse_callback)

    while True:
        temp = frame.copy()
        cv2.putText(temp, "Drag to set radius. Drag center to move. Enter=OK, R=reset, ESC=Cancel",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # 圆心小点
        cv2.drawMarker(temp, center, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
        if radius > 0:
            cv2.circle(temp, center, radius, (0, 255, 0), 2)
        cv2.imshow(win, temp)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):  # Enter
            break
        if key in (ord('r'), ord('R')):
            radius = 0
        if key == 27:  # ESC
            cv2.destroyAllWindows()
            raise RuntimeError("ROI selection cancelled.")

    cv2.destroyAllWindows()
    if radius <= 0:
        raise RuntimeError("Invalid ROI radius.")
    return center, radius

# 刻度尺
def get_scale_factor_mm_per_px(frame: np.ndarray) -> float:
    h, w = frame.shape[:2]
    points: List[Tuple[int, int]] = []
    dragging_idx: Optional[int] = None

    def near_point(x: int, y: int, p: Tuple[int, int], tol: int = 12) -> bool:
        return (x - p[0])**2 + (y - p[1])**2 <= tol**2

    def mouse_callback(event, x, y, flags, param):
        nonlocal points, dragging_idx
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, p in enumerate(points):
                if near_point(x, y, p):
                    dragging_idx = i
                    return
            if len(points) < 2:
                points.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging_idx is not None:
                points[dragging_idx] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            dragging_idx = None

    win = "Set Scale (click/drag 2 points)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        cv2.resizeWindow(win, w, h)
    except Exception:
        pass
    cv2.setMouseCallback(win, mouse_callback)

    while True:
        disp = frame.copy()

        if len(points) >= 1:
            cv2.circle(disp, points[0], 4, (0, 0, 255), -1)
        if len(points) == 2:
            cv2.circle(disp, points[1], 4, (0, 0, 255), -1)
            cv2.line(disp, points[0], points[1], (0, 255, 0), 2)
            p0 = np.array(points[0], dtype=float)
            p1 = np.array(points[1], dtype=float)
            px_dist = float(np.linalg.norm(p0 - p1))
            mid = ((points[0][0] + points[1][0]) // 2, (points[0][1] + points[1][1]) // 2)
            cv2.putText(disp, f"{px_dist:.2f} px", (mid[0] + 10, mid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(disp, "Click/drag 2 points. Enter=OK  R=reset  ESC=Cancel",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow(win, disp)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):  # Enter
            break
        if key in (ord('r'), ord('R')):
            points = []
            dragging_idx = None
        if key == 27:  # ESC
            cv2.destroyAllWindows()
            raise RuntimeError("Scale selection cancelled.")

    cv2.destroyAllWindows()

    if len(points) != 2:
        raise RuntimeError("Need exactly 2 points for scale reference (after you press Enter).")
    p0 = np.array(points[0], dtype=float)
    p1 = np.array(points[1], dtype=float)
    px_dist = float(np.linalg.norm(p0 - p1))
    if px_dist <= 1e-9:
        raise RuntimeError("Scale points too close.")

    s = input("请输入这两点之间的真实距离 (mm): ").strip()
    if not s:
        raise RuntimeError("未输入真实距离(mm)。")
    real_mm = float(s)
    if real_mm <= 0:
        raise RuntimeError("真实距离(mm)必须为正数。")

    scale = real_mm / px_dist  # mm/px
    print(f"[Scale] pixel_distance={px_dist:.3f}px, real_distance={real_mm:.3f}mm -> scale={scale:.6f} mm/px")
    return scale
# 屏蔽矩形
def compute_canny(gray: np.ndarray, low: int, high: int) -> np.ndarray:
    return cv2.Canny(gray, low, high)

def select_ignored_rectangles(frame_bgr: np.ndarray, edges: np.ndarray) -> List[Tuple[int, int, int, int]]:
    h, w = edges.shape
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    ys, xs = np.where(edges > 0)
    base[ys, xs] = (255, 255, 255)

    rects: List[Tuple[int, int, int, int]] = []
    drawing = False
    x0, y0 = 0, 0
    current: Optional[Tuple[int, int, int, int]] = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, x0, y0, current, rects
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            x0, y0 = x, y
            current = (x0, y0, x0, y0)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            current = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            drawing = False
            if current is not None:
                x1, y1, x2, y2 = current
                if (x2 - x1) >= 5 and (y2 - y1) >= 5:
                    rects.append((x1, y1, x2, y2))
            current = None

    def render():
        img = base.copy()
        for (x1, y1, x2, y2) in rects:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if current is not None:
            x1, y1, x2, y2 = current
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(img, "Drag=mask. R=undo. Enter=finish.",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 255), 2)
        return img

    win = "Mask rectangles"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    try:
        cv2.resizeWindow(win, w, h)
    except Exception:
        pass
    cv2.setMouseCallback(win, mouse_callback)

    while True:
        cv2.imshow(win, render())
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10, 27, ord('q')):
            break
        if key in (ord('r'), ord('R')) and rects:
            rects.pop()

    cv2.destroyAllWindows()
    return rects


def apply_rect_masks(binary_img: np.ndarray, rects: List[Tuple[int, int, int, int]]) -> np.ndarray:
    out = binary_img.copy()
    for (x1, y1, x2, y2) in rects:
        out[y1:y2, x1:x2] = 0
    return out

# 投票 + SSE 选优
def sobel_grad(gray: np.ndarray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return gx, gy


def solve_center_voting_sse_fixed_radius(edges: np.ndarray,
                                        gray_for_grad: np.ndarray,
                                        fixed_radius_px: float,
                                        roi_box: Tuple[int, int, int, int],
                                        vote_threshold: int = 15,
                                        step_limit: int = 250,
                                        max_edge_points: int = 8000) -> Optional[Dict[str, object]]:
    """
    1) 用边缘点梯度方向对 ROI 内候选圆心投票
    2) 取 vote > vote_threshold 的像素点作为候选集合
    3) 在候选集合里，用 SSE(c)=Σ(||p-c||-r)^2 选最小者作为圆心
    """
    h, w = edges.shape
    x_min, y_min, x_max, y_max = roi_box
    x_min = max(0, x_min); y_min = max(0, y_min)
    x_max = min(w, x_max); y_max = min(h, y_max)
    if x_min >= x_max or y_min >= y_max:
        return None

    ys, xs = np.where(edges > 0)
    if xs.size < 30:
        return None

    m = (xs >= x_min) & (xs < x_max) & (ys >= y_min) & (ys < y_max)
    xs = xs[m]; ys = ys[m]
    if xs.size < 30:
        return None

   
    if xs.size > max_edge_points:
        idx = np.random.choice(xs.size, size=max_edge_points, replace=False)
        xs = xs[idx]; ys = ys[idx]

    gx, gy = sobel_grad(gray_for_grad)
    vote_map = np.zeros((h, w), dtype=np.int32)


    for x, y in zip(xs, ys):
        gxi = float(gx[y, x]); gyi = float(gy[y, x])
        norm = (gxi*gxi + gyi*gyi) ** 0.5
        if norm < 1e-9:
            continue
        ux, uy = gxi / norm, gyi / norm

        for direction in (+1.0, -1.0):
            cx_f, cy_f = float(x), float(y)
            steps = 0
            while True:
                cx_f += direction * ux
                cy_f += direction * uy
                cx = int(round(cx_f)); cy = int(round(cy_f))
                if cx < x_min or cx >= x_max or cy < y_min or cy >= y_max:
                    break
                vote_map[cy, cx] += 1
                steps += 1
                if steps >= step_limit:
                    break

    cand = np.argwhere(vote_map > vote_threshold)  # [[cy,cx],...]
    if cand.size == 0:
        return None

    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    r = float(fixed_radius_px)

    best_c = None
    best_loss = None
    best_votes = 0


    for cy, cx in cand:
        c = np.array([float(cx), float(cy)], dtype=np.float32)
        d = np.linalg.norm(pts - c[None, :], axis=1)
        res = d - r
        loss = float(np.sum(res * res))
        v = int(vote_map[cy, cx])
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_c = c
            best_votes = v

    if best_c is None:
        return None
    return {"center": best_c, "votes": best_votes}

# 滤波：降采样到目标 fs -> SOS带通 -> 回插值
def resample_to_target(x: np.ndarray, fs: float, fs_target: float) -> Tuple[np.ndarray, float, int, int]:
    """
    返回:x_res, fs_res, up, down
    使得 fs_res = fs * up / down ≈ fs_target
    - fractions.Fraction 的 (numerator, denominator) 构造要求“有理数/整数”；
      因此这里用 ratio = fs_target / fs 的浮点近似再转 Fraction。
    - 若 fs 与 fs_target 近似整数倍关系如 2000 -> 200,优先用整数 down,以保证更稳。
    """
    fs = float(fs)
    fs_target = float(fs_target)
    if fs <= 0 or fs_target <= 0:
        raise ValueError("fs and fs_target must be positive.")

    ratio = fs_target / fs  

    inv = fs / fs_target
    inv_round = int(round(inv))
    if inv_round >= 1 and abs(inv - inv_round) < 1e-3:
        up, down = 1, inv_round
        x_res = resample_poly(x, up, down)
        fs_res = fs * up / down
        return x_res, fs_res, up, down

    frac = Fraction(ratio).limit_denominator(2000)
    up, down = frac.numerator, frac.denominator
    if up <= 0 or down <= 0:
        raise ValueError("Bad resample ratio.")
    x_res = resample_poly(x, up, down)
    fs_res = fs * up / down
    return x_res, fs_res, up, down


def bandpass_sos_zero_phase(x: np.ndarray, fs: float, low: float, high: float, order: int) -> np.ndarray:
    nyq = 0.5 * fs
    sos = butter(order, [low/nyq, high/nyq], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def routeB_filter_xy(x: np.ndarray, y: np.ndarray, fs: float,
                     low: float, high: float,
                     order: int = 4,
                     fs_target: float = 200.0) -> Tuple[np.ndarray, np.ndarray]:

    N = len(x)
    t = np.arange(N, dtype=float) / fs

    x_ds, fs_ds, up, down = resample_to_target(x, fs, fs_target)
    y_ds, fs_ds2, _, _ = resample_to_target(y, fs, fs_target)
    fs_ds = float(fs_ds)
    if abs(fs_ds - fs_ds2) > 1e-6:
        fs_ds = float(fs_ds2)

    # 带通（低采样率域）
    x_bp = bandpass_sos_zero_phase(x_ds, fs_ds, low, high, order)
    y_bp = bandpass_sos_zero_phase(y_ds, fs_ds, low, high, order)

    t_ds = np.arange(len(x_bp), dtype=float) / fs_ds
    x_f = np.interp(t, t_ds, x_bp)
    y_f = np.interp(t, t_ds, y_bp)
    return x_f, y_f


def routeB_filter_1d(sig: np.ndarray, fs: float,
                     low: float, high: float,
                     order: int = 4,
                     fs_target: float = 200.0) -> np.ndarray:

    N = len(sig)
    t = np.arange(N, dtype=float) / fs

    sig_ds, fs_ds, _, _ = resample_to_target(sig, fs, fs_target)
    sig_bp = bandpass_sos_zero_phase(sig_ds, fs_ds, low, high, order)

    t_ds = np.arange(len(sig_bp), dtype=float) / fs_ds
    sig_f = np.interp(t, t_ds, sig_bp)
    return sig_f

# 频谱分析
def frequency_analysis(signal: np.ndarray, fs: float, title: str = "Frequency Spectrum", show: bool = True):
    f, Pxx = periodogram(signal, fs)
    if show:
        plt.figure(figsize=(8, 4))
        plt.semilogy(f, Pxx)
        plt.title(title)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power Spectrum')
        plt.xlim(0, min(100, fs/2))
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    mask = (f >= 1) & (f <= 30)
    if np.any(mask):
        dom_freq = float(f[mask][np.argmax(Pxx[mask])])
        dom_amp = float(np.max(Pxx[mask]))
    else:
        dom_freq = None
        dom_amp = None
    return dom_freq, dom_amp

# max pairwise distance
def max_pairwise_distance_mm(points_px: np.ndarray, scale_mm_per_px: float) -> float:
    pts = np.asarray(points_px, dtype=float)
    n = pts.shape[0]
    if n < 2:
        return 0.0
    if n <= 2000:
        dmax = 0.0
        for i in range(n):
            for j in range(i+1, n):
                d = float(np.linalg.norm(pts[i] - pts[j]))
                if d > dmax:
                    dmax = d
        return dmax * scale_mm_per_px

    try:
        from scipy.spatial import ConvexHull
    except Exception:
        return max_pairwise_distance_mm(pts[:2000], scale_mm_per_px)

    hull = ConvexHull(pts)
    H = pts[hull.vertices]
    m = H.shape[0]
    if m < 2:
        return 0.0

    def area2(a, b, c):
        return abs(np.cross(b-a, c-a))

    j = 1
    dmax = 0.0
    for i in range(m):
        ni = (i + 1) % m
        while area2(H[i], H[ni], H[(j+1) % m]) > area2(H[i], H[ni], H[j]):
            j = (j + 1) % m
        d = float(np.linalg.norm(H[i] - H[j]))
        if d > dmax:
            dmax = d
        d = float(np.linalg.norm(H[ni] - H[j]))
        if d > dmax:
            dmax = d
    return dmax * scale_mm_per_px

# 主流程
def main(show_plots: bool = True):
    plt.rcParams['axes.unicode_minus'] = False

    video_path = pick_video_path()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, first = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Cannot read first frame.")

    # 1) 刻度尺
    scale = get_scale_factor_mm_per_px(first)  # mm/px

    # 2) ROI（初始中心）
    init_center, _ = select_circle_roi(first)
    last_center = np.array([float(init_center[0]), float(init_center[1])], dtype=np.float32)

    # 3) 固定半径(px)
    fixed_radius_px = FIXED_RADIUS_MM / scale
    print(f"[Radius] fixed={FIXED_RADIUS_MM:.3f}mm -> {fixed_radius_px:.3f}px")

    # 4) 屏蔽矩形
    gray0 = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    blur0 = cv2.GaussianBlur(gray0, (9, 9), 2.0)
    edges0 = compute_canny(blur0, CANNY_LOW, CANNY_HIGH)
    ignored_rects = select_ignored_rectangles(first, edges0)
    print(f"[Mask] rectangles={len(ignored_rects)}")

    # 输出路径
    out_path = os.path.join(
        os.path.dirname(video_path),
        os.path.splitext(os.path.basename(video_path))[0] + "_improved_output.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open output writer: {out_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    centers: List[Tuple[float, float]] = []
    votes: List[int] = []

    search_margin_px = 20
    ring_margin_px = 10
    grad_step_limit = 250

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 2.0)
        edges = compute_canny(blur, CANNY_LOW, CANNY_HIGH)
        edges = apply_rect_masks(edges, ignored_rects)

        search_r = int(round(fixed_radius_px + search_margin_px))
        x_min = max(0, int(round(last_center[0] - search_r)))
        y_min = max(0, int(round(last_center[1] - search_r)))
        x_max = min(width, int(round(last_center[0] + search_r)))
        y_max = min(height, int(round(last_center[1] + search_r)))
        roi_box = (x_min, y_min, x_max, y_max)

        mask = np.zeros_like(edges, dtype=np.uint8)
        cv2.circle(mask, (int(round(last_center[0])), int(round(last_center[1]))), search_r, 255, -1)
        edges_roi = cv2.bitwise_and(edges, edges, mask=mask)

        ys, xs = np.where(edges_roi > 0)
        edges_ring = edges_roi
        if xs.size >= 40:
            pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
            d = np.linalg.norm(pts - last_center[None, :], axis=1)
            ring = (d >= (fixed_radius_px - ring_margin_px)) & (d <= (fixed_radius_px + ring_margin_px))
            pts_ring = pts[ring]
            if pts_ring.shape[0] >= 40:
                edges_ring = np.zeros_like(edges_roi)
                edges_ring[pts_ring[:, 1].astype(int), pts_ring[:, 0].astype(int)] = 255

        sol = solve_center_voting_sse_fixed_radius(
            edges_ring, blur,
            fixed_radius_px=fixed_radius_px,
            roi_box=roi_box,
            vote_threshold=VOTE_THRESHOLD,
            step_limit=grad_step_limit
        )

        if sol is not None:
            center = sol["center"]
            last_center = center.copy()
            v = int(sol.get("votes", 0))
        else:
            center = last_center.copy()
            v = 0

        centers.append((float(center[0]), float(center[1])))
        votes.append(v)

        cx, cy = int(round(center[0])), int(round(center[1]))
        cv2.circle(frame, (cx, cy), int(round(fixed_radius_px)), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 2, (0, 0, 255), 3)
        cv2.circle(frame, (int(round(last_center[0])), int(round(last_center[1]))), search_r, (255, 0, 0), 1)
        for (x1, y1, x2, y2) in ignored_rects:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
        out.write(frame)

        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"[Progress] {frame_idx}/{total_frames}")

    cap.release()
    out.release()

    centers_np = np.array(centers, dtype=float)
    stable_center = np.mean(centers_np, axis=0)
    offsets_px = centers_np - stable_center[None, :]

    x_offsets = offsets_px[:, 0] * scale
    y_offsets = offsets_px[:, 1] * scale
    radial_offsets = np.sqrt(x_offsets**2 + y_offsets**2)
    # 滤波
    lowcut, highcut = FILTER_BAND_HZ
    print(f"\n--- Route-B Filter: downsample->{TARGET_FS_HZ:.1f}Hz, bandpass {lowcut}-{highcut}Hz, order={FILTER_ORDER} ---")
    filtered_x, filtered_y = routeB_filter_xy(
        x_offsets, y_offsets,
        fs=fps,
        low=lowcut, high=highcut,
        order=FILTER_ORDER,
        fs_target=TARGET_FS_HZ
    )
    filtered_radial = routeB_filter_1d(
        radial_offsets,
        fs=fps,
        low=lowcut,
        high=highcut,
        order=FILTER_ORDER,
        fs_target=TARGET_FS_HZ
    )
    filtered_radial_mag = np.sqrt(filtered_x**2 + filtered_y**2)

    filtered_centers = np.column_stack([
        stable_center[0] + filtered_x / scale,
        stable_center[1] + filtered_y / scale
    ])

    max_raw_pairwise = max_pairwise_distance_mm(centers_np, scale)
    max_filt_pairwise = max_pairwise_distance_mm(filtered_centers, scale)

    mean_dev = float(np.mean(np.abs(filtered_radial)))
    std_dev = float(np.std(np.abs(filtered_radial)))
    max_dev = float(np.max(np.abs(filtered_radial)))

    dom_freq, dom_amp = frequency_analysis(radial_offsets, fps, "Radial Offset Frequency Spectrum (RAW)", show=show_plots)

    print("\n--- Analysis Results (same metrics/types as original) ---")
    print(f"Video: {video_path}")
    print(f"Output video: {out_path}")
    print(f"FPS: {fps:.3f}, Frames used: {len(centers_np)}")
    print(f"Stable center (px): ({stable_center[0]:.2f}, {stable_center[1]:.2f})")
    print(f"Scale (mm/px): {scale:.6f}")
    print(f"Fixed radius: {FIXED_RADIUS_MM:.3f} mm ({fixed_radius_px:.3f} px)")
    print(f"Max radial deviation (filtered): {max_dev:.4f} mm")
    print(f"Mean radial deviation (filtered): {mean_dev:.4f} mm")
    print(f"Std radial deviation (filtered): {std_dev:.4f} mm")
    print(f"Max pairwise distance (raw): {max_raw_pairwise:.4f} mm")
    print(f"Max pairwise distance (filtered): {max_filt_pairwise:.4f} mm")
    if dom_freq is not None:
        print(f"Dominant freq (RAW radial, 1-30Hz): {dom_freq:.2f} Hz, amp={dom_amp:.3e}")
    else:
        print("Dominant freq: None")

    if show_plots:
        units = "mm"
        time_axis = np.arange(len(radial_offsets)) / fps

        plt.figure(figsize=(15, 12))
        plt.suptitle(f"Offsets: Raw vs Filtered ({units})", fontsize=16)

        # 径向偏移（滤波前后）
        plt.subplot(3, 2, 1)
        plt.plot(time_axis, radial_offsets, label='Radial Raw')
        plt.title('Radial Offset (Raw)')
        plt.xlabel('Time (s)')
        plt.ylabel(f'Radial Offset ({units})')
        plt.grid(True)

        plt.subplot(3, 2, 2)
        plt.plot(time_axis, filtered_radial, label='Radial Filtered', color='orange')
        plt.title(f'Radial Offset (Filtered {lowcut}-{highcut} Hz, BP(r))')
        plt.xlabel('Time (s)')
        plt.ylabel(f'Radial Offset ({units})')
        plt.grid(True)

        # X偏移（滤波前后）
        plt.subplot(3, 2, 3)
        plt.plot(time_axis, x_offsets, label='X Raw')
        plt.title('X Offset (Raw)')
        plt.xlabel('Time (s)')
        plt.ylabel(f'X Offset ({units})')
        plt.grid(True)

        plt.subplot(3, 2, 4)
        plt.plot(time_axis, filtered_x, label='X Filtered', color='orange')
        plt.title(f'X Offset (Filtered {lowcut}-{highcut} Hz)')
        plt.xlabel('Time (s)')
        plt.ylabel(f'X Offset ({units})')
        plt.grid(True)

        # Y偏移（滤波前后）
        plt.subplot(3, 2, 5)
        plt.plot(time_axis, y_offsets, label='Y Raw')
        plt.title('Y Offset (Raw)')
        plt.xlabel('Time (s)')
        plt.ylabel(f'Y Offset ({units})')
        plt.grid(True)

        plt.subplot(3, 2, 6)
        plt.plot(time_axis, filtered_y, label='Y Filtered', color='orange')
        plt.title(f'Y Offset (Filtered {lowcut}-{highcut} Hz)')
        plt.xlabel('Time (s)')
        plt.ylabel(f'Y Offset ({units})')
        plt.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # 频率谱图
        fig, axs = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle(f'Frequency Spectra: Raw vs Filtered ({units})', fontsize=16)

        # 径向
        f_raw, Pxx_raw = periodogram(radial_offsets, fps)
        f_filt, Pxx_filt = periodogram(filtered_radial, fps)
        axs[0,0].semilogy(f_raw, Pxx_raw)
        axs[0,0].set_title("Radial Raw Spectrum")
        axs[0,0].set_xlabel("Frequency (Hz)")
        axs[0,0].set_ylabel("Power")
        axs[0,0].set_xlim(0, min(100, fps/2))
        axs[0,0].grid(True)
        axs[0,1].semilogy(f_filt, Pxx_filt)
        axs[0,1].set_title("Radial Filtered Spectrum")
        axs[0,1].set_xlabel("Frequency (Hz)")
        axs[0,1].set_ylabel("Power")
        axs[0,1].set_xlim(0, min(100, fps/2))
        axs[0,1].grid(True)

        # X方向
        fx_raw, Px_raw = periodogram(x_offsets, fps)
        fx_filt, Px_filt = periodogram(filtered_x, fps)
        axs[1,0].semilogy(fx_raw, Px_raw)
        axs[1,0].set_title("X Raw Spectrum")
        axs[1,0].set_xlabel("Frequency (Hz)")
        axs[1,0].set_ylabel("Power")
        axs[1,0].set_xlim(0, min(100, fps/2))
        axs[1,0].grid(True)
        axs[1,1].semilogy(fx_filt, Px_filt)
        axs[1,1].set_title("X Filtered Spectrum")
        axs[1,1].set_xlabel("Frequency (Hz)")
        axs[1,1].set_ylabel("Power")
        axs[1,1].set_xlim(0, min(100, fps/2))
        axs[1,1].grid(True)

        # Y方向
        fy_raw, Py_raw = periodogram(y_offsets, fps)
        fy_filt, Py_filt = periodogram(filtered_y, fps)
        axs[2,0].semilogy(fy_raw, Py_raw)
        axs[2,0].set_title("Y Raw Spectrum")
        axs[2,0].set_xlabel("Frequency (Hz)")
        axs[2,0].set_ylabel("Power")
        axs[2,0].set_xlim(0, min(100, fps/2))
        axs[2,0].grid(True)
        axs[2,1].semilogy(fy_filt, Py_filt)
        axs[2,1].set_title("Y Filtered Spectrum")
        axs[2,1].set_xlabel("Frequency (Hz)")
        axs[2,1].set_ylabel("Power")
        axs[2,1].set_xlim(0, min(100, fps/2))
        axs[2,1].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    return {
        "video": video_path,
        "output_video": out_path,
        "scale_mm_per_px": scale,
        "fixed_radius_mm": FIXED_RADIUS_MM,
        "stable_center_px": stable_center,
        "centers_px": centers_np,
        "filtered_centers_px": filtered_centers,
        "x_mm": x_offsets,
        "y_mm": y_offsets,
        "r_mm": radial_offsets,
        "x_filt_mm": filtered_x,
        "y_filt_mm": filtered_y,
        "r_filt_mm": filtered_radial,
        "max_raw_pairwise_mm": max_raw_pairwise,
        "max_filt_pairwise_mm": max_filt_pairwise,
        "dom_freq_raw_r": dom_freq,
        "dom_amp_raw_r": dom_amp,
        "votes": np.array(votes, dtype=int),
    }

if __name__ == "__main__":
    main(show_plots=True)