import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, periodogram

def select_circle_roi(frame):
    display_frame = frame.copy()
    center, radius = None, 0
    drawing = False
    dragging_center = False
    adjusting_radius = False
    initial_click_pos = None
    initial_center = None
    initial_radius = 0

    def draw_center_marker(img, center_point):
        """标记圆心位置：红点 + 黄色十字线，仅为视觉辅助"""
        cross_len = 12
      
        cv2.circle(img, center_point, 3, (0, 0, 255), -1)
      
        cv2.line(img,
                 (center_point[0] - cross_len, center_point[1]),
                 (center_point[0] + cross_len, center_point[1]),
                 (0, 255, 255), 1)
        cv2.line(img,
                 (center_point[0], center_point[1] - cross_len),
                 (center_point[0], center_point[1] + cross_len),
                 (0, 255, 255), 1)

    def is_near_center(x, y, center, threshold=15):
        """检查鼠标是否在圆心附近"""
        if center is None:
            return False
        dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        return dist < threshold

    def is_near_edge(x, y, center, radius, threshold=10):
        """检查鼠标是否在圆边缘附近"""
        if center is None or radius <= 0:
            return False
        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        return abs(dist_from_center - radius) < threshold

    def mouse_callback(event, x, y, flags, param):
        nonlocal center, radius, drawing, dragging_center, adjusting_radius
        nonlocal initial_click_pos, initial_center, initial_radius
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if center is None:
                
                drawing = True
                center = (x, y)
                radius = 0
            elif is_near_center(x, y, center):
               
                dragging_center = True
                initial_click_pos = (x, y)
                initial_center = center
            elif radius > 0 and is_near_edge(x, y, center, radius):
              
                adjusting_radius = True
                initial_click_pos = (x, y)
                initial_radius = radius
            else:
              
                center = (x, y)
                radius = 0
                drawing = True
                
        elif event == cv2.EVENT_MOUSEMOVE:
            temp_frame = frame.copy()
            
            if drawing:
                # 正在绘制新圆
                radius = int(np.sqrt((x - center[0])**2 + (y - center[1])**2))
                cv2.circle(temp_frame, center, radius, (0, 255, 0), 2)
                draw_center_marker(temp_frame, center)
            elif dragging_center and initial_center is not None:
                # 正在拖动圆心
                dx = x - initial_click_pos[0]
                dy = y - initial_click_pos[1]
                center = (initial_center[0] + dx, initial_center[1] + dy)
                cv2.circle(temp_frame, center, radius, (0, 255, 0), 2)
                draw_center_marker(temp_frame, center)
            elif adjusting_radius and initial_radius > 0:
                # 正在调整半径
                radius = int(np.sqrt((x - center[0])**2 + (y - center[1])**2))
                if radius < 5:  # 最小半径限制
                    radius = 5
                cv2.circle(temp_frame, center, radius, (0, 255, 0), 2)
                draw_center_marker(temp_frame, center)
            elif center is not None and radius > 0:
               
                cv2.circle(temp_frame, center, radius, (0, 255, 0), 2)
                draw_center_marker(temp_frame, cent)
                if is_near_center(x, y, center):
                    cv2.circle(temp_frame, center, 15, (255, 255, 0), 1)  
                elif is_near_edge(x, y, center, radius):
                   
                    edge_angle = np.arctan2(y - center[1], x - center[0])
                    edge_x = int(center[0] + radius * np.cos(edge_angle))
                    edge_y = int(center[1] + radius * np.sin(edge_angle))
                    cv2.circle(temp_frame, (edge_x, edge_y), 8, (255, 255, 0), 1)
            
            cv2.imshow("Select Circular ROI", temp_frame)
            
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            dragging_center = False
            adjusting_radius = False
            initial_click_pos = None
            initial_center = None
            initial_radius = 0
            
            # 更新显示
            if center is not None and radius > 0:
                temp_frame = frame.copy()
                cv2.circle(temp_frame, center, radius, (0, 255, 0), 2)
                draw_center_marker(temp_frame, center)
                cv2.imshow("Select Circular ROI", temp_frame)

    cv2.namedWindow("Select Circular ROI")
    cv2.setMouseCallback("Select Circular ROI", mouse_callback)
    instruction = "Drag to create circle | Drag center to move | Drag edge to resize | Enter to confirm, ESC to cancel"
    cv2.putText(display_frame, instruction, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow("Select Circular ROI", display_frame)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter
            if center is not None and radius > 0:
                break
            else:
                print("Please select a circle first!")
        elif key == 27:  # ESC
            center, radius = None, None
            break
    cv2.destroyAllWindows()
    return center, radius

def get_scale_reference(frame):
    """Allow user to select two points and input real distance to establish scale"""
    points = []
    display_frame = frame.copy()
    h, w = frame.shape[:2]
    dragging_point_idx = None  
    dragging_midpoint = False  
    initial_click_pos = None
    initial_points = None
    
    def is_near_point(x, y, point, threshold=10):
        """检查鼠标是否在点附近"""
        dist = np.sqrt((x - point[0])**2 + (y - point[1])**2)
        return dist < threshold
    
    def is_near_midpoint(x, y, mid_point, threshold=10):
        """检查鼠标是否在中点附近"""
        dist = np.sqrt((x - mid_point[0])**2 + (y - mid_point[1])**2)
        return dist < threshold
    
    def draw_line_and_info(img, pt1, pt2):
        """绘制直线、点、中点和距离信息"""
        temp_img = img.copy()
        
     
        cv2.circle(temp_img, pt1, 5, (0, 0, 255), -1)
        cv2.circle(temp_img, pt2, 5, (0, 0, 255), -1)
        
      
        cv2.line(temp_img, pt1, pt2, (0, 255, 0), 2)
        
       
        mid_point = ((pt1[0] + pt2[0])//2, (pt1[1] + pt2[1])//2)
        
        cv2.circle(temp_img, mid_point, 6, (0, 0, 255), -1)
        
    
        pixel_dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
        

        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        line_length = np.sqrt(dx*dx + dy*dy)
        if line_length > 0:
          
            perp_x = -dy / line_length
            perp_y = dx / line_length
        else:
            perp_x, perp_y = 0, -1
        
     
        offset_distance = 50
        text_pos = (int(mid_point[0] + perp_x * offset_distance - 80),
                   int(mid_point[1] + perp_y * offset_distance))
        
     
        text_pos = (max(10, min(text_pos[0], w - 200)), 
                   max(30, min(text_pos[1], h - 20)))
        
    
        cv2.putText(temp_img, f"Distance: {pixel_dist:.2f} pixels", 
                   text_pos, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return temp_img, mid_point
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal points, display_frame, dragging_point_idx, dragging_midpoint
        nonlocal initial_click_pos, initial_points
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 2:
               
                points.append((x, y))
                if len(points) == 2:
                    temp_frame, _ = draw_line_and_info(frame, points[0], points[1])
                    display_frame = temp_frame
            elif len(points) == 2:
                
                mid_point = ((points[0][0] + points[1][0])//2, 
                            (points[0][1] + points[1][1])//2)
                
                if is_near_point(x, y, points[0]):
                   
                    dragging_point_idx = 0
                    initial_click_pos = (x, y)
                    initial_points = [points[0][:], points[1][:]]
                elif is_near_point(x, y, points[1]):
                 
                    dragging_point_idx = 1
                    initial_click_pos = (x, y)
                    initial_points = [points[0][:], points[1][:]]
                elif is_near_midpoint(x, y, mid_point):
                 
                    dragging_midpoint = True
                    initial_click_pos = (x, y)
                    initial_points = [points[0][:], points[1][:]]
                else:
                   
                    points = [(x, y)]
                    
        elif event == cv2.EVENT_MOUSEMOVE:
            temp_frame = frame.copy()
            
            if len(points) == 2:
                if dragging_point_idx is not None and initial_points is not None:
                 
                    dx = x - initial_click_pos[0]
                    dy = y - initial_click_pos[1]
                    points[dragging_point_idx] = (initial_points[dragging_point_idx][0] + dx,
                                                  initial_points[dragging_point_idx][1] + dy)
                    temp_frame, _ = draw_line_and_info(temp_frame, points[0], points[1])
                elif dragging_midpoint and initial_points is not None:
                
                    dx = x - initial_click_pos[0]
                    dy = y - initial_click_pos[1]
                    points[0] = (initial_points[0][0] + dx, initial_points[0][1] + dy)
                    points[1] = (initial_points[1][0] + dx, initial_points[1][1] + dy)
                    temp_frame, _ = draw_line_and_info(temp_frame, points[0], points[1])
                else:
                    temp_frame, mid_point = draw_line_and_info(temp_frame, points[0], points[1])
                    
                    if is_near_point(x, y, points[0]) or is_near_point(x, y, points[1]):
                
                        for pt in points:
                            if is_near_point(x, y, pt):
                                cv2.circle(temp_frame, pt, 10, (255, 255, 0), 1)
                    elif is_near_midpoint(x, y, mid_point):
                  
                        cv2.circle(temp_frame, mid_point, 10, (255, 255, 0), 1)
            elif len(points) == 1:
              
                cv2.circle(temp_frame, points[0], 5, (0, 0, 255), -1)
            
            cv2.imshow("Set Scale Reference", temp_frame)
            
        elif event == cv2.EVENT_LBUTTONUP:
            dragging_point_idx = None
            dragging_midpoint = False
            initial_click_pos = None
            initial_points = None

            if len(points) == 2:
                temp_frame, _ = draw_line_and_info(frame, points[0], points[1])
                display_frame = temp_frame
                cv2.imshow("Set Scale Reference", display_frame)
    
    cv2.namedWindow("Set Scale Reference")
    cv2.setMouseCallback("Set Scale Reference", mouse_callback)
    
    instruction = "Click two points | Drag endpoints to resize | Drag midpoint to move | Enter to confirm, ESC to cancel"
    cv2.putText(display_frame, instruction, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    cv2.imshow("Set Scale Reference", display_frame)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in (13, 10, ord('\r'), ord('\n'), ord('q')):  # Enter/Return/q
            if len(points) == 2:
                break
            else:
                print("Please select two points first!")
        elif key == 27:  
            points = None
            break
    
    cv2.destroyAllWindows()
    if points and len(points) == 2:
        pixel_distance = np.linalg.norm(np.array(points[0]) - np.array(points[1]))
        while True:
            try:
                real_distance = float(input("Enter the real distance between the two points (in mm, can be decimal): "))
                if real_distance <= 0:
                    print("Distance must be positive. Please try again.")
                    continue
                break
            except ValueError:
                print("Invalid input. Please enter a number (can be decimal).")
        scale_factor = real_distance / pixel_distance
        print(f"Scale factor: {scale_factor:.6f} mm/pixel")
        return scale_factor
    else:
        return None

def playback_tracked_video(video_path, fps, preview_speed=1.0, window_name="Tracking Preview"):
    """Smooth sequential playback of the tracked video with frame counter.
    
    Args:
        video_path: Path to the video file
        fps: Original frames per second of the video
        preview_speed: Playback speed multiplier (1.0 = normal, 0.5 = half speed, 2.0 = double speed)
        window_name: Name of the preview window
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open tracked video for preview: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print("Tracked video has no frames, skipping preview.")
        cap.release()
        return

    fps = fps if fps and fps > 0 else 30
    # 根据预览速度调整延迟时间
    delay = max(int(1000 / (fps * preview_speed)), 1)

    cv2.namedWindow(window_name)

    frame_idx = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        # Frame counter text
        text = f"Frame {frame_idx}/{total_frames}"
        text_size, _ = cv2.getTextSize(text, font, 0.6, 2)
        text_x = max(frame.shape[1] - text_size[0] - 10, 10)
        text_y = max(text_size[1] + 10, 20)
        cv2.putText(frame, text, (text_x, text_y), font, 0.6, (0, 255, 255), 2)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(delay) & 0xFF
        if key in (ord('q'), 27):  # q or ESC
            break

    cap.release()
    cv2.destroyWindow(window_name)

def save_slow_motion_video(input_path, output_path, original_fps, slow_factor):
    """Save a slower version of the tracked video by lowering the fps metadata."""
    if slow_factor <= 1.0:
        return  # no need to create slow version

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Cannot open tracked video for slow-motion export: {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    new_fps = max(original_fps / slow_factor, 1)

    slow_writer = cv2.VideoWriter(output_path, fourcc, new_fps, (width, height))
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        slow_writer.write(frame)
        frame_count += 1

    cap.release()
    slow_writer.release()
    print(f"Saved slow-motion video ({slow_factor}x slower, {new_fps:.2f} fps) to: {output_path}")

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def frequency_analysis(signal, fs, plot_title="Frequency Spectrum", show_plot=True):
    # Compute and plot frequency spectrum
    f, Pxx = periodogram(signal, fs)
    if show_plot:
        plt.figure(figsize=(8, 4))
        plt.semilogy(f, Pxx)
        plt.title(plot_title)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power Spectrum')
        plt.xlim(0, min(100, fs/2))
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    # Find dominant frequency and amplitude in 1-30 Hz
    mask = (f >= 1) & (f <= 30)
    if np.any(mask):
        dom_freq = f[mask][np.argmax(Pxx[mask])]
        dom_amp = np.max(Pxx[mask])
    else:
        dom_freq = dom_amp = None
    return dom_freq, dom_amp

def detect_shaft_wobble(
    video_path, 
    output_path=None, 
    show_plots=True, 
    freq_interval=(1, 30),   
    filter_order=4,
    show_tracking_preview=False,
    slow_motion_factor=5.0,
    preview_speed=0.2
):
    
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video file: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if output_path is None:
        output_path = video_path.rsplit('.', 1)[0] + '_output.mp4'
        print(f"视频将保存到: {output_path}")

    if slow_motion_factor > 1.0:
        output_fps = max(fps / slow_motion_factor, 1)  
        print(f"保存为慢速视频: {slow_motion_factor}x 慢速 (FPS: {output_fps:.2f})")
    else:
        output_fps = fps  


    fourcc = cv2.VideoWriter_fourcc(*'avc1')  
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
    
    if not out.isOpened():
        print(f"警告: 无法使用H.264编码,尝试使用MPEG-4编码...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 备用MPEG-4编码
        out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
        
        if not out.isOpened():
            print(f"警告: 无法使用MPEG-4编码，使用XVID编码（AVI格式）...")
            # 如果都失败，回退到XVID
            output_path = output_path.rsplit('.', 1)[0] + '.avi'
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
            print(f"视频将保存为AVI格式: {output_path}")

    ret, first_frame = cap.read()
    if not ret:
        print(f"Cannot read video frame: {video_path}")
        cap.release()
        return None

    # Get scale reference first
    scale_factor = get_scale_reference(first_frame)
    if scale_factor is None:
        print("No scale reference provided, using pixel units")
        scale_factor = 1.0
        units = "pixels"
    else:
        units = "mm"

    initial_center, initial_radius = select_circle_roi(first_frame)
    if initial_center is None:
        print("No circular ROI selected, skipping this video.")
        cap.release()
        out.release()
        return None

    print(f"Selected initial circular ROI: center={initial_center}, radius={initial_radius} pixels")
    print(f"Scale factor: {scale_factor:.6f} {units}/pixel")

    detected_centers = []
    frame_count = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    last_known_center = initial_center

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        search_radius = initial_radius + 20
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, last_known_center, search_radius, 255, -1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        roi_gray = cv2.bitwise_and(blurred, blurred, mask=mask)

        circles = cv2.HoughCircles(roi_gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                 param1=100, param2=30, 
                                 minRadius=initial_radius-15, 
                                 maxRadius=initial_radius+15)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            if len(circles[0, :]) > 1:
                distances = np.linalg.norm(circles[0, :, :2] - np.array(last_known_center), axis=1)
                best_circle = circles[0, np.argmin(distances)]
            else:
                best_circle = circles[0, 0]
            cx, cy, r = best_circle
            center_detected = (cx, cy)
            detected_centers.append(center_detected)
            last_known_center = center_detected
            cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 2, (0, 0, 255), 3)
        else:
            detected_centers.append(last_known_center)

        # 绘制搜索区域（蓝色圆圈）
        cv2.circle(frame, last_known_center, search_radius, (255, 0, 0), 1)
        
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    if show_tracking_preview:
        playback_tracked_video(output_path, output_fps, preview_speed)

    if not detected_centers:
        print("No circular cross-section detected.")
        return None

    detected_centers = np.array(detected_centers)
    stable_center = np.mean(detected_centers, axis=0)
    offsets = detected_centers - stable_center
    radial_offsets = np.linalg.norm(offsets, axis=1) * scale_factor

    # 滤波处理
    lowcut, highcut = freq_interval
    print(f"\n--- Frequency filter: {lowcut} Hz to {highcut} Hz ---")
    
    # 对X和Y偏移分别进行滤波
    x_offsets = offsets[:, 0] * scale_factor
    y_offsets = offsets[:, 1] * scale_factor
    
    if len(x_offsets) > filter_order * 3:
        try:
            filtered_x = butter_bandpass_filter(x_offsets, lowcut, highcut, fps, order=filter_order)
            filtered_y = butter_bandpass_filter(y_offsets, lowcut, highcut, fps, order=filter_order)
            # 创建滤波后的圆心位置数组（像素单位）
            filtered_centers = np.column_stack([
                stable_center[0] + filtered_x / scale_factor,
                stable_center[1] + filtered_y / scale_factor
            ])
        except Exception as e:
            print(f"Bandpass filter failed for x/y: {e}")
            filtered_x = x_offsets
            filtered_y = y_offsets
            filtered_centers = detected_centers
    else:
        print("Data too short for filtering x/y.")
        filtered_x = x_offsets
        filtered_y = y_offsets
        filtered_centers = detected_centers

    # 计算滤波后的最大两两距离
    if len(filtered_centers) >= 2:
        pairwise_distances = []
        for i in range(len(filtered_centers)):
            for j in range(i+1, len(filtered_centers)):
                dist = np.linalg.norm(filtered_centers[i] - filtered_centers[j]) * scale_factor
                pairwise_distances.append(dist)
        max_pairwise_distance = max(pairwise_distances) if pairwise_distances else 0
    else:
        max_pairwise_distance = 0

    # 计算原始数据的最大两两距离（用于比较）
    if len(detected_centers) >= 2:
        raw_pairwise_distances = []
        for i in range(len(detected_centers)):
            for j in range(i+1, len(detected_centers)):
                dist = np.linalg.norm(detected_centers[i] - detected_centers[j]) * scale_factor
                raw_pairwise_distances.append(dist)
        max_raw_pairwise_distance = max(raw_pairwise_distances) if raw_pairwise_distances else 0
    else:
        max_raw_pairwise_distance = 0

    # 滤波径向偏移
    if len(radial_offsets) > filter_order * 3:
        try:
            filtered_radial = butter_bandpass_filter(radial_offsets, lowcut, highcut, fps, order=filter_order)
        except Exception as e:
            print(f"Bandpass filter failed: {e}")
            filtered_radial = radial_offsets
    else:
        print("Data too short for filtering.")
        filtered_radial = radial_offsets

    # FFT分析
    dom_freq, dom_amp = frequency_analysis(radial_offsets, fps, plot_title="Radial Offset Frequency Spectrum", show_plot=show_plots)
    if dom_freq is not None:
        print(f"Dominant frequency: {dom_freq:.2f} Hz, amplitude: {dom_amp:.3e}")
    else:
        print("No dominant frequency detected.")

    # 统计滤波后的信号
    mean_deviation = np.mean(filtered_radial)
    std_deviation = np.std(filtered_radial)
    max_deviation = np.max(filtered_radial)

    print("\n--- Analysis Results ---")
    print(f"Video file: {video_path}")
    print(f"Stable reference center: ({stable_center[0]:.2f}, {stable_center[1]:.2f}) pixels")
    print(f"Max radial deviation: {max_deviation:.4f} {units} (filtered)")
    print(f"Mean radial deviation: {mean_deviation:.4f} {units} (filtered)")
    print(f"Std deviation: {std_deviation:.4f} {units} (filtered)")
    print(f"Maximum distance between any two center points (raw): {max_raw_pairwise_distance:.4f} {units}")
    print(f"Maximum distance between any two center points (filtered): {max_pairwise_distance:.4f} {units}")

    if show_plots:
        time_axis = np.arange(len(radial_offsets)) / fps

        # 绘图：径向、X、Y偏移（滤波前后）
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
        plt.title(f'Radial Offset (Filtered {lowcut}-{highcut} Hz)')
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

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 为标题预留空间
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
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  #
        plt.show()

        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        fig.suptitle(f'Center Trajectory (X-Y Coordinates, {units})', fontsize=16)
        
     
        filt_x = filtered_centers[:, 0] * scale_factor
        filt_y = filtered_centers[:, 1] * scale_factor
        
        
        ax.plot(filt_x, filt_y, 'r-', linewidth=2, label='Filtered Trajectory')
        
        # 标记起点和终点（使用过滤后的坐标）
        ax.plot(filt_x[0], filt_y[0], 'go', markersize=10, label='Start')
        ax.plot(filt_x[-1], filt_y[-1], 'bo', markersize=10, label='End')
        
        ax.set_xlabel(f'X Position ({units})', fontsize=14)
        ax.set_ylabel(f'Y Position ({units})', fontsize=14)
        ax.set_title('Center Movement Trajectory', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        ax.set_aspect('equal', adjustable='box')
        
        
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=10)
        
        plt.tight_layout()
        

        trajectory_path = video_path.rsplit('.', 1)[0] + '_trajectory.png'
        plt.savefig(trajectory_path, dpi=300, bbox_inches='tight')
        print(f"轨迹图已保存到: {trajectory_path}")
        
        plt.show()

    return {
        "video": video_path,
        "stable_center": stable_center,
        "mean_dev": mean_deviation,
        "std_dev": std_deviation,
        "max_dev": max_deviation,
        "max_raw_pairwise_distance": max_raw_pairwise_distance,
        "max_pairwise_distance": max_pairwise_distance,
        "dom_freq": dom_freq,
        "dom_amp": dom_amp,
        "units": units,
        "scale_factor": scale_factor
    }

if __name__ == "__main__":
    video_files = ["/Volumes/Extreme SSD/20251109/20251109/C01_09_C001H001S0001/C01_09_C001H001S0001.avi",]

    print("Batch analysis of arrow shaft videos started...")
    print("=========================================\n")
    freq_interval = (4, 6)  

    for video_path in video_files:
        print(f"--- Analyzing: {video_path} ---")
        try:
            detect_shaft_wobble(
                video_path, 
                show_plots=True, 
                freq_interval=freq_interval, 
                filter_order=4
            )
            print(f"--- Analysis finished: {video_path} ---\n")
        except Exception as e:
            print(f"!!!!!! Error when processing {video_path}: {e} !!!!!!")
            print("!!!!!! Skipping to next video. !!!!!!\n")

    print("=========================================")
    print("All video analysis completed.")
