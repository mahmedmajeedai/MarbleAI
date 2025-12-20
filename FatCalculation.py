import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
 
# =========================
# Helpers
# =========================
 
def _white_mask_from_segment_bgr(
    seg_bgr,
    lower_white=np.array([105, 105, 140], dtype=np.uint8),
    upper_white=np.array([255, 255, 255], dtype=np.uint8),
    do_morph=True,
):
    """Binary mask (uint8 0/255) of 'white/fat' pixels inside seg_bgr using BGR thresholds."""
    mask = cv2.inRange(seg_bgr, lower_white, upper_white)
    if do_morph:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask
 
 
def _put_boxed_text(
    img,
    center_xy,
    text,
    font_scale=0.9,      # medium-ish
    thickness=2,
    pad=6,
    text_color=(255, 0, 0),
    box_color=(255, 255, 255),
):
    """Draw a white rectangle behind one line of text at a given center."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    w = tw + 2 * pad
    h = th + 2 * pad
    cX, cY = center_xy
    x1, y1 = cX - w // 2, cY - h // 2
    x2, y2 = x1 + w, y1 + h
    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, -1)
    cv2.putText(
        img, text, (x1 + pad, y1 + pad + th),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA
    )
 
# =========================
# Main
# =========================
 
def process_video(
    model_path,
    video_path,
    output_path,
    imgsz=640,
    conf=0.5,
    device="cuda",
    # ---- BGR white thresholds ----
    lower_white=np.array([105, 105, 140], dtype=np.uint8),
    upper_white=np.array([255, 255, 255], dtype=np.uint8),
    meat_class_ids=None,
    line_x_ratio=0.50,
    line_color=(255, 255, 0),
    line_thickness=2,
):
    """
    YOLO-seg + BGR fat %.
    Draw a vertical line at W*line_x_ratio. For any instance whose centroid cX >= line_x,
    we SKIP segmentation visualization and fat calculation (to avoid dark region on the right).
    """
 
    model = YOLO(model_path)
 
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return
 
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0
 
    line_x = int(W * float(line_x_ratio))  # x-position of the cutoff line
 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(output_path, fourcc, FPS, (W, H))
 
    print("Processing video...")
 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
 
        # draw the vertical cutoff line first (for visibility on every frame)
        cv2.line(frame, (line_x, 0), (line_x, H), line_color, line_thickness)
 
        results = model.predict(frame, imgsz=imgsz, conf=conf, device=device, verbose=False)
        r = results[0]
        masks = r.masks
        boxes = r.boxes
 
        if masks is not None and boxes is not None and len(masks) == len(boxes):
            bin_masks = masks.data.cpu().numpy().astype(np.uint8)  # [N,Hmask,Wmask] in {0,1}
            polys = masks.xy
            cls = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros((len(boxes),), dtype=int)
 
            for i in range(len(bin_masks)):
                if meat_class_ids is not None and cls[i] not in meat_class_ids:
                    continue
 
                # Build seg mask and ensure same size as frame
                seg_mask = (bin_masks[i] * 255).astype(np.uint8)
                if seg_mask.shape[:2] != frame.shape[:2]:
                    seg_mask = cv2.resize(seg_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                if cv2.countNonZero(seg_mask) == 0:
                    continue
 
                # Contour for centroid + drawing (skip safely if no polygon)
                cX = cY = None
                if len(polys) > i and polys[i] is not None and len(polys[i]) >= 3:
                    contour = np.array(polys[i], dtype=np.int32).reshape(-1, 1, 2)
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                    else:
                        cX, cY = int(contour[0, 0, 0]), int(contour[0, 0, 1])
 
                # If we don't have a centroid from polygon (rare), estimate from mask moments
                if cX is None:
                    M = cv2.moments(seg_mask, binaryImage=True)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                    else:
                        # fallback: skip
                        continue
 
                # ----- SKIP logic: once the item crosses the line (centroid on/after it), ignore it -----
                if cX >= line_x:
                    # Do NOT draw its contour or compute fat. Just ignore this instance.
                    continue
 
                # Extract pixels inside the segment (left side only, before crossing)
                seg_bgr = cv2.bitwise_and(frame, frame, mask=seg_mask)
 
                # Fat (white) mask inside the segment - BGR thresholds (like senior code)
                fat_mask = _white_mask_from_segment_bgr(seg_bgr, lower_white=lower_white, upper_white=upper_white, do_morph=True)
 
                # Areas (in pixels)
                total_area = int(cv2.countNonZero(seg_mask))
                fat_area   = int(cv2.countNonZero(fat_mask))
                fat_pct    = (fat_area / total_area * 100.0) if total_area > 0 else 0.0
 
                # Visualize: paint fat pixels pure white; blend back only where segment exists
                fat_pixels = fat_mask > 0
                seg_bgr[fat_pixels] = (255, 255, 255)
                frame[seg_mask > 0] = seg_bgr[seg_mask > 0]
 
                # Draw contour (green) and the text (only Fat %)
                if len(polys) > i and polys[i] is not None and len(polys[i]) >= 3:
                    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
                _put_boxed_text(frame, (cX, cY), f"Fat: {fat_pct:.2f}%")
 
        writer.write(frame)
 
    cap.release()
    writer.release()
    print(f"Done! Processed video saved at: {output_path}")
 
 
# =========================
# Run directly
# =========================
if __name__ == "__main__":
    model_path = "/home/traxian/Ahmed_intern/samworth/models/segmentation_exp/weights/best.pt"
    video_path = "/home/traxian/Ahmed_intern/samworth/data/raw_data/videos/video2.avi"
    output_path = "/home/traxian/Ahmed_intern/samworth/processed_video/samworth_017_again.mp4"
 
    process_video(
        model_path=model_path,
        video_path=video_path,
        output_path=output_path,
        imgsz=640,
        conf=0.5,
        device="cuda",
        lower_white=np.array([74, 120, 212], dtype=np.uint8),
        upper_white=np.array([255, 255, 255], dtype=np.uint8),
        meat_class_ids=None,
        line_x_ratio=0.35, 
        line_color=(255, 255, 0),
        line_thickness=2,
    )