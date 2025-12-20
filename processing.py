import cv2
from ultralytics import YOLO

def process_video(model_path, video_path, output_path, imgsz=640, conf=0.5, device="cuda"):

    model = YOLO(model_path)


    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return


    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, FPS, (W, H))

    print("Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break


        results = model.predict(frame, imgsz=imgsz, conf=conf, device=device, verbose=False)

        annotated_frame = results[0].plot()

        writer.write(annotated_frame)

    cap.release()
    writer.release()
    print(f"Done! Processed video saved at: {output_path}")


if __name__ == "__main__":
    model_path = "/home/traxian/Ahmed_intern/samworth/models/segmentation_exp/weights/best.pt"
    video_path = "/home/traxian/Ahmed_intern/samworth/data/raw_data/videos/video2.avi"
    output_path = "/home/traxian/Ahmed_intern/samworth/processed_video/processed_video2.mp4"

    process_video(model_path, video_path, output_path, device="cuda")