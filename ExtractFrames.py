#------------------------------------------------------------------------------------------------
#                                  IMPORTS
#------------------------------------------------------------------------------------------------
import cv2
import os

#------------------------------------------------------------------------------------------------
#                                  CONFIG
#------------------------------------------------------------------------------------------------
VIDEO_PATH   = "/home/muhammad-ahmed/Documents/ThingTrax/samworth/data/raw_data/videos/video1.avi"
OUTPUT_DIR   = "/home/muhammad-ahmed/Documents/ThingTrax/samworth/data/raw_data/frames/video1"
FRAMES       = 5

#------------------------------------------------------------------------------------------------
#                                  FUNCTIONS
#------------------------------------------------------------------------------------------------
def extract_frames(video_path, output_dir, fps):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    # Get video FPS
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps // fps) if video_fps > fps else 1

    frame_count = 25
    saved_count = 24

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save every 'frame_interval'-th frame
        if frame_count % frame_interval == 0:
            frame_name = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_name, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Done. Extracted {saved_count} frames to '{output_dir}'.")


#------------------------------------------------------------------------------------------------
#                                  MAIN
#------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    extract_frames(VIDEO_PATH, OUTPUT_DIR, FRAMES)
