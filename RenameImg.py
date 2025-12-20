#---------------------------------------------------------
#                                  CONFIG
#---------------------------------------------------------
INPUT_DIR   = "D:/TT Internship/VideoAnnotation/ExtractedFrames/dataset"
OUTPUT_DIR  = "D:/TT Internship/VideoAnnotation/ExtractedFrames/F2694A_Line1_RHCam"
PREFIX      = "Img"
EXTENSION   = ".jpg"


#---------------------------------------------------------
#                                  IMPORTS
#---------------------------------------------------------
import os
import cv2


#---------------------------------------------------------
#                                  FUNCTIONS
#---------------------------------------------------------
def rename_and_save_images(input_dir, output_dir, prefix, ext):
    # Get list of files in input directory
    files = sorted(os.listdir(input_dir))

    count = 1
    for file in files:
        file_path = os.path.join(input_dir, file)

        # Read image
        img = cv2.imread(file_path)
        if img is None:
            continue  # skip non-image files

        # Build new filename
        new_name = f"{prefix}{count}{ext}"
        output_path = os.path.join(output_dir, new_name)

        # Save image
        cv2.imwrite(output_path, img)
        print(f"Saved: {output_path}")

        count += 1


#---------------------------------------------------------
#                                  MAIN
#---------------------------------------------------------
if __name__ == "__main__":
    rename_and_save_images(INPUT_DIR, OUTPUT_DIR, PREFIX, EXTENSION)
