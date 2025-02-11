import os
import cv2
from src import U2NetSegmenter

def main():
    # Initialize the segmenter
    segmenter = U2NetSegmenter()

    # Resolve the input folder relative to this file's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(base_dir, "demo_images")

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        file_path = os.path.join(input_folder, filename)
        image = cv2.imread(file_path)

        if image is None:
            print(f"Skipping {filename}: unable to load the image.")
            continue

        result = segmenter.process_image(image)
        cv2.imshow("Segmented Image", result)
        print(f"Processed {filename}. Press any key to view the next image...")
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
