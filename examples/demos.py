import os
import cv2
from src import U2NetSegmenter, AIDetector

import os
import cv2


def main():
    # Initialize the segmentation pipeline.
    segmenter = U2NetSegmenter()  
    # For the AI detection model, let the user (or demo) specify the weights location.
    # Here we assume the weights are stored in a "weights" folder at the repository root.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Adjust the following path as needed; for example, it might be ../weights/...
    weights_path = os.path.join(base_dir, "..", "src", "weights", "AIArtDetection.pth")
    detector = AIDetector(model_path=weights_path)

    # Resolve the demo images folder relative to this file.
    input_folder = os.path.join(base_dir, "demo_images")

    # Process every image in the demo folder.
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        file_path = os.path.join(input_folder, filename)
        image = cv2.imread(file_path)
        if image is None:
            print(f"Skipping {filename}: unable to load the image.")
            continue

        # Run the U²-Net segmentation pipeline.
        processed_image = segmenter.process_image(image)
        if processed_image is None:
            print(f"Segmentation failed for {filename}.")
            continue

        # Run the AI detection on the processed image.
        probability = detector.predict(processed_image)
        if probability is None:
            print(f"AI detection failed for {filename}.")
        else:
            print(f"{filename}: Probability that the image is AI generated: {probability:.4f}")

        # Optionally display the processed image.
        cv2.imshow("Segmented & Processed Image", processed_image)
        print(f"Processed {filename}. Press any key to view the next image...")
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
