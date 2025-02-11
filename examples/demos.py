# demos.py
import os
import cv2
from u2net_segmenter import U2NetSegmenter 

def main():
    # Initialize the segmenter (ensure your weights are in the correct location)
    segmenter = U2NetSegmenter()

    # Define the input folder containing demo images.
    input_folder = "demo_images"

    # Loop over each file in the folder.
    for filename in os.listdir(input_folder):
        # Process only common image file types.
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        file_path = os.path.join(input_folder, filename)
        image = cv2.imread(file_path)

        if image is None:
            print(f"Skipping {filename}: unable to load the image.")
            continue

        # Process the image using our API. The result is the final segmented image.
        result = segmenter.process_image(image)

        # Display the resulting image.
        cv2.imshow("Segmented Image", result)
        print(f"Processed {filename}. Press any key to view the next image...")
        cv2.waitKey(0)  # Wait for a key press before processing the next image

    # Clean up any OpenCV windows.
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
