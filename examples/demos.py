import os
import cv2
from src.AIDA_Network import U2NetSegmenter, AIDetector

def main():
    # Define weights paths
    base_dir = os.path.dirname(os.path.abspath(__file__))  # examples/
    u2net_model_path = os.path.join(base_dir, "..", "src", "weights", "u2net.pth")
    ai_model_path = os.path.join(base_dir, "..", "src", "weights", "AIArtDetection.pth")

    # Initialize with explicit model paths
    segmenter = U2NetSegmenter(model_path=u2net_model_path)
    detector = AIDetector(model_path=ai_model_path)

    input_folder = os.path.join(base_dir, "demo_images")
    output_folder = os.path.join(base_dir, "demo_images_heatmaps")
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        file_path = os.path.join(input_folder, filename)
        image = cv2.imread(file_path)
        if image is None:
            print(f"Skipping {filename}: unable to load the image.")
            continue

        processed_image = segmenter.process_image(image)
        if processed_image is None:
            print(f"Segmentation failed for {filename}.")
            continue

        probability, heatmap = detector.predict(processed_image, return_heatmap=True)
        if probability is None:
            print(f"AI detection failed for {filename}.")
            continue
        
        print(f"{filename}: Probability that the image is AI generated: {probability:.4f}")

        if heatmap is not None:
            overlay_image = detector.overlay_heatmap(processed_image, heatmap, alpha=0.5)
            base_name, ext = os.path.splitext(filename)
            output_filename = f"{base_name}_heatmap{ext}"
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))
            print(f"Saved heatmap overlay for {filename} as {output_filename}")

        cv2.imshow("Segmented & Processed Image", processed_image)
        if heatmap is not None:
            cv2.imshow("Heatmap Overlay", overlay_image)
        print(f"Processed {filename}. Press any key to view the next image...")
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()