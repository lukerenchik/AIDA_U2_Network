# U²-Net Image Segmentation Package

A Python package for in-memory image segmentation using the U²-Net model. This package processes images entirely in memory—cropping or warping them based on detected contours—so that its output can be fed directly into downstream components (e.g. a CNN for determining if an image is AI‑generated).

## Features

- **In-Memory Processing:** No disk writes; images are processed and returned as NumPy arrays.
- **U²-Net Based:** Uses the U²-Net model to generate saliency masks.
- **Perspective Transformation:** Applies a perspective warp when a quadrilateral is detected.
- **Fallback Cropping:** Falls back to bounding box cropping if a quadrilateral is not detected.
- **Cloud-Ready API:** Designed to be integrated in cloud services where images are processed and passed directly to subsequent stages.

## Package Structure


## Getting Started

AIDA_U2/ 
├── init.py # Exposes the main API class. 
├── config.py # Configuration constants (e.g. model weights path, image size). 
├── segmenter.py # Contains the U2NetSegmenter class with the segmentation API. 
├── u2net.py # Contains the U²-Net network definition. 
├── utils.py # Helper functions for image preprocessing and transforms. 
├── weights/  
    └── u2net.pth # Pre-trained U²-Net model weights. 
└── examples/
    └── demos.py # Demo script to process all images in the demo_images folder.



### Prerequisites

- **Python 3.7+**
- **PyTorch** and **Torchvision**
- **OpenCV** (`opencv-python`)
- **NumPy**
- **Matplotlib** (used in demos/visualizations)

### Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd u2net_segmenter

2. **Install the Required Packages:**
    ```bash
    pip install -r requirements.txt

3. **Download the U^2 Net Weights:**

    Place your pre-trained U²-Net weights (e.g. u2net.pth) inside the weights/ folder. Ensure the path in config.py matches the location of your weights file

### Usage

import cv2
from u2net_segmenter import U2NetSegmenter

# Initialize the segmenter.
segmenter = U2NetSegmenter()

# Load an image (ensure the image is in BGR format as used by OpenCV).
image = cv2.imread("path/to/your/image.jpg")
if image is None:
    raise ValueError("Unable to load image.")

# Process the image.
result = segmenter.process_image(image)

# 'result' is a NumPy array containing the segmented (warped or cropped) image.
# It can now be passed directly into your downstream CNN.
cv2.imshow("Segmented Image", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
