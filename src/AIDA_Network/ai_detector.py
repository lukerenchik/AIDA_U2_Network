# Assuming this is in src/AIDA_Network/AIDA_Network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models import resnet50, ResNet50_Weights
import cv2
from PIL import Image
import os
import numpy as np

class CustomResNet50(nn.Module):
    def __init__(self, extra_conv_layers=5):
        super(CustomResNet50, self).__init__()
        base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.base = nn.Sequential(*list(base_model.children())[:-2])
        conv_layers = []
        for i in range(extra_conv_layers):
            in_channels = 2048 if i == 0 else 128
            conv_layers.append(nn.Conv2d(in_channels, 128, kernel_size=3, padding=1))
            conv_layers.append(nn.ReLU(inplace=True))
            conv_layers.append(nn.BatchNorm2d(128))
        self.extra_convs = nn.Sequential(*conv_layers)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.base(x)
        x = self.extra_convs(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class AIDetector:
    def __init__(self, model_path=None, device=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        if model_path is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            model_path = os.path.join(project_root, "weights", "AIArtDetection.pth")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Error: Model weights not found at {model_path}")

        self.model = CustomResNet50(extra_conv_layers=5).to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print("Error loading model:", e)
            self.model = None

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Heatmap setup
        self.feature_maps = None
        self.gradients = None
        self.target_layer = self.model.extra_convs[-1]
        self.target_layer.register_forward_hook(self._save_feature_maps)
        self.target_layer.register_backward_hook(self._save_gradients)

    def _save_feature_maps(self, module, input, output):
        self.feature_maps = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def predict(self, image, return_heatmap=False):
        if self.model is None:
            print("Model not loaded. Prediction aborted.")
            return None

        try:
            if image is None:
                print("Invalid image input: None")
                return None

            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image

            input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

            if return_heatmap:
                input_tensor.requires_grad_(True)  # Enable gradients for heatmap
                output = self.model(input_tensor)  # Forward pass with gradients
            else:
                with torch.no_grad():  # No gradients if only predicting probability
                    output = self.model(input_tensor)

            probability = torch.sigmoid(output).item()

            if not return_heatmap:
                return probability

            # Compute heatmap
            self.model.zero_grad()
            output.backward()

            if self.feature_maps is None or self.gradients is None:
                print("Warning: Failed to capture feature maps or gradients for heatmap.")
                return probability, None

            gradients = self.gradients.cpu().data.numpy()[0]
            feature_maps = self.feature_maps.cpu().data.numpy()[0]
            weights = np.mean(gradients, axis=(1, 2))
            heatmap = np.zeros(feature_maps.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights):
                heatmap += w * feature_maps[i]
            heatmap = np.maximum(heatmap, 0)
            heatmap = heatmap / (np.max(heatmap) + 1e-10)
            heatmap = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_LINEAR)

            heatmap_rgb = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)

            return probability, heatmap_rgb

        except Exception as e:
            print("Error during prediction:", e)
            return None if not return_heatmap else (None, None)

    def overlay_heatmap(self, original_image, heatmap, alpha=0.5):
        if original_image.shape[:2] != heatmap.shape[:2]:
            original_image = cv2.resize(original_image, (heatmap.shape[1], heatmap.shape[0]))
        if len(original_image.shape) == 3 and original_image.shape[2] == 3:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)



# Example usage
if __name__ == "__main__":
    detector = AIDetector()
    image = cv2.imread("path/to/image.jpg")
    
    # Just probability
    prob = detector.predict(image)
    print(f"Probability: {prob}")
    
    # Probability + heatmap
    prob, heatmap = detector.predict(image, return_heatmap=True)
    if heatmap is not None:
        result = detector.overlay_heatmap(image, heatmap)
        cv2.imwrite("heatmap_result.jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))