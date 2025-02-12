import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import resnet50, ResNet50_Weights
import cv2
from PIL import Image

# If your CustomResNet50 model is not already defined elsewhere, define it here.
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
        self.fc2 = nn.Linear(128, 1)  # Single output logit

    def forward(self, x):
        x = self.base(x)
        x = self.extra_convs(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class AIDetector:
    def __init__(self, model_path, device=None):
        """
        Initializes the AI detector.
        
        Parameters:
            model_path (str): Path to the model weights file for torch_resnet50_base_additional_convs-1_model.
            device (torch.device, optional): Device to run the model on (e.g., 'cuda' or 'cpu').
        """
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        self.model = CustomResNet50(extra_conv_layers=5).to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print("Model loaded successfully from", model_path)
        except Exception as e:
            print("Error loading model:", e)
            self.model = None

        # Define the transforms that match your training.
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert a NumPy array (RGB) to a PIL image.
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        """
        Accepts an image (from your U²-Net pipeline) and returns the probability that the image is AI generated.
        
        Parameters:
            image (np.array): The input image as a NumPy array. (Assumed to be in BGR format.)
            
        Returns:
            float: The probability (0-1) that the image is AI generated, or None on error.
        """
        if self.model is None:
            print("Model not loaded. Prediction aborted.")
            return None

        try:
            if image is None:
                print("Invalid image input: None")
                return None

            # Convert from BGR to RGB if necessary.
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Apply the transformation.
            input_tensor = self.transform(image)
            input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension.
            input_tensor = input_tensor.to(self.device)

            with torch.no_grad():
                output = self.model(input_tensor)
                probability = torch.sigmoid(output).item()  # Apply sigmoid to get probability.
            
            return probability

        except Exception as e:
            print("Error during prediction:", e)
            return None
