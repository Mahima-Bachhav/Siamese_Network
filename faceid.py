from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image as KivyImage
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import os
import numpy as np
from PIL import Image as PILImage
import torchvision.transforms as transforms
import torch.nn.functional as F


# Define the L1 Distance layer
# class L1Dist(nn.Module):
#     def __init__(self, **kwargs):
#         super().__init__()

#     def forward(self, input_embedding, validation_embedding):
#         return torch.abs(input_embedding - validation_embedding)

# Siamese Model architecture
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        # Shared feature extractor
        self.conv1 = nn.Conv2d(3, 64, kernel_size=10, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=7, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=4, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=1)
        self.fc1 = nn.Linear(256 * 5 * 5, 4096)

        # Classifier
        self.fc2 = nn.Linear(4096, 1)
        self.sigmoid = nn.Sigmoid()

    def forward_one(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, input1, input2):
        embedding1 = self.forward_one(input1)
        embedding2 = self.forward_one(input2)
        
        # Compute L1 Distance
        distance = torch.abs(embedding1 - embedding2)
        
        # Classifier
        output = self.sigmoid(self.fc2(distance))
        return output

class CamApp(App):

    def build(self):
        self.web_cam = KivyImage(size_hint=(1, .8))
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1, .1))
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1, .1))

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # Load PyTorch model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SiameseNetwork().to(self.device)
        self.model.load_state_dict(torch.load('siamese_model_normalized.pth', map_location=self.device))
        self.model.eval()

        # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 33.0)

        return layout

    def update(self, *args):
        ret, frame = self.capture.read()
        frame = frame[120:120 + 250, 200:200 + 250, :]

        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    def preprocess(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = PILImage.fromarray(img)

        transform = transforms.Compose([
            transforms.Resize((100, 100)), 
            transforms.ToTensor(),         
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        img_tensor = transform(img).unsqueeze(0) 
        return img_tensor

    def verify(self, *args):
        detection_threshold = 0.99
        verification_threshold = 0.8
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:120 + 250, 200:200 + 250, :]
        cv2.imwrite(SAVE_PATH, frame)

        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))

            # Make predictions
            with torch.no_grad():
                result = self.model(input_img, validation_img).item()
            results.append(result)

        # Detection Threshold: Metric above which a prediction is considered positive
        detection = np.sum(np.array(results) > detection_threshold)

        # Verification Threshold: Proportion of positive predictions / total positive samples
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images')))
        verified = verification > verification_threshold

        # Set verification text
        self.verification_label.text = 'Verified' if verified else 'Unverified'

        # Log out details
        Logger.info(f"Results: {results}")
        Logger.info(f"Detection: {detection}")
        Logger.info(f"Verification: {verification}")
        Logger.info(f"Verified: {verified}")

        return results, verified


if __name__ == '__main__':
    CamApp().run()
