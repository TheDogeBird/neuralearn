import torch
import torch.nn as nn
import cv2

class OccipitalLobe(nn.Module):
    def __init__(self, num_classes=128, input_size=(448, 448)):
        super(OccipitalLobe, self).__init__()
        self.input_size = input_size

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, num_classes)

        # Pattern recognition
        self.pattern_recognition = nn.Sequential(
            nn.Linear(num_classes, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))

        # Pattern recognition
        pattern_features = self.pattern_recognition(x)

        return x, pattern_features

    def capture_and_process(self, max_attempts=10):
        attempt = 0
        while attempt < max_attempts:
            try:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("Camera not available. Retrying...")
                    attempt += 1
                    continue

                ret, frame = cap.read()
                cap.release()

                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                    frame = cv2.resize(frame, self.input_size)
                    frame = torch.tensor(frame).permute(2, 0, 1).float()
                    frame = (frame / 255.0 - 0.5) * 2.0
                    frame = frame.unsqueeze(0)
                    return frame
                else:
                    print("Failed to capture frame. Retrying...")
                    attempt += 1
            except Exception as e:
                print(f"Error during capture and processing: {e}. Retrying...")
                attempt += 1

        print("Max attempts reached. Giving up.")
        return None
