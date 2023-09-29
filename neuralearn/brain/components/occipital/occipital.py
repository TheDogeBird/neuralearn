import torch
import torch.nn as nn
import cv2
import tensorflow as tf


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

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        return x

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
                    frame = frame.unsqueeze(0).to(next(self.parameters()).device)
                    return frame
                else:
                    print("Failed to capture frame. Retrying...")
                    attempt += 1
            except Exception as e:
                print(f"Error during capture and processing: {e}. Retrying...")
                attempt += 1

        print("Max attempts reached. Giving up.")
        return None


def pytorch_to_tensorflow(tensor):
    return tf.convert_to_tensor(tensor.cpu().detach().numpy())


class Brain(tf.keras.Model):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Brain, self).__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, **kwargs):
        x = self.dense1(inputs)
        return self.dense2(x)


if __name__ == "__main__":
    occipital_lobe = OccipitalLobe(num_classes=128, input_size=(448, 448))
    occipital_lobe.eval()  # set the model in evaluation mode

    processed_image = occipital_lobe.capture_and_process()

    if processed_image is not None:
        with torch.no_grad():  # Ensure no gradients are computed for evaluation
            pytorch_output = occipital_lobe(processed_image)
        tf_input = pytorch_to_tensorflow(pytorch_output)

        # Make sure tensorflow uses CPU to avoid potential GPU conflicts
        with tf.device('/CPU:0'):
            brain = Brain(input_size=128, hidden_size=64, num_classes=10)
            brain_output = brain(tf_input)

        print(brain_output)
