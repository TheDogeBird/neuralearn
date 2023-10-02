import torch
import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
import pyaudio
import speech_recognition as sr

class TemporalLobe(nn.Module):
    def __init__(self, input_size=None, hidden_size=None, output_size=None, interaction_modules=None):
        super(TemporalLobe, self).__init__()

        # LSTM for auditory processing
        self.lstm = nn.LSTM(input_size=13, hidden_size=128, num_layers=2, batch_first=True)

        # Transformer for language understanding
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=512, nhead=8), num_layers=6)
        self.fc = nn.Linear(512, 256)

        # Interaction Mechanism with other brain components
        self.interaction_modules = interaction_modules or {}

        # Setting up microphone input
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

        # Placeholder tensor to hold incoming data from the 'Occipital' network
        self.occipital_output_tensor = torch.zeros(1, 128)  # Adjust shape as needed

        # Initialize the speech recognizer
        self.recognizer = sr.Recognizer()

        # Curiosity-related parameters
        self.curiosity_threshold = 0.5  # Adjust this threshold as needed

    def forward(self, x, mode="auditory"):
        if mode == "auditory":
            x, _ = self.lstm(x)
            x = x[:, -1, :]
        elif mode == "language":
            x = self.transformer_encoder(x)
            x = self.fc(x)
        return x

    def listen(self):
        try:
            audio_data = self.stream.read(1024)
            audio_data = np.frombuffer(audio_data, dtype=np.int16)
            audio_data = audio_data / 32767.
            features = self.extract_features(audio_data)
            output = self.forward(features, mode="auditory")

            # Perform speech recognition
            recognized_text = self.speech_recognition(audio_data)

            # Check for curiosity and take actions
            if self.check_curiosity(output, recognized_text):
                self.take_curiosity_actions()

            for module_name, module in self.interaction_modules.items():
                if hasattr(module, "react_to_audio"):
                    module.react_to_audio(output, recognized_text)
        except Exception as e:
            print(f"Error while listening: {e}")

    def extract_features(self, audio_data):
        audio_tensor = torch.tensor(audio_data).float().unsqueeze(0)
        specgram = torch.stft(audio_tensor, n_fft=400, hop_length=160, win_length=400, window=torch.hamming_window(400))
        power_spectrum = specgram.pow(2).sum(-1)
        features = power_spectrum.mean(-1).unsqueeze(0)
        return features

    def speech_recognition(self, audio_data):
        try:
            with sr.AudioFile(audio_data) as source:
                audio = self.recognizer.record(source)
                recognized_text = self.recognizer.recognize_google(audio)
                return recognized_text
        except Exception as e:
            print(f"Error during speech recognition: {e}")
            return ""

    def check_curiosity(self, auditory_features, recognized_text):
        # Implement logic to check for curiosity based on auditory features and recognized text
        # Return True if curiosity is detected, otherwise False
        pass

    def take_curiosity_actions(self):
        # Define actions to be taken when curiosity is detected
        # These actions can include capturing additional data, asking questions, or exploration
        pass

    def react_to_occipital_output(self, np_data):
        self.occipital_output_tensor = torch.from_numpy(np_data).float()

    def add_interaction_module(self, module_name, module):
        self.interaction_modules[module_name] = module

    def close(self):
        try:
            self.stream.stop_stream()
            self.stream.close()
        except Exception as e:
            print(f"Error while closing the stream: {e}")
        finally:
            self.p.terminate()

    def recognize_pattern_from_camera(self, max_attempts=10):
        """
        Recognize a pattern from the camera feed using the OccipitalLobe.
        :param max_attempts: Maximum number of attempts to capture and process a frame.
        :return: Recognized pattern features.
        """
        frame = self.capture_and_process(max_attempts)
        if frame is not None:
            return self.process_image(frame)
        else:
            return None

    def process_image(self, image):
        """
        Process an image (already in the appropriate format) using the OccipitalLobe.
        :param image: Torch tensor representing an image.
        :return: Pattern recognition features.
        """
        with torch.no_grad():
            image = image.to(self.device)
            _, pattern_features = self.forward(image)
            return pattern_features

    def recognize_pattern(self, image):
        """
        Recognize a pattern in an image using the OccipitalLobe.
        :param image: Torch tensor representing an image.
        :return: Recognized pattern features.
        """
        return self.process_image(image)

    def set_device(self, device):
        """
        Set the device (CPU or GPU) for processing.
        :param device: Torch device.
        """
        self.device = device

    def load_weights(self, weights_path):
        """
        Load pre-trained weights for the OccipitalLobe.
        :param weights_path: Path to the weights file.
        """
        try:
            self.load_state_dict(torch.load(weights_path))
            print(f"Loaded weights from {weights_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")
