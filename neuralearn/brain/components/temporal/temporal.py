import torch
import torch.nn as nn
import pyaudio
import numpy as np
import librosa
from scipy.fftpack import dct


class TemporalLobe(nn.Module):
    def __init__(self, interaction_modules=None):
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

    def forward(self, x, mode="auditory"):
        if mode == "auditory":
            # LSTM forward pass for auditory data
            x, _ = self.lstm(x)
            x = x[:, -1, :]

        elif mode == "language":
            # Transformer forward pass for textual/language data
            x = self.transformer_encoder(x)
            x = self.fc(x)

        return x

    def listen(self):
        # Get audio data
        audio_data = self.stream.read(1024)
        audio_data = np.frombuffer(audio_data, dtype=np.int16)

        # Extract features (e.g., MFCCs)
        features = self.extract_features(audio_data)
        output = self.forward(features, mode="auditory")

        # Send the recognized patterns to other parts of the "brain"
        for module_name, module in self.interaction_modules.items():
            if module_name == "occipital":
                module.visual_verification(output)
            else:
                module.react(output)

    def extract_features(audio_data, sr=44100, n_mfcc=13):
        # Compute MFCCs using librosa
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
        return torch.tensor(mfccs).float()

    def add_interaction_module(self, module_name, module):
        self.interaction_modules[module_name] = module

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

