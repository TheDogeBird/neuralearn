import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pyaudio

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
            for module_name, module in self.interaction_modules.items():
                if hasattr(module, "react_to_audio"):
                    module.react_to_audio(output)
        except Exception as e:
            print(f"Error while listening: {e}")

    def extract_features(self, audio_data):
        audio_tensor = torch.tensor(audio_data).float().unsqueeze(0)
        specgram = torch.stft(audio_tensor, n_fft=400, hop_length=160, win_length=400, window=torch.hamming_window(400))
        power_spectrum = specgram.pow(2).sum(-1)
        features = power_spectrum.mean(-1).unsqueeze(0)
        return features

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
