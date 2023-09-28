import torch
import torch.nn as nn


class Hippocampus(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, interaction_modules=None):
        super(Hippocampus, self).__init__()

        # Basic processing layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(hidden_size, memory_size)

        # Memory storage - using tensors to simulate short-term and long-term memory
        self.short_term_memory = torch.randn((memory_size, input_size))
        self.long_term_memory = torch.randn((memory_size, input_size))

        # Spatial processing (contextualization) - for the sake of our model, let's use an additional
        # fully connected layer to process this 'spatial' context
        self.spatial_processing = nn.Linear(input_size, input_size)

        # Interaction Mechanism - List of modules this module can interact with
        self.interaction_modules = interaction_modules or []

    def forward(self, x):
        # Basic flow
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = self.fc3(x)

        # Memory processing - Here we can determine if the input is to be stored or if it's a trigger to
        # retrieve a memory. For our basic example, we'll just assume all input is to be stored.
        self.store_memory(x)

        # Spatial processing - contextualize the information
        x = self.spatial_processing(x)

        # Similar to our amygdala model, we can incorporate interaction with other components here.
        # This can be expanded upon later.

        return x

    def store_memory(self, x):
        # For simplicity, we're just storing every input into our short-term memory
        # and periodically moving data to long-term memory. This can be enhanced.
        self.short_term_memory = torch.roll(self.short_term_memory, shifts=-1, dims=0)
        self.short_term_memory[-1] = x

        # Periodically move data to long-term memory (For our example, we'll assume after every 100 inputs)
        # This is a simplistic approach.
        if torch.rand(1).item() > 0.99:
            self.long_term_memory = torch.roll(self.long_term_memory, shifts=-1, dims=0)
            self.long_term_memory[-1] = self.short_term_memory[0]

