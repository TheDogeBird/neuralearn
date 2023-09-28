import torch
import torch.nn as nn


class Amygdala(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, interaction_modules=None):
        super(Amygdala, self).__init__()

        # Basic structure
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

        # Interaction Mechanism - List of modules this module can interact with
        # For now, this is a placeholder. In the future, this can be used to
        # query other modules (like hippocampus for memories or temporal for sounds)
        # when processing emotions.
        self.interaction_modules = interaction_modules or []

    def forward(self, x):
        # Basic flow
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = self.fc3(x)

        # At this juncture, we can incorporate interaction with other components.
        # For instance, based on initial emotion prediction, we can query the
        # hippocampus to verify if a related memory might change the emotional context.
        # But for simplicity, we're skipping it in this example.

        return self.softmax(x)