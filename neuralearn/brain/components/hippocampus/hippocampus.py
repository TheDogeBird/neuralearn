import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.testing._internal.common_nn import input_size
from torch.utils.data import DataLoader, random_split, Dataset

class Hippocampus(nn.Module):
    MEMORY_SIZE = 12
    INTERMEDIATE_SIZE = 20

    def __init__(self, input_size, hidden_size, output_size, memory_size=MEMORY_SIZE,
                 intermediate_size=INTERMEDIATE_SIZE, interaction_modules=None):
        super(Hippocampus, self).__init__()

        # Basic processing layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_size, input_size)

        # Memory storage
        self.short_term_memory = nn.Parameter(torch.randn((memory_size, input_size)))
        self.long_term_memory = nn.Parameter(torch.randn((memory_size, input_size)))

        # Spatial processing (contextualization)
        self.spatial_processing = nn.Linear(input_size, intermediate_size)

        # Interaction Mechanism with other brain components
        self.interaction_modules = interaction_modules or {}

    def forward(self, x):
        x = x.view(x.size(0), -1)

        # Basic flow
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)

        # Memory processing
        self.store_memory(x)

        # Spatial processing
        x = self.spatial_processing(x)
        return x

    def init_memory(self):
        # Initialize memory storage (short-term and long-term memory)
        self.short_term_memory = nn.Parameter(torch.randn((self.MEMORY_SIZE, 602112)))
        self.long_term_memory = nn.Parameter(torch.randn((self.MEMORY_SIZE, 602112)))

    def store_memory(self, x):
        self.short_term_memory = nn.Parameter(torch.roll(self.short_term_memory, shifts=-1, dims=0))
        self.short_term_memory[-1] = x.squeeze()

        # Move short term memories to long term with a probability of 0.01
        if torch.rand(1).item() > 0.99:
            self.long_term_memory = nn.Parameter(torch.roll(self.long_term_memory, shifts=-1, dims=0))
            self.long_term_memory[-1] = self.short_term_memory[0]

    def retrieve_similar_memories(self, x, k=5, memory_type='short_term'):
        """Retrieve k most similar memories."""
        if memory_type == 'short_term':
            memory = self.short_term_memory
        else:
            memory = self.long_term_memory

        # Compute cosine similarities
        x = x / x.norm(dim=1, keepdim=True)
        memory = memory / memory.norm(dim=1, keepdim=True)
        similarities = torch.mm(x, memory.t())

        _, indices = torch.topk(similarities, k=k, dim=1)
        return indices

    def add_interaction_module(self, module_name, module):
        self.interaction_modules[module_name] = module


# Sample Dataset. Replace with your own dataset structure.
class SampleDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(1000, 602112)  # Just a placeholder
        self.labels = torch.randint(0, 10, (1000,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def initialize_hippocampus(input_size, hidden_size, output_size, memory_size, intermediate_size):
    """
    Initialize and configure the Hippocampus neural network.

    Args:
    - input_size (int): The input size of the Hippocampus.
    - hidden_size (int): The hidden size of the Hippocampus.
    - output_size (int): The output size of the Hippocampus.
    - memory_size (int): The size of the memory storage in the Hippocampus.
    - intermediate_size (int): The size of the intermediate layer for spatial processing.

    Returns:
    - model (Hippocampus): The initialized Hippocampus neural network.
    - train_loader (DataLoader): The DataLoader for the training dataset.
    - test_loader (DataLoader): The DataLoader for the testing dataset.
    """
    # Initialize the Hippocampus neural network
    model = Hippocampus(input_size, hidden_size, output_size, memory_size, intermediate_size)

    # Create a sample dataset (Replace this with your own dataset structure)
    dataset = SampleDataset()

    # Split the dataset into training and testing sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoader for training and testing datasets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    return model, train_loader, test_loader

def train_hippocampus(model, train_loader, optimizer, criterion, epochs):
    """
    Train the Hippocampus neural network.

    Args:
    - model (Hippocampus): The initialized Hippocampus neural network.
    - train_loader (DataLoader): The DataLoader for the training dataset.
    - optimizer (optim.Optimizer): The optimizer for training.
    - criterion (nn.Module): The loss criterion for training.
    - epochs (int): The number of training epochs.

    Returns:
    - avg_epoch_loss (float): The average loss over all training epochs.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    avg_epoch_loss = 0  # Initialize avg_epoch_loss here

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for data, _ in train_loader:  # We do not use labels here
            data = data.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data)  # Comparing outputs with the original data
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_epoch_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss}")

    return avg_epoch_loss


def main():
    input_size = 602112
    hidden_size = 1024
    output_size = 602112
    memory_size = 12
    intermediate_size = 20
    epochs = 10

    model, train_loader, test_loader = initialize_hippocampus(input_size, hidden_size, output_size, memory_size, intermediate_size)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    avg_epoch_loss = train_hippocampus(model, train_loader, optimizer, criterion, epochs)
    print("Training finished. Average epoch loss:", avg_epoch_loss)


if __name__ == "__main__":
    main()
