import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset

# Constants
CHECKPOINT_DIR = "/checkpoints/"


# Sample Dataset. Replace with your own dataset structure.
class SampleDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(1000, 602112)  # Just a placeholder
        self.labels = torch.randint(0, 10, (1000,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class Amygdala(nn.Module):
    def __init__(self, input_size=602112, hidden_size=1024, output_size=10, interaction_modules=None):
        super(Amygdala, self).__init__()

        # Basic structure
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

        # Interaction Mechanism with other brain components
        self.interaction_modules = interaction_modules or {}

        # Initializing the Hippocampus instance
        self.hippocampus = self.interaction_modules.get("hippocampus", None)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.softmax(x)

        if self.hippocampus:
            self.hippocampus.st_memory.append(x.detach().clone())
            if len(self.hippocampus.st_memory) > self.hippocampus.st_memory_size:
                del self.hippocampus.st_memory[0]

        return x


def save_model_weights(model, model_name):
    path = os.path.join(CHECKPOINT_DIR, f"{model_name}_checkpoint.pth")
    try:
        torch.save(model.state_dict(), path)
        print(f"Model weights saved to {path}")
    except Exception as e:
        print(f"Error saving model weights to {path}. Error: {e}")


def load_model_weights(model, model_name):
    path = os.path.join(CHECKPOINT_DIR, f"{model_name}_checkpoint.pth")
    if os.path.exists(path):
        try:
            model.load_state_dict(torch.load(path))
            print(f"Loaded model weights from {path}")
        except Exception as e:
            print(f"Error loading model weights from {path}. Error: {e}")
    else:
        print(f"No checkpoint found at {path}")


def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct_predictions += predicted.eq(labels).sum().item()
    accuracy = 100. * correct_predictions / len(loader.dataset)
    return total_loss / len(loader), accuracy


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Placeholder. You need to properly instantiate the Hippocampus model.
    hippocampus_instance = None

    model = Amygdala(interaction_modules={"hippocampus": hippocampus_instance}).to(device)

    dataset = SampleDataset()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    epochs = 10
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        test_loss, accuracy = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {accuracy:.2f}%")

        # Save model weights
        save_model_weights(model, 'amygdala')

    print("Training finished.")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Amygdala().to(device)

    # Load previous model weights (if any)
    load_model_weights(model, 'amygdala')

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    main()
