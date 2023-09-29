import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset


class Hippocampus(nn.Module):
    MEMORY_SIZE = 100
    INTERMEDIATE_SIZE = 500

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
        self.short_term_memory = torch.randn((memory_size, input_size))
        self.long_term_memory = torch.randn((memory_size, input_size))

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

    def store_memory(self, x):
        self.short_term_memory = torch.roll(self.short_term_memory, shifts=-1, dims=0)
        self.short_term_memory[-1] = x.squeeze()

        # Move short term memories to long term with a probability of 0.01
        if torch.rand(1).item() > 0.99:
            self.long_term_memory = torch.roll(self.long_term_memory, shifts=-1, dims=0)
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

    def maml_inner_loop(self, support_data, query_data, num_adaptation_steps=1, step_size=0.1):
        loss_fn = F.cross_entropy
        adapted_state_dict = self.state_dict()
        for step in range(num_adaptation_steps):
            support_predictions = self(support_data[0], mode="language")
            loss = loss_fn(support_predictions, support_data[1])
            grads = torch.autograd.grad(loss, list(self.parameters()), create_graph=True)  # Convert to list
            adapted_state_dict = {name: param - step_size * grad for (name, param), grad in
                                  zip(self.named_parameters(), grads)}
        self.load_state_dict(adapted_state_dict)
        query_predictions = self(query_data[0], mode="language")
        meta_loss = loss_fn(query_predictions, query_data[1])
        return meta_loss


# Sample Dataset. Replace with your own dataset structure.
class SampleDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(1000, 602112)  # Just a placeholder
        self.labels = torch.randint(0, 10, (1000,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Hippocampus(input_size=602112, hidden_size=1024, output_size=602112).to(device)

    dataset = SampleDataset()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    epochs = 10
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
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")

    print("Training finished.")


if __name__ == "__main__":
    main()
