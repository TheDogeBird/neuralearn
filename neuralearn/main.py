import os
import torch
import tensorflow as tf
from torchvision import datasets, transforms

from brain.components.MainTFB.tf_brain_model import MainTFBrain
from brain.components.amygdala.amygdala import Amygdala
from brain.components.hippocampus.hippocampus import Hippocampus
from brain.components.occipital.occipital import OccipitalLobe
from brain.components.temporal.temporal import TemporalLobe

DATA_DIR = 'path_to_dataset_directory'
CHECKPOINT_DIR = "E:\\seriousprojects\\neuralearn\\checkpoints"
BATCH_SIZE = 32

def load_dataset():
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader

def train_amygdala_with_maml(amygdala, train_loader):
    optimizer = torch.optim.Adam(amygdala.parameters(), lr=0.001)
    for support_data, query_data in train_loader:
        meta_loss = amygdala.maml_inner_loop(support_data, query_data)
        optimizer.zero_grad()
        meta_loss.backward()
        optimizer.step()

def save_model_weights(model, model_name):
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"{model_name}_checkpoint.pth"))

def load_model_weights(model, model_name):
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, f"{model_name}_checkpoint.pth")))

def initialize_brain_components():
    params = {
        'amygdala': {
            'input_size': 602112,
            'hidden_size': 256,
            'output_size': 64
        },
        'hippocampus': {
            'input_size': 10000,
            'hidden_size': 256,
            'output_size': 64
        },
        'occipital_lobe': {
            'num_classes': 128,
            'input_size': (448, 448)
        },
        'temporal_lobe': {
            'output_size': 128
        }
    }

    amygdala = Amygdala(**params['amygdala'])
    hippocampus = Hippocampus(**params['hippocampus'])
    occipital_lobe = OccipitalLobe(**params['occipital_lobe'])
    temporal_lobe = TemporalLobe(**params['temporal_lobe'])

    return amygdala, hippocampus, occipital_lobe, temporal_lobe

def main():
    amygdala, hippocampus, occipital_lobe, temporal_lobe = initialize_brain_components()

    # Load pretrained weights
    load_model_weights(amygdala, 'amygdala')
    load_model_weights(hippocampus, 'hippocampus')
    load_model_weights(occipital_lobe, 'occipital_lobe')
    load_model_weights(temporal_lobe, 'temporal_lobe')

    amygdala.eval()
    hippocampus.eval()
    occipital_lobe.eval()
    temporal_lobe.eval()

    train_loader, _ = load_dataset()
    train_amygdala_with_maml(amygdala, train_loader)

    rnn_units = 512
    input_dim = sum([comp.output_size for comp in [amygdala, hippocampus, occipital_lobe, temporal_lobe]])
    output_dim = 64

    brain = MainTFBrain(num_classes=output_dim, amygdala_model=amygdala, hippocampus_model=hippocampus,
                        occipital_lobe_model=occipital_lobe, temporal_lobe_model=temporal_lobe)

    test_input = tf.random.uniform((1, 448, 448, 3))
    input_data = {'x': test_input, 'hidden': None}

    output, new_state, attention_weights = brain(input_data)
    print("Output:", output)
    print("New State:", new_state)
    print("Attention Weights:", attention_weights)

    demonstrate_hippocampus_memory(hippocampus)

def demonstrate_hippocampus_memory(hippocampus):
    st_mem, lt_mem = hippocampus.recall_memory()
    print("\n---- Hippocampus Memory Demonstration ----")
    print("Short Term Memory:", st_mem)
    print("Long Term Memory:", lt_mem)

    hippocampus.clear_memory()

    st_mem, lt_mem = hippocampus.recall_memory()
    print("\nAfter Clearing:")
    print("Short Term Memory:", st_mem)
    print("Long Term Memory:", lt_mem)

if __name__ == "__main__":
    main()