import tensorflow as tf
from brain.components.MainTFB.tf_brain_model import MainTFBrain
from brain.components.amygdala.amygdala import Amygdala
from brain.components.hippocampus.hippocampus import Hippocampus
from brain.components.occipital.occipital import OccipitalLobe
from brain.components.temporal.temporal import TemporalLobe

def initialize_brain_components():
    # Parameters for each component
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
            'input_size': 128,
            'hidden_size': 256,
            'output_size': 128
        },
        'temporal_lobe': {
            'input_size': 128,
            'hidden_size': 256,
            'output_size': 128
        }
    }

    # Initialization of each component
    amygdala = Amygdala(**params['amygdala'])
    hippocampus = Hippocampus(**params['hippocampus'])
    occipital_lobe = OccipitalLobe(**params['occipital_lobe'])
    temporal_lobe = TemporalLobe(**params['temporal_lobe'])

    return amygdala, hippocampus, occipital_lobe, temporal_lobe

def main():
    # Get initialized components
    amygdala, hippocampus, occipital_lobe, temporal_lobe = initialize_brain_components()

    # Set PyTorch models to evaluation mode (assuming you're not training them here)
    amygdala.eval()
    hippocampus.eval()
    occipital_lobe.eval()
    temporal_lobe.eval()

    # Parameters for MainTFBrain
    rnn_units = 512
    input_dim = sum([comp.output_size for comp in [amygdala, hippocampus, occipital_lobe, temporal_lobe]])
    output_dim = 64  # Adjust based on your needs

    # Initialize MainTFBrain with the created components
    brain = MainTFBrain(rnn_units=rnn_units, input_dim=input_dim, output_dim=output_dim,
                        amygdala_model=amygdala, hippocampus_model=hippocampus,
                        occipital_lobe_model=occipital_lobe, temporal_lobe_model=temporal_lobe)

    # Dummy data for testing (adjust this to your actual data format and dimensionality)
    test_input = tf.random.uniform((1, 448, 448, 3))
    input_data = {
        'x': test_input,
        'hidden': None  # replace with an actual initial hidden state if needed
    }

    # Call the brain model with the input data
    output, new_state, attention_weights = brain(input_data)

    # Display results
    print("Output:", output)
    print("New State:", new_state)
    print("Attention Weights:", attention_weights)

    # Demonstrating the Hippocampus functionalities
    demonstrate_hippocampus_memory(hippocampus)

def demonstrate_hippocampus_memory(hippocampus):
    print("\n---- Hippocampus Memory Demonstration ----")
    # Store and recall memories
    st_mem, lt_mem = hippocampus.recall_memory()
    print("Short Term Memory:", st_mem)
    print("Long Term Memory:", lt_mem)

    # Clear memory (demonstration purposes)
    hippocampus.clear_memory()

    # Check memories post-clearing
    st_mem, lt_mem = hippocampus.recall_memory()
    print("\nAfter Clearing:")
    print("Short Term Memory:", st_mem)
    print("Long Term Memory:", lt_mem)

if __name__ == "__main__":
    main()
