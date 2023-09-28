import tensorflow as tf
from brain.components.MainTFB.tf_brain_model import MainTFBrain
from brain.components.amygdala.amygdala import Amygdala
from brain.components.hippocampus.hippocampus import Hippocampus
from brain.components.occipital.occipital import OccipitalLobe
from brain.components.temporal.temporal import TemporalLobe

# Parameters
amygdala_input_size = 128
amygdala_hidden_size = 256
amygdala_output_size = 64

hippocampus_input_size = 128
hippocampus_hidden_size = 256
hippocampus_output_size = 64

occipital_input_size = 128
occipital_hidden_size = 256
occipital_output_size = 128

temporal_input_size = 128
temporal_hidden_size = 256
temporal_output_size = 128

# Initialize the components of the brain
amygdala = Amygdala(amygdala_input_size, amygdala_hidden_size, amygdala_output_size)
hippocampus = Hippocampus(hippocampus_input_size, hippocampus_hidden_size, hippocampus_output_size)
occipital_lobe = OccipitalLobe()
temporal_lobe = TemporalLobe()

# Set the PyTorch models to evaluation mode
amygdala.eval()
hippocampus.eval()
occipital_lobe.eval()
temporal_lobe.eval()

# Set up the parameters for MainTFBrain
rnn_units = 512
input_dim = 128  # replace with the actual input dimension
output_dim = 64  # replace with the actual output dimension

# Initialize MainTFBrain with the created components
brain = MainTFBrain(rnn_units=rnn_units, input_dim=input_dim, output_dim=output_dim,
                    amygdala_model=amygdala, hippocampus_model=hippocampus,
                    occipital_lobe_model=occipital_lobe, temporal_lobe_model=temporal_lobe)

# To run a test, you can create some dummy input data and call the brain model
test_input = tf.random.uniform((1, input_dim))
hidden_state = None  # replace with actual initial hidden state if needed

# Call the brain model with the test input
output, new_state, attention_weights = brain.call(test_input, hidden_state)

# Print the results
print("Output:", output)
print("New State:", new_state)
print("Attention Weights:", attention_weights)
