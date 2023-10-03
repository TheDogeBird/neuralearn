## Neuron Model

---

### 1. Membrane Potential

The **membrane potential** (`V_m`) is the voltage difference across the neuron's cell membrane. It fluctuates based on the inputs the neuron receives and determines whether the neuron will fire an action potential.

### 2. Firing Threshold

The **firing threshold** is the membrane potential value at which the neuron fires an action potential. When `V_m` reaches this value, the neuron produces a spike.

### 3. Refractory Period

After firing, there's a period during which the neuron cannot fire again, ensuring signal clarity and directionality.

#### Full Code:
```python
import numpy as np

class Neuron:
    def __init__(self):
        self.V_m = -70  # Initial membrane potential in mV (resting potential)
        self.resting_potential = -70  # Resting potential in mV
        self.capacitance = 1.0  # Capacitance of the neuronal membrane
        self.threshold = -55  # Firing threshold in mV
        self.refractory_period = 2  # Refractory period in ms
        self.last_spike_time = None  # Time of the last spike
        self.synaptic_inputs = []  # List to store incoming synaptic activities

    class Synapse:
        def __init__(self, pre_neuron, weight, delay, neurotransmitter_type):
            self.pre_neuron = pre_neuron  # Neuron sending the spike
            self.weight = weight  # Strength of the connection
            self.delay = delay  # Time delay for spike propagation
            self.neurotransmitter_type = neurotransmitter_type  # 'excitatory' or 'inhibitory'

        def propagate_spike(self, spike_time):
            """Calculate the effect of an incoming spike."""
            if self.neurotransmitter_type == 'excitatory':
                effect = self.weight
            else:
                effect = -self.weight
            return spike_time + self.delay, effect

    def receive_spike(self, synapse, spike_time):
        """Handle incoming spikes through synapses."""
        new_spike_time, effect = synapse.propagate_spike(spike_time)
        self.synaptic_inputs.append((new_spike_time, effect))

    def integrate_inputs(self, current_time):
        """Integrate inputs over time and adjust the membrane potential."""
        total_effect = 0
        new_inputs = []
        for spike_time, effect in self.synaptic_inputs:
            if spike_time <= current_time:
                total_effect += effect
            else:
                new_inputs.append((spike_time, effect))
        self.synaptic_inputs = new_inputs
        self.V_m += total_effect

    def check_firing(self, current_time):
        """Check if the neuron should fire based on the membrane potential and refractory period."""
        if (self.V_m >= self.threshold) and (current_time - self.last_spike_time > self.refractory_period):
            self.fire_action_potential(current_time)
            return True
        return False

    def fire_action_potential(self, current_time):
        """Fire an action potential and reset the membrane potential."""
        self.V_m = self.resting_potential
        self.last_spike_time = current_time
        # Here, you'd also propagate the spike to connected neurons
```



### 4. Ion Channels and Dynamics (Optional)

Ion channels control the flow of ions across the neuron's membrane, influencing the membrane potential.

#### Code:
```python
    def __init__(self):
        self.Na_concentration = 50  # Sodium ion concentration
        self.K_concentration = 20   # Potassium ion concentration
        self.Ca_concentration = 0.05  # Calcium ion concentration
        self.ion_channels = {"Na": True, "K": True, "Ca": False}  # Active ion channels
```


### 5. Receiving Inputs

Neurons receive inputs through synapses, which are connections from other neurons. Each synapse has a weight, which determines the strength of the connection. Additionally, synapses can be excitatory (increasing the membrane potential) or inhibitory (decreasing the membrane potential). We'll also consider a time delay for the propagation of the spike from one neuron to another.

#### Code:
```python
class Neuron:
    def __init__(self):
        # ... (previous initialization code)
        self.synaptic_inputs = []  # List to store incoming synaptic activities

    class Synapse:
        def __init__(self, pre_neuron, weight, delay, neurotransmitter_type):
            self.pre_neuron = pre_neuron  # Neuron sending the spike
            self.weight = weight  # Strength of the connection
            self.delay = delay  # Time delay for spike propagation
            self.neurotransmitter_type = neurotransmitter_type  # 'excitatory' or 'inhibitory'

        def propagate_spike(self, spike_time):
            """Calculate the effect of an incoming spike."""
            if self.neurotransmitter_type == 'excitatory':
                effect = self.weight
            else:
                effect = -self.weight
            return spike_time + self.delay, effect

    def receive_spike(self, synapse, spike_time):
        """Handle incoming spikes through synapses."""
        new_spike_time, effect = synapse.propagate_spike(spike_time)
        self.synaptic_inputs.append((new_spike_time, effect))
        # In a real-time simulation, you'd adjust V_m based on the effect at the appropriate time
```

In this model, each neuron can have multiple synapses. When a spike is received, its effect on the post-synaptic neuron's membrane potential (`V_m`) is determined by the synapse's weight and neurotransmitter type. The spike also has a time delay associated with it, representing the time it takes for the spike to propagate from the pre-synaptic neuron to the post-synaptic neuron.

### 6. Processing Inputs and Firing

Once a neuron receives inputs, it needs to process them and determine if the accumulated input is sufficient to cause the neuron to fire an action potential. This involves integrating the inputs over time, checking against the firing threshold, and considering the refractory period.

#### Code:
```python
class Neuron:
    def __init__(self):
        # ... (previous initialization code)
        self.last_fired_time = -np.inf  # Time the neuron last fired

    def integrate_inputs(self, current_time):
        """Integrate inputs over time and adjust the membrane potential."""
        total_effect = 0
        new_inputs = []
        for spike_time, effect in self.synaptic_inputs:
            if spike_time <= current_time:
                total_effect += effect
            else:
                new_inputs.append((spike_time, effect))
        self.synaptic_inputs = new_inputs
        self.V_m += total_effect

    def check_firing(self, current_time):
        """Check if the neuron should fire based on the membrane potential and refractory period."""
        if (self.V_m >= self.threshold) and (current_time - self.last_fired_time > self.refractory_period):
            self.fire_action_potential(current_time)
            return True
        return False

    def fire_action_potential(self, current_time):
        """Fire an action potential and reset the membrane potential."""
        self.V_m = self.resting_potential
        self.last_fired_time = current_time
        # Here, you'd also propagate the spike to connected neurons
```

In this model, the `integrate_inputs` method processes the inputs that have arrived up to the current time, adjusting the membrane potential (`V_m`) accordingly. The `check_firing` method then checks if the neuron should fire based on the current membrane potential and the time since the neuron last fired (to account for the refractory period). If the neuron fires, the `fire_action_potential` method resets the membrane potential and records the firing time.



### 7. Resetting the Neuron After Firing

After a neuron fires an action potential, it's crucial to reset certain properties to ensure the neuron can process subsequent inputs correctly. This reset process often involves returning the membrane potential to its resting state and updating the time of the last spike. Additionally, in more complex models, other properties like ion concentrations might be reset.

#### Code for Resetting:
```python
    def reset_after_firing(self):
        """Reset the neuron's properties after firing."""
        self.V_m = self.resting_potential
        # Additional reset logic can be added here if needed
```

By invoking this method after the neuron fires, we ensure that it's ready to process new inputs and fire again when the conditions are met.

---

#### Full Comprehensive Neuron Code:
```python
import numpy as np

class Neuron:
    def __init__(self):
        self.V_m = -70  # Initial membrane potential in mV (resting potential)
        self.resting_potential = -70  # Resting potential in mV
        self.capacitance = 1.0  # Capacitance of the neuronal membrane
        self.threshold = -55  # Firing threshold in mV
        self.refractory_period = 2  # Refractory period in ms
        self.last_spike_time = None  # Time of the last spike
        self.synaptic_inputs = []  # List to store incoming synaptic activities

    class Synapse:
        def __init__(self, pre_neuron, weight, delay, neurotransmitter_type):
            self.pre_neuron = pre_neuron  # Neuron sending the spike
            self.weight = weight  # Strength of the connection
            self.delay = delay  # Time delay for spike propagation
            self.neurotransmitter_type = neurotransmitter_type  # 'excitatory' or 'inhibitory'

        def propagate_spike(self, spike_time):
            """Calculate the effect of an incoming spike."""
            if self.neurotransmitter_type == 'excitatory':
                effect = self.weight
            else:
                effect = -self.weight
            return spike_time + self.delay, effect

    def receive_spike(self, synapse, spike_time):
        """Handle incoming spikes through synapses."""
        new_spike_time, effect = synapse.propagate_spike(spike_time)
        self.synaptic_inputs.append((new_spike_time, effect))

    def integrate_inputs(self, current_time):
        """Integrate inputs over time and adjust the membrane potential."""
        total_effect = 0
        new_inputs = []
        for spike_time, effect in self.synaptic_inputs:
            if spike_time <= current_time:
                total_effect += effect
            else:
                new_inputs.append((spike_time, effect))
        self.synaptic_inputs = new_inputs
        self.V_m += total_effect

    def check_firing(self, current_time):
        """Check if the neuron should fire based on the membrane potential and refractory period."""
        if (self.V_m >= self.threshold) and (current_time - self.last_spike_time > self.refractory_period):
            self.fire_action_potential(current_time)
            self.reset_after_firing()
            return True
        return False

    def fire_action_potential(self, current_time):
        """Fire an action potential and reset the membrane potential."""
        self.V_m = self.resting_potential
        self.last_spike_time = current_time
        # Here, you'd also propagate the spike to connected neurons

    def reset_after_firing(self):
        """Reset the neuron's properties after firing."""
        self.V_m = self.resting_potential
        # Additional reset logic can be added here if needed
```


# Synapse Model

---

## Synaptic Weights and Their Initializations

In neural networks, synaptic weights represent the strength or amplitude of a connection between two neurons. The value of these weights determines the magnitude of the influence one neuron has on another. Proper initialization of these weights is crucial for the convergence and performance of the network.

There are several methods for weight initialization, and the choice often depends on the type of activation function used in the neurons. Some common initialization methods include:

1. **Zero Initialization**: Setting all weights to zero. This is generally not recommended as it leads to the problem where every neuron in the network behaves the same way, making learning impossible.
2. **Random Initialization**: Weights are initialized with small random numbers. This breaks the symmetry but can sometimes lead to slow convergence.
3. **Xavier/Glorot Initialization**: Suitable for the sigmoid and hyperbolic tangent activation functions. The weights are initialized with values drawn from a distribution with zero mean and a specific variance.
4. **He Initialization**: Designed for ReLU activation functions. The weights are initialized with values drawn from a distribution with zero mean and a variance of 2/n, where n is the number of input units.

#### Code:

```python
import numpy as np

class Synapse:
    def __init__(self, input_size, output_size, initialization="xavier"):
        self.input_size = input_size
        self.output_size = output_size
        self.initialization = initialization
        self.weights = self.initialize_weights()

    def initialize_weights(self):
        if self.initialization == "zero":
            return np.zeros((self.input_size, self.output_size))
        elif self.initialization == "random":
            return np.random.randn(self.input_size, self.output_size) * 0.01
        elif self.initialization == "xavier":
            return np.random.randn(self.input_size, self.output_size) / np.sqrt(self.input_size)
        elif self.initialization == "he":
            return np.random.randn(self.input_size, self.output_size) * np.sqrt(2. / self.input_size)
        else:
            raise ValueError(f"Unknown initialization method: {self.initialization}")
```


## Implement Synaptic Delay

Synaptic delay is the time taken for a signal (or spike) to travel from the presynaptic neuron to the postsynaptic neuron across a synapse. This delay is crucial in simulating more realistic neuron interactions and can affect the timing and synchronization of neural activities.

In biological systems, synaptic delays can range from 1ms to tens of milliseconds, depending on various factors like the type of synapse, distance between neurons, and the type of neurotransmitter.

To implement synaptic delay in our model, we can use a queue or buffer mechanism. When a spike is generated by the presynaptic neuron, it's placed in a buffer, and only after the delay duration has passed, the spike is delivered to the postsynaptic neuron.

#### Code:

```python
from collections import deque

class Synapse:
    def __init__(self, delay=5):
        self.delay = delay  # Synaptic delay in ms
        self.spike_buffer = deque(maxlen=delay)

    def transmit_spike(self, spike):
        # Add spike to the buffer
        self.spike_buffer.append(spike)
        
        # If buffer is full, return the oldest spike (FIFO)
        if len(self.spike_buffer) == self.delay:
            return self.spike_buffer.popleft()
        else:
            return None
```


## Implement Plasticity Mechanisms

Neural plasticity refers to the ability of the neural network to change its connections and behavior over time based on experience. One of the most well-known plasticity mechanisms is Spike-Timing-Dependent Plasticity (STDP).

### Spike-Timing-Dependent Plasticity (STDP)

STDP is a biological learning rule based on the relative timing of spikes between the presynaptic and postsynaptic neurons. The basic idea is:
- If the presynaptic neuron fires before the postsynaptic neuron (causal), the synapse is strengthened.
- If the postsynaptic neuron fires before the presynaptic neuron (anti-causal), the synapse is weakened.

#### Code:

```python
class Synapse:
    def __init__(self, weight=0.5, max_weight=1.0, min_weight=0.0, stdp_window=20):
        self.weight = weight
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.stdp_window = stdp_window  # Time window for STDP in ms

    def apply_stdp(self, pre_spike_time, post_spike_time):
        delta_t = post_spike_time - pre_spike_time

        # Causal spike pair
        if delta_t > 0:
            self.weight += self.stdp_function(delta_t)
        # Anti-causal spike pair
        else:
            self.weight -= self.stdp_function(-delta_t)

        # Ensure weights remain within bounds
        self.weight = max(min(self.weight, self.max_weight), self.min_weight)

    def stdp_function(self, delta_t):
        # Example exponential STDP function
        return np.exp(-delta_t / self.stdp_window)
```

In the above code, the `apply_stdp` method adjusts the synaptic weight based on the relative spike timings. The `stdp_function` provides an example exponential decay based on the time difference, but other STDP functions can be used.

(Note: More advanced STDP models can incorporate factors like neuromodulation, frequency dependence, and more.)

### (Optional) Other Biologically Plausible Plasticity Rules

There are many other plasticity rules observed in biological systems, such as homeostatic plasticity, metaplasticity, and more. Implementing these would require a deeper dive into the specific mechanisms and their computational models.


### Spike-Timing-Dependent Plasticity (STDP)

STDP is a form of synaptic plasticity that adjusts the strength of connections between neurons based on the relative timing of their spikes. It's a rule that strengthens or weakens synapses based on the time difference between pre-synaptic and post-synaptic spikes.

#### Principles:

1. **Potentiation**: If a pre-synaptic neuron fires (leading to an action potential) just before a post-synaptic neuron, the synapse is strengthened.
2. **Depression**: If a post-synaptic neuron fires before the pre-synaptic neuron, the synapse is weakened.

#### Code:

```python
import numpy as np

class STDP:
    def __init__(self, A_plus=0.005, A_minus=0.005, tau_plus=20.0, tau_minus=20.0):
        # Parameters for STDP
        self.A_plus = A_plus      # Maximum weight change for potentiation
        self.A_minus = A_minus    # Maximum weight change for depression
        self.tau_plus = tau_plus  # Time constant for potentiation
        self.tau_minus = tau_minus # Time constant for depression

    def compute_weight_change(self, delta_t):
        """Compute weight change based on spike time difference."""
        if delta_t > 0:
            return self.A_plus * np.exp(-delta_t / self.tau_plus)
        else:
            return -self.A_minus * np.exp(delta_t / self.tau_minus)

    def apply_stdp(self, synapse, pre_spike_time, post_spike_time):
        """Apply STDP rule to adjust synaptic weight."""
        delta_t = post_spike_time - pre_spike_time
        weight_change = self.compute_weight_change(delta_t)
        synapse.weight += weight_change
        # Ensure weights are within bounds
        synapse.weight = np.clip(synapse.weight, 0, 1)
```

In the above code, the `STDP` class provides methods to compute the weight change based on the time difference between pre-synaptic and post-synaptic spikes and apply this change to a given synapse. The parameters `A_plus`, `A_minus`, `tau_plus`, and `tau_minus` can be adjusted based on experimental data or specific requirements.


### Rate-Based Plasticity

Rate-Based Plasticity adjusts synaptic weights based on the average firing rate of the pre-synaptic neuron over a certain time window. The idea is that neurons which fire more frequently should exert a stronger influence on their post-synaptic partners.

#### Principles:

1. **Potentiation**: If the average firing rate of the pre-synaptic neuron over a certain time window exceeds a threshold, the synaptic weight is increased.
2. **Depression**: If the average firing rate is below the threshold, the synaptic weight is decreased.

#### Code:

```python
class RateBasedPlasticity:
    def __init__(self, learning_rate=0.01, rate_threshold=10.0):
        # Parameters for Rate-Based Plasticity
        self.learning_rate = learning_rate      # Learning rate for weight adjustments
        self.rate_threshold = rate_threshold    # Firing rate threshold for potentiation/depression

    def compute_weight_change(self, avg_rate):
        """Compute weight change based on average firing rate."""
        if avg_rate > self.rate_threshold:
            return self.learning_rate
        else:
            return -self.learning_rate

    def apply_plasticity(self, synapse, avg_rate):
        """Apply rate-based plasticity rule to adjust synaptic weight."""
        weight_change = self.compute_weight_change(avg_rate)
        synapse.weight += weight_change
        # Ensure weights are within bounds
        synapse.weight = np.clip(synapse.weight, 0, 1)
```

In the above code, the `RateBasedPlasticity` class provides methods to compute the weight change based on the average firing rate of the pre-synaptic neuron and apply this change to a given synapse. The parameters `learning_rate` and `rate_threshold` can be adjusted based on experimental data or specific requirements.






# Construct Layers

### Input Layer

The input layer is the first layer in a neural network and is responsible for receiving and processing the raw input data. In spiking neural networks, the input data is typically encoded into spike trains using various encoding schemes. One of the most common encoding schemes is rate coding.

#### Rate Coding

Rate coding is a method where the frequency of spikes is proportional to the intensity of the input. For example, a brighter pixel in an image or a louder sound would result in a higher spike rate.

##### Code:

```python
def rate_coding(input_data, max_rate=100):
    """Convert input data to spike rates."""
    normalized_data = (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data))
    spike_rates = max_rate * normalized_data
    return spike_rates
```

#### (Optional) Other Encoding Schemes

There are various other encoding schemes like time-to-first-spike, phase coding, etc. These can be explored based on the specific requirements of the project.

---

### Full Code for Input Layer:

```python
import numpy as np

class InputLayer:
    def __init__(self, size, max_rate=100):
        self.size = size
        self.max_rate = max_rate

    def encode(self, input_data):
        """Encode input data into spike rates."""
        normalized_data = (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data))
        spike_rates = self.max_rate * normalized_data
        return spike_rates
```

### Implement Rate Coding to Encode Sensory Data into Spikes

Rate coding is a method where the frequency of spikes is proportional to the intensity of the input. This encoding scheme is commonly used in spiking neural networks to transform continuous input data (like pixel intensities) into discrete spike trains.

#### Concept:

In rate coding, the value of the input data determines the frequency of the spikes. For instance, higher values in the input data would result in a higher frequency of spikes, and vice versa. This method allows for a simple yet effective way to represent continuous values in the form of spikes.

#### Code Snippet:

```python
def rate_coding(input_data, max_rate=100):
    """Convert input data to spike rates."""
    normalized_data = (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data))
    spike_rates = max_rate * normalized_data
    return spike_rates
```

This function takes in the `input_data` and a `max_rate` which denotes the maximum possible spike rate. The data is first normalized to a range between 0 and 1, and then multiplied by the `max_rate` to get the spike rates.

---

### Full Code for Rate Coding:

```python
import numpy as np

class InputLayer:
    def __init__(self, size, max_rate=100):
        self.size = size
        self.max_rate = max_rate

    def encode(self, input_data):
        """Encode input data into spike rates using rate coding."""
        normalized_data = (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data))
        spike_rates = self.max_rate * normalized_data
        return spike_rates
```

In the `InputLayer` class, the `encode` method implements the rate coding scheme. The method normalizes the input data and then multiplies it by the maximum spike rate to get the spike rates for each data point.


---

### Construct Layers with Biologically-Inspired Neurons

The hidden layers in our neural network will consist of biologically-inspired neurons. These neurons will have properties like membrane potential, firing threshold, refractory period, and potentially ion channel dynamics. They will be interconnected through synapses, which will have their own properties like synaptic weights and delays.

#### Concept:

Hidden layers are crucial in neural networks as they allow the network to learn and represent complex patterns and relationships in the input data. In a biologically-inspired neural network, these layers will consist of spiking neurons that communicate through spikes.

#### Code Snippet:

```python
class HiddenLayer:
    def __init__(self, size):
        self.neurons = [BiologicalNeuron() for _ in range(size)]

    def forward(self, inputs):
        spikes = [neuron.process_input(input) for neuron, input in zip(self.neurons, inputs)]
        return spikes
```

In the `HiddenLayer` class, we initialize a list of `BiologicalNeuron` instances. The `forward` method processes the inputs through each neuron and returns the spikes generated by them.

---

### Full Code for Hidden Layer:

```python
import numpy as np

class BiologicalNeuron:
    def __init__(self):
        self.membrane_potential = 0
        self.firing_threshold = -55
        self.refractory_period = 2
        self.refractory_timer = 0

    def process_input(self, input):
        if self.refractory_timer > 0:
            self.refractory_timer -= 1
            return 0

        self.membrane_potential += input
        if self.membrane_potential >= self.firing_threshold:
            self.membrane_potential = 0
            self.refractory_timer = self.refractory_period
            return 1
        return 0

class HiddenLayer:
    def __init__(self, size):
        self.neurons = [BiologicalNeuron() for _ in range(size)]

    def forward(self, inputs):
        spikes = [neuron.process_input(input) for neuron, input in zip(self.neurons, inputs)]
        return spikes
```

In the above code, we have the `BiologicalNeuron` class which simulates the behavior of a biologically-inspired neuron. The `HiddenLayer` class contains multiple instances of these neurons and processes inputs through them.

---

### Layer-wise Properties Inspired by the Human Brain

#### Conceptual Overview:

The human brain is organized into different regions, each with its unique properties and functions. When constructing artificial neural networks inspired by the human brain, it's essential to consider these properties to make the model more biologically plausible. Some of these properties include:

1. **Layer Density**: Different regions of the brain have varying densities of neurons. For instance, the primary visual cortex has a higher density of neurons compared to other areas.
2. **Neuron Types**: The brain consists of various neuron types, each with its characteristics and functions. Examples include excitatory pyramidal cells and inhibitory interneurons.
3. **Connectivity Patterns**: Neurons in the brain have specific connectivity patterns. For instance, in the cortex, there's a columnar organization where neurons in a column process similar types of information.
4. **Synaptic Plasticity**: Different regions of the brain exhibit varying degrees and types of synaptic plasticity, which is the ability of synapses to strengthen or weaken over time.

#### Code Snippet:

```python
class LayerProperties:
    def __init__(self, neuron_density, neuron_type, connectivity_pattern, synaptic_plasticity):
        self.neuron_density = neuron_density
        self.neuron_type = neuron_type
        self.connectivity_pattern = connectivity_pattern
        self.synaptic_plasticity = synaptic_plasticity
```

This class represents the properties of a neural layer inspired by the human brain. Each layer can have its density of neurons, neuron types, connectivity patterns, and synaptic plasticity mechanisms.

#### Full Code:

```python
class LayerProperties:
    def __init__(self, neuron_density=1000, neuron_type="pyramidal", connectivity_pattern="columnar", synaptic_plasticity="STDP"):
        self.neuron_density = neuron_density  # Number of neurons per unit area
        self.neuron_type = neuron_type  # Type of neurons in the layer
        self.connectivity_pattern = connectivity_pattern  # Connectivity pattern of the neurons
        self.synaptic_plasticity = synaptic_plasticity  # Type of synaptic plasticity mechanism

    def describe(self):
        description = f"Layer with {self.neuron_density} neurons per unit area of type {self.neuron_type}. "
        description += f"It follows a {self.connectivity_pattern} connectivity pattern with {self.synaptic_plasticity} as its synaptic plasticity mechanism."
        return description
```
