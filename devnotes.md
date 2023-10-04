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




## Output Layer

The output layer of our spiking neural network is responsible for decoding the spiking activity of the network into task-specific outputs. This can be a classification label, a continuous value, or any other type of output depending on the application.

### Decoding Mechanisms

Decoding the spiking activity is a crucial step in our network. The spikes generated by the neurons in the network need to be translated into a format that can be understood and used for the specific task at hand. There are various methods to decode spiking activity, and the choice of method can depend on the nature of the task and the architecture of the network.

#### Rate-based Decoding

One of the simplest methods to decode spiking activity is rate-based decoding. In this method, the firing rate of a neuron (or a group of neurons) over a specific time window is used as the output. The firing rate can be calculated by counting the number of spikes in the time window and dividing by the length of the window.

```python
def rate_based_decoding(spikes, time_window):
    """
    Decode spiking activity based on firing rate.

    Parameters:
    - spikes: A list where each entry represents a spike time.
    - time_window: Duration of the time window over which to calculate the firing rate.

    Returns:
    - firing_rate: The firing rate of the neuron over the time window.
    """
    num_spikes = len([spike for spike in spikes if spike <= time_window])
    firing_rate = num_spikes / time_window
    return firing_rate
```

#### Time-to-first-spike Decoding

Another method to decode spiking activity is time-to-first-spike decoding. In this method, the time at which the first spike occurs is used as the output. This method can be particularly useful when the precise timing of spikes is important.

```python
def time_to_first_spike_decoding(spikes):
    """
    Decode spiking activity based on the time of the first spike.

    Parameters:
    - spikes: A list where each entry represents a spike time.

    Returns:
    - first_spike_time: The time of the first spike or None if there are no spikes.
    """
    if spikes:
        return spikes[0]
    else:
        return None
```

#### Placeholder for a more complex decoding mechanism based on spike timings

In the future, we can explore more complex decoding mechanisms that take into account the precise timings of spikes, the relative timings of spikes from different neurons, and other factors. Such methods can potentially provide more accurate and nuanced outputs, especially for tasks where the temporal dynamics of the network are important.

### Full Code for Output Layer

Combining the above snippets, here's the full code for the output layer:

```python
class OutputLayer:

    def __init__(self, decoding_method="rate_based"):
        self.decoding_method = decoding_method

    def decode(self, spikes, time_window=None):
        if self.decoding_method == "rate_based":
            return self.rate_based_decoding(spikes, time_window)
        elif self.decoding_method == "time_to_first_spike":
            return self.time_to_first_spike_decoding(spikes)
        else:
            raise ValueError(f"Unknown decoding method: {self.decoding_method}")

    @staticmethod
    def rate_based_decoding(spikes, time_window):
        num_spikes = len([spike for spike in spikes if spike <= time_window])
        firing_rate = num_spikes / time_window
        return firing_rate

    @staticmethod
    def time_to_first_spike_decoding(spikes):
        if spikes:
            return spikes[0]
        else:
            return None
```
This code provides a basic structure for the output layer, with two decoding methods implemented. As the project progresses, more decoding methods and functionalities can be added to this layer.

---

# Connectivity

### Local Connectivity

Biological neural networks exhibit a property where not every neuron is connected to every other neuron. This is in stark contrast to many artificial neural network architectures where layers are fully connected. In biological systems, neurons often have what's termed as "local receptive fields", meaning they are only connected to a small, localized group of neurons in the previous layer. This phenomenon is especially pronounced in sensory systems, such as the visual system.

#### Advantages of Local Connectivity:

1. **Computational Efficiency**: By limiting the number of connections, the network can process information more efficiently.
2. **Feature Specialization**: Neurons can become specialized in detecting specific features in the input data. For instance, in the visual system, certain neurons might become edge detectors, while others might specialize in detecting textures.
3. **Biological Plausibility**: Emulating this property makes our artificial network more closely resemble its biological counterparts.

#### Implementing Local Connectivity in Code:

To simulate local connectivity, we can define a receptive field size for each neuron. When constructing our network, each neuron will only form connections with a subset of neurons from the previous layer, based on this receptive field.

```python
class Neuron:
    def __init__(self, neuron_id):
        self.id = neuron_id
        self.connections = []

    def connect(self, other_neuron):
        self.connections.append(other_neuron)

class Layer:
    def __init__(self, num_neurons, receptive_field_size=None):
        self.neurons = [Neuron(i) for i in range(num_neurons)]
        self.receptive_field_size = receptive_field_size

    def connect_to(self, previous_layer):
        for neuron in self.neurons:
            # Determine the range of neurons in the previous layer to connect to
            start_idx = max(0, neuron.id - self.receptive_field_size // 2)
            end_idx = min(len(previous_layer.neurons), neuron.id + self.receptive_field_size // 2 + 1)
            
            # Connect the neuron to the determined range of neurons from the previous layer
            for prev_neuron in previous_layer.neurons[start_idx:end_idx]:
                neuron.connect(prev_neuron)
```

This code establishes a basic framework for creating layers of neurons with local connectivity. The `Layer` class contains a method `connect_to` that establishes connections between its neurons and a subset of neurons from a previous layer, based on the defined receptive field size.



## Connectivity
### Local Receptive Fields for the Input Layer

In biological systems, especially in the visual cortex, neurons don't respond to every part of the visual field but to specific regions known as receptive fields. This concept can be applied to artificial neural networks, especially convolutional neural networks (CNNs), where filters act as receptive fields, scanning the input data (like an image) in local patches.

#### Advantages of Local Receptive Fields:

1. **Parameter Efficiency**: Reduces the number of parameters, as each neuron doesn't connect to every input but only a localized region.
2. **Feature Detection**: Allows the network to detect local features, which can be combined in subsequent layers to detect more complex patterns.
3. **Spatial Hierarchy**: By having multiple layers of neurons with local receptive fields, the network can build a spatial hierarchy of features.

#### Implementing Local Receptive Fields in Code:

In traditional artificial neural networks, this can be achieved using convolutional layers. Here's a basic example using Python's TensorFlow:

```python
import tensorflow as tf

# Assuming input_data is your input layer (e.g., an image)
input_data = tf.keras.layers.Input(shape=(28, 28, 1))

# Implementing a local receptive field using a convolutional layer
conv_layer = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_data)
```

In the above code, the `Conv2D` layer scans the `input_data` using a 3x3 filter (our local receptive field) and produces 32 feature maps. Each unit in the `conv_layer` has a receptive field of 3x3 units in the `input_data`.

#### Comprehensive Real-World Complexity Version:

In a real-world scenario, a neural network might have multiple such convolutional layers, pooling layers to downsample the spatial dimensions, and more:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

This model first uses a convolutional layer with 32 filters and a 3x3 receptive field, followed by a max-pooling layer to downsample the feature maps. Another convolutional layer with 64 filters is applied, followed by another max-pooling layer. The resulting feature maps are flattened and passed through dense layers for classification.

### Recurrent Connections

Recurrent connections are a fundamental aspect of neural networks, especially in the context of processing sequences and temporal data. These connections loop back from a neuron to itself or to its preceding neurons, allowing the network to maintain a form of 'memory' of previous inputs.

#### Advantages of Recurrent Connections:

1. **Temporal Dynamics**: They allow networks to process sequences, such as time series data or sentences, by maintaining information about previous inputs.
2. **Internal Memory**: Recurrent connections give the network an internal state that can hold information over time.
3. **Complex Pattern Recognition**: They can recognize patterns in sequences and can even generate sequences.

#### Implementing Recurrent Connections in Code:

In modern deep learning frameworks, recurrent connections are typically implemented using specialized layers like RNN, LSTM, or GRU. Here's a basic example using Python's TensorFlow:

```python
import tensorflow as tf

# Assuming input_data is a sequence (e.g., a sentence or time series)
input_data = tf.keras.layers.Input(shape=(None, feature_size))

# Implementing a recurrent connection using an LSTM layer
lstm_layer = tf.keras.layers.LSTM(units=128, return_sequences=True)(input_data)
```

In the above code, the `LSTM` layer processes the `input_data` sequence and maintains its internal state across the sequence, thanks to its recurrent connections.

#### Comprehensive Real-World Complexity Version:

In a real-world scenario, a neural network might have multiple stacked recurrent layers, dropout for regularization, and more:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(None, feature_size)),
    tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    tf.keras.layers.Dense(output_size, activation='softmax')
])
```

This model first uses an LSTM layer with 128 units, followed by another LSTM layer with 64 units. Dropout is added for regularization. The resulting sequence is then passed through a dense layer for classification or regression tasks.


### Feedback Connections Within and Between Layers

Feedback connections, also known as top-down connections, are essential for many neural network architectures, especially those that aim to replicate the hierarchical and recurrent nature of biological neural networks. These connections send information from higher layers back to lower layers, allowing for the integration of high-level context into low-level processing.

#### Advantages of Feedback Connections:

1. **Contextual Integration**: They allow for the integration of broader contextual information into the processing of specific details.
2. **Error Correction**: Feedback can be used to correct predictions or refine representations based on higher-level knowledge.
3. **Stability in Learning**: Feedback connections can stabilize learning by providing a consistent context.

#### Implementing Feedback Connections in Code:

In many deep learning frameworks, feedback connections can be implemented using skip connections or recurrent layers. Here's a basic example using Python's TensorFlow:

```python
import tensorflow as tf

input_data = tf.keras.layers.Input(shape=(input_shape))

# Forward pass
x1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_data)
x2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x1)

# Feedback connection from x2 to x1
feedback = tf.keras.layers.Conv2DTranspose(32, (3, 3), activation='relu')(x2)
merged = tf.keras.layers.Add()([x1, feedback])
```

In the above code, the `Conv2DTranspose` layer is used to create a feedback connection from the `x2` layer back to the `x1` layer.

#### Comprehensive Real-World Complexity Version:

In a more complex scenario, feedback connections can be integrated into deeper architectures, and combined with other mechanisms like batch normalization and pooling:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_shape)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2DTranspose(32, (3, 3), activation='relu'),
    tf.keras.layers.Add(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(output_size, activation='softmax')
])
```

This model integrates feedback connections within a deeper architecture, using batch normalization for stability and pooling for down-sampling.



### Lateral Connections

Lateral connections, also known as lateral inhibitions, play a crucial role in neural networks, especially in the context of feature competition and normalization. These connections are primarily inhibitory and allow neurons to inhibit their neighbors, ensuring that only the most prominent features are processed.

#### Advantages of Lateral Connections:

1. **Feature Competition**: Lateral connections allow neurons to compete, ensuring that only the most significant features are processed.
2. **Normalization**: They help in normalizing the activity across a layer.
3. **Noise Reduction**: By inhibiting less significant activations, lateral connections can help reduce noise.

#### Implementing Lateral Connections in Code:

In many deep learning frameworks, lateral connections can be implemented using custom layers or operations. Here's a basic example using Python's TensorFlow:

```python
import tensorflow as tf

def lateral_inhibition(inputs):
    # This is a simplified example. In a real-world scenario, 
    # more complex operations might be needed.
    max_val = tf.reduce_max(inputs, axis=-1, keepdims=True)
    inhibited = tf.where(inputs < max_val, 0, inputs)
    return inhibited

input_data = tf.keras.layers.Input(shape=(input_shape))
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_data)
x = tf.keras.layers.Lambda(lateral_inhibition)(x)
```

In the above code, the `lateral_inhibition` function inhibits all activations except the maximum one in the feature maps.

#### Comprehensive Real-World Complexity Version:

In a more complex scenario, lateral connections can be integrated into deeper architectures and combined with other mechanisms:

```python
def complex_lateral_inhibition(inputs):
    # This function would contain a more complex lateral inhibition mechanism,
    # possibly considering spatial neighborhoods, different inhibition strengths, etc.
    pass

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_shape)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.Lambda(complex_lateral_inhibition),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Lambda(complex_lateral_inhibition),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(output_size, activation='softmax')
])
```

This model integrates lateral connections within a deeper architecture, using batch normalization for stability and pooling for down-sampling.


### Lateral Inhibition within Layers

Lateral inhibition is a mechanism in neural networks where a neuron's activity inhibits the activity of its neighbors. This process enhances the contrast between active and less active neurons, making the patterns of activity more distinct.

#### Advantages of Lateral Inhibition:

1. **Contrast Enhancement**: Helps in emphasizing the differences between neighboring neurons.
2. **Noise Reduction**: Reduces the chance of multiple neurons firing for similar stimuli.
3. **Sparse Coding**: Encourages sparsity in neural activations, which can be beneficial for memory and computational efficiency.

#### Basic Implementation of Lateral Inhibition:

Here's a basic example using Python's TensorFlow:

```python
import tensorflow as tf

def lateral_inhibition(inputs):
    # This is a simplified example. In a real-world scenario, 
    # more complex operations might be needed.
    max_val = tf.reduce_max(inputs, axis=-1, keepdims=True)
    inhibited = tf.where(inputs < max_val, 0, inputs)
    return inhibited

input_data = tf.keras.layers.Input(shape=(input_shape))
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(input_data)
x = tf.keras.layers.Lambda(lateral_inhibition)(x)
```

#### Comprehensive Real-World Complexity Version:

In a more complex scenario, lateral inhibition can be implemented considering spatial neighborhoods and different inhibition strengths:

```python
def complex_lateral_inhibition(inputs, inhibition_radius=3):
    # Assuming inputs are of shape (batch_size, height, width, channels)
    # This function would apply lateral inhibition within a spatial neighborhood defined by inhibition_radius
    inhibited = tf.nn.max_pool(inputs, ksize=[1, inhibition_radius, inhibition_radius, 1], 
                               strides=[1, 1, 1, 1], padding='SAME')
    return inhibited

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_shape)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.Lambda(lambda x: complex_lateral_inhibition(x, inhibition_radius=3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Lambda(lambda x: complex_lateral_inhibition(x, inhibition_radius=5)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(output_size, activation='softmax')
])
```

This model integrates lateral inhibition considering spatial neighborhoods within a deeper architecture, using batch normalization for stability and pooling for down-sampling.




# Learning Mechanisms

### Hebbian Learning

Hebbian Learning is a learning principle based on the idea that "neurons that fire together, wire together." It's a form of unsupervised learning where synaptic weights are adjusted based on the correlation between pre-synaptic and post-synaptic activity.

#### Basic Principles:

1. **Strengthening Synapses**: If a neuron A consistently takes part in firing neuron B, then the synapse from A to B is strengthened.
2. **Weakening Synapses**: If neuron A's firing is not correlated with neuron B's firing, the synapse might be weakened or remain unchanged.

#### Basic Implementation:

Here's a basic example using Python:

```python
def hebbian_update(pre_synaptic_activity, post_synaptic_activity, weights, learning_rate=0.01):
    delta_w = learning_rate * pre_synaptic_activity * (post_synaptic_activity - weights * pre_synaptic_activity)
    new_weights = weights + delta_w
    return new_weights
```

#### Comprehensive Real-World Complexity Version:

In a more complex scenario, Hebbian learning can be combined with other learning rules, decay mechanisms, and normalization:

```python
def complex_hebbian_update(pre_synaptic_activity, post_synaptic_activity, weights, learning_rate=0.01, decay_rate=0.001):
    # Hebbian learning rule
    delta_w = learning_rate * pre_synaptic_activity * (post_synaptic_activity - weights * pre_synaptic_activity)
    
    # Weight decay
    decay = decay_rate * weights
    
    # Update weights
    new_weights = weights + delta_w - decay
    
    # Normalize weights to prevent them from growing too large
    new_weights = new_weights / tf.norm(new_weights, axis=1, keepdims=True)
    
    return new_weights
```

This function integrates Hebbian learning with weight decay and normalization to ensure stability and prevent weights from growing indefinitely.


## Spike-Timing-Dependent Plasticity (STDP)

STDP is a biological learning rule based on the relative timing of spikes between pre-synaptic and post-synaptic neurons. The basic idea is that the synaptic weight is adjusted based on the time difference between the firing of the pre-synaptic and post-synaptic neurons.

#### Basic Principles:

1. **Potentiation**: If the pre-synaptic neuron fires just before the post-synaptic neuron (within a certain time window), the synapse is strengthened.
2. **Depression**: If the post-synaptic neuron fires before the pre-synaptic neuron, the synapse is weakened.

#### Basic Implementation:

Here's a basic example using Python:

```python
def stdp_update(pre_spike_time, post_spike_time, weight, a_plus=0.005, a_minus=0.005, tau_plus=20.0, tau_minus=20.0):
    delta_t = post_spike_time - pre_spike_time
    if delta_t > 0:
        # Potentiation
        delta_w = a_plus * np.exp(-delta_t / tau_plus)
    else:
        # Depression
        delta_w = -a_minus * np.exp(delta_t / tau_minus)
    new_weight = weight + delta_w
    return new_weight
```

#### Comprehensive Real-World Complexity Version:

In a more complex scenario, STDP can be combined with other mechanisms like weight normalization, weight limits, and more:

```python
def complex_stdp_update(pre_spike_times, post_spike_times, weights, a_plus=0.005, a_minus=0.005, tau_plus=20.0, tau_minus=20.0, max_weight=1.0, min_weight=0.0):
    for pre_time in pre_spike_times:
        for post_time in post_spike_times:
            delta_t = post_time - pre_time
            if delta_t > 0:
                # Potentiation
                delta_w = a_plus * np.exp(-delta_t / tau_plus)
            else:
                # Depression
                delta_w = -a_minus * np.exp(delta_t / tau_minus)
            weights += delta_w
    # Ensure weights stay within limits
    weights = np.clip(weights, min_weight, max_weight)
    # Normalize weights
    weights = weights / np.sum(weights)
    return weights
```

This function integrates STDP with weight normalization and weight limits to ensure stability and prevent weights from growing or shrinking indefinitely.


## Backpropagation in Spiking Neural Networks

Backpropagation is a supervised learning algorithm used for training feedforward artificial neural networks. However, its direct application to spiking neural networks (SNNs) is non-trivial due to the non-differentiable nature of spikes. Several methods have been proposed to adapt backpropagation for SNNs, one of which is the surrogate gradient method.

#### Basic Principles:

1. **Surrogate Gradient**: Since the spike function is non-differentiable, we use a differentiable surrogate function to approximate the gradient during backpropagation.
2. **Error Calculation**: The difference between the desired output and the actual output is calculated.
3. **Weight Update**: Weights are adjusted in the direction that minimizes the error.

#### Basic Implementation:

Here's a basic example using Python:

```python
import numpy as np

def surrogate_gradient(x):
    """Surrogate gradient for the spike function."""
    return 0.3 * np.exp(-x**2 / 2)

def backprop_update(weights, learning_rate, error, activations):
    """Basic backpropagation weight update."""
    dW = np.outer(error, activations)
    weights -= learning_rate * dW
    return weights
```

#### Comprehensive Real-World Complexity Version:

In a more complex scenario, backpropagation in SNNs would involve multiple layers, time steps, and additional considerations for the unique properties of spiking neurons:

```python
class SpikingNN:
    def __init__(self, input
```



## Loss Functions for Spiking Neural Networks

Loss functions quantify the difference between the predicted output and the actual target. In traditional neural networks, common loss functions include Mean Squared Error (MSE) and Cross-Entropy. However, for SNNs, the non-continuous nature of spikes introduces challenges. Therefore, we need to adapt or define new loss functions suitable for the spiking domain.

#### Basic Principles:

1. **Rate-based Loss**: This approach involves converting spike trains into firing rates and then applying traditional loss functions.
2. **Spike Timing-based Loss**: This focuses on the timing of spikes, penalizing deviations in spike times between the predicted and target spike trains.

#### Basic Implementation:

Here's a basic example using Python:

```python
import numpy as np

def rate_based_mse_loss(predicted_spikes, target_spikes, T):
    """Rate-based Mean Squared Error Loss."""
    predicted_rate = np.sum(predicted_spikes) / T
    target_rate = np.sum(target_spikes) / T
    return (predicted_rate - target_rate) ** 2

def spike_timing_loss(predicted_spikes, target_spikes):
    """Spike Timing-based Loss."""
    # Find the spike times
    predicted_times = np.where(predicted_spikes)[0]
    target_times = np.where(target_spikes)[0]
    # Calculate the timing difference
    timing_difference = np.abs(predicted_times - target_times)
    return np.sum(timing_difference)
```

#### Comprehensive Real-World Complexity Version:

In a more complex scenario, we might consider a combination of rate-based and timing-based losses, and possibly introduce regularization terms:

```python
class SpikingLoss:
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.01):
        self.alpha = alpha  # Weight for rate-based loss
        self.beta = beta    # Weight for timing-based loss
        self.gamma = gamma  # Regularization term

    def combined_loss(self, predicted_spikes, target_spikes, T):
        rate_loss = rate_based_mse_loss(predicted_spikes, target_spikes, T)
        timing_loss = spike_timing_loss(predicted_spikes, target_spikes)
        regularization = self.gamma * np.sum(predicted_spikes)
        return self.alpha * rate_loss + self.beta * timing_loss + regularization
```

This combined loss function allows for balancing between rate-based and timing-based objectives, while also introducing a regularization term to penalize excessive spiking, which is a common concern in SNNs.




## Learning Mechanisms

### Gradient Descent and Weight Updates for Spiking Neural Networks

Gradient descent is an optimization algorithm used to minimize the loss function by adjusting the weights in the network. In the context of SNNs, gradient descent can be more challenging due to the non-differentiable nature of spikes. However, approximations and surrogate gradients can be used to make the learning rule differentiable.

#### Basic Principles:

1. **Surrogate Gradient**: An approximation to the real gradient, making it possible to use gradient-based methods.
2. **Weight Update Rule**: The weights are adjusted based on the gradient of the loss with respect to the weights.

#### Basic Implementation:

Here's a basic example using Python:

```python
import numpy as np

def surrogate_gradient(x):
    """Surrogate gradient for the non-differentiable spiking function."""
    return 0.3 * np.exp(-x**2 / (2 * 0.3**2))

def gradient_descent(weights, gradients, learning_rate=0.01):
    """Basic gradient descent update."""
    return weights - learning_rate * gradients

def compute_gradients(predicted_spikes, target_spikes, weights, loss_function):
    """Compute the gradient of the loss with respect to the weights."""
    loss = loss_function(predicted_spikes, target_spikes)
    dloss_doutput = (predicted_spikes - target_spikes)
    doutput_dinput = surrogate_gradient(weights)
    return dloss_doutput * doutput_dinput
```

#### Comprehensive Real-World Complexity Version:

In a more complex scenario, we might consider momentum and other optimization techniques:

```python
class GradientDescentOptimizer:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.previous_gradient = 0

    def update_weights(self, weights, gradients):
        """Gradient descent with momentum."""
        update = self.momentum * self.previous_gradient + self.learning_rate * gradients
        self.previous_gradient = update
        return weights - update
```

This optimizer introduces momentum, which can help accelerate gradients in the right directions and dampen oscillations. It's
