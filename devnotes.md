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


