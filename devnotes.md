## Neuron Model

### 1. Membrane Potential

The **membrane potential** (`V_m`) is the voltage difference across the neuron's cell membrane. It fluctuates based on the inputs the neuron receives and determines whether the neuron will fire an action potential.

#### Code:
```python
import numpy as np

class Neuron:
    def __init__(self):
        self.V_m = -70  # Initial membrane potential in mV (resting potential)
        self.resting_potential = -70  # Resting potential in mV
        self.capacitance = 1.0  # Capacitance of the neuronal membrane
```

### 2. Firing Threshold

The **firing threshold** is the membrane potential value at which the neuron fires an action potential. When `V_m` reaches this value, the neuron produces a spike.

#### Code:
```python
    def __init__(self):
        self.V_m = -70  # Membrane potential in mV
        self.threshold = -55  # Firing threshold in mV
```

### 3. Refractory Period

After firing, there's a period during which the neuron cannot fire again, ensuring signal clarity and directionality.

#### Code:
```python
    def __init__(self):
        self.V_m = -70  # Membrane potential in mV
        self.threshold = -55  # Firing threshold in mV
        self.refractory_period = 2  # Refractory period in ms
        self.last_spike_time = None  # Time of the last spike
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
