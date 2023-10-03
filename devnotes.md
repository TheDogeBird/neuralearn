## Neuron Model

### 1. Membrane Potential

The **membrane potential** (`V_m`) represents the voltage difference across the neuron's cell membrane. It's the primary variable that determines whether a neuron will fire or not.

#### Code:
```python
import numpy as np

class Neuron:
    def __init__(self):
        self.V_m = -70  # Initial membrane potential in mV (resting potential)
```

### 2. Firing Threshold

The **firing threshold** is the value of the membrane potential at which the neuron fires an action potential (or spike). When the membrane potential reaches this value, the neuron produces a spike and sends a signal down its axon.

#### Code:
```python
class Neuron:
    def __init__(self):
        self.V_m = -70  # Membrane potential in mV
        self.threshold = -55  # Firing threshold in mV
```

### 3. Refractory Period

After a neuron fires an action potential, there's a period during which it's either impossible or difficult for the neuron to fire again. This is known as the **refractory period**.

#### Code:
```python
class Neuron:
    def __init__(self):
        self.V_m = -70  # Membrane potential in mV
        self.threshold = -55  # Firing threshold in mV
        self.refractory_period = 2  # Refractory period in ms
```
