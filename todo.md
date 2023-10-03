# NeuraLearn TODO List

This document outlines the development tasks required to build the NeuraLearn project, a biologically-inspired neural network framework.

## Table of Contents

- [Neuron Model](#neuron-model)
- [Synapses](#synapses)
- [Construct Layers](#construct-layers)
- [Connectivity](#connectivity)
- [Learning Mechanisms](#learning-mechanisms)
- [Training Infrastructure](#training-infrastructure)
- [Utilities](#utilities)

## Neuron Model

- [✅] Define the basic properties of a neuron:
  - [✅] Membrane potential
  - [✅] Firing threshold
  - [✅] Refractory period
  - [✅] (Optional) Ion channels and dynamics
- [✅] Implement a method to receive inputs (spikes from other neurons).
- [✅] Implement a method to process inputs and determine if the neuron should fire.
- [✅] Implement a method to reset the neuron after firing.

## Synapses

- [✅] Define synaptic weights and their initializations.
- [✅] Implement synaptic delay.
- [✅] Implement plasticity mechanisms:
  - [✅] Spike-Timing-Dependent Plasticity (STDP)
  - [✅] (Optional) Other biologically plausible plasticity rules.

## Construct Layers

- [✅] **Input Layer**:
  - [✅] Implement rate coding to encode sensory data into spikes.
  - [❌] (Optional) Explore other encoding schemes.
- [✅] **Hidden Layers**:
  - [✅] Construct layers with biologically-inspired neurons.
  - [✅] Define layer-wise properties, if any (stay true to the human brain)
- [✅] **Output Layer**:
  - [✅] Implement decoding mechanisms to convert spiking activity into task-specific outputs.

## Connectivity

- [✅] **Local Connectivity**:
  - [✅] Implement local receptive fields for the input layer.
- [❌] **Recurrent Connections**:
  - [❌] Allow for feedback connections within and between layers.
- [❌] **Lateral Connections**:
  - [❌] Implement lateral inhibition within layers.

## Learning Mechanisms

- [❌] Implement Hebbian Learning.
- [❌] Implement STDP.
- [❌] Implement backpropagation:
  - [❌] Define loss functions.
  - [❌] Implement gradient descent and weight updates.

## Training Infrastructure

- [❌] Implement a training loop:
  - [❌] Data presentation
  - [❌] Weight adjustment based on performance
  - [❌] Evaluation on test data
- [❌] Implement model saving and loading mechanisms.
- [❌] Implement early stopping, checkpoints, and other training utilities.

## Utilities

- [❌] Implement data preprocessing functions.
- [❌] Implement performance metrics (e.g., accuracy, loss).
- [❌] Implement logging utilities.
- [❌] Implement visualization tools for network activity and performance.
