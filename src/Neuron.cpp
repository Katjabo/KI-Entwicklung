#include "Neuron.h"

Neuron::Neuron(const NeuronId& id) : id(id) {}

bool Neuron::should_fire(double global_threshold) const {
    return this->total_input > global_threshold;
}