#pragma once
#include "NeuronId.h"

class Neuron {
public:
    NeuronId id;
    
    double activity = 0.0;
    double confidence = 0.0;
    double energy = 1.0;
    bool fired_last_cycle = false;

    // Bleibt ein normaler double, der Schutz erfolgt extern.
    double total_input = 0.0;

    Neuron(const NeuronId& id);

    bool should_fire(double global_threshold) const;
};