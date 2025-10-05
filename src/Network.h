#pragma once
#include <vector>
#include <mutex> 
#include "Neuron.h"
#include "Synapse.h"
#include "GlobalHomeostasis.h"

class Network {
public:
    Network();

    void add_neuron(const  NeuronId& id);
    void add_synapse(int source_idx, int target_idx);

    // Setzt den Zustand eines Neurons von außen (z.B. für Sensor-Input)
    void set_neuron_state(int neuron_idx, double activity, double confidence);
    
    // Führt einen einzelnen Simulationsschritt durch
    HomeostasisData network_cycle_step();

    // Wendet eine Belohnung an, um das Lernen zu steuern
    void apply_reward(double reward);

    // Hilfsfunktionen
    size_t  get_total_neurons() const;
    Neuron get_neuron_copy(int idx) const;
    void print_network_state() const;
    void print_synapse_trust() const;

private:
    std::vector<Neuron> neurons;
    std::vector<Synapse> synapses;
    GlobalHomeostasis homeostasis;

    // Mutex zum Schutz aller geteilten Daten bei parallelem Zugriff
    mutable std::mutex network_mutex;

    // Lernparameter (Platzhalter für ΔV und β_effektiv)
    double learning_rate = 0.1;  // ΔV
    double trust_decay = 0.02;   // β_effektiv
};