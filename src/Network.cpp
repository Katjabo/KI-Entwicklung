#include "Network.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>

Network::Network() {}

void Network::add_neuron(const NeuronId& id) {
    std::lock_guard<std::mutex> lock(network_mutex);
    neurons.emplace_back(id);
}

void Network::add_synapse(int source_idx, int target_idx) {
    std::lock_guard<std::mutex> lock(network_mutex);
    synapses.emplace_back(source_idx, target_idx);
}

void Network::set_neuron_state(int neuron_idx, double activity, double confidence) {
    std::lock_guard<std::mutex> lock(network_mutex);
    if (neuron_idx < static_cast<int>(neurons.size())) {
        neurons[neuron_idx].activity = activity;
        neurons[neuron_idx].confidence = confidence;
    }
}

HomeostasisData Network::network_cycle_step() {
    std::lock_guard<std::mutex> lock(network_mutex);

    // Konzept: Globale Homöostase (Kapitel 2)
    // 1. Homöostase-Update basierend auf der Aktivität des letzten Zyklus
    int active_neurons_count = 0;
    for (const auto& neuron : neurons) {
        if (neuron.fired_last_cycle) {
            active_neurons_count++;
        }
    }
    HomeostasisData data = homeostasis.update_homeostasis(active_neurons_count, static_cast<int>(neurons.size()));
    double current_theta = homeostasis.get_current_threshold();

    for (auto& neuron : neurons) {
        neuron.total_input = 0.0;
        neuron.fired_last_cycle = false;
    }
     for (auto& synapse : synapses) {
        synapse.eligibility_trace *= 0.5;
    }

    // Konzept: Konfidenz-gewichtete Signalweitergabe (Kapitel 2b.3.1)
    // 2. Signalweitergabe
    for (const auto& synapse : synapses) {
        const auto& source_neuron = neurons[synapse.source_neuron_idx];
        
        // Formel: A_ij = Ai * Ci
        double effective_signal = source_neuron.activity * source_neuron.confidence;
        
        // Das Signal wird zusätzlich mit dem synaptischen Vertrauen (V_ij) gewichtet
        neurons[synapse.target_neuron_idx].total_input += effective_signal * synapse.trust_value;
    }

    // 3. Neuronale Verarbeitung (Feuern basierend auf Theta)
    for (auto& neuron : neurons) {
        // Konzept: Herleitung der Konfidenz für abstrakte Neuronen (Kapitel 2b.2)
        // Die Konfidenz wird nur für Neuronen aktualisiert, die auch Input erhalten.
        if (neuron.total_input > 0) {
            neuron.confidence = std::min(1.0, neuron.total_input);
        }
        if (neuron.should_fire(current_theta)) {
            neuron.fired_last_cycle = true;
        }
    }
    
    // Konzept: Eligibility Trace / Kausalitätsgedächtnis (Kapitel 1.3)
    // Markiert kausal relevante Verbindungen für potenzielles Lernen.
    for (auto& synapse : synapses) {
        if(neurons[synapse.source_neuron_idx].activity > 0.1 && neurons[synapse.target_neuron_idx].fired_last_cycle){
            synapse.eligibility_trace = 1.0;
        }
    }
    return data;
}

void Network::apply_reward(double reward) {
    std::lock_guard<std::mutex> lock(network_mutex);

    // Konzept: Konfidenz-gewichtetes Lernen (Kapitel 2b.3.2)
    for (auto& synapse : synapses) {
        if (synapse.eligibility_trace > 0.1) {
            const auto& source_neuron = neurons[synapse.source_neuron_idx];
            const auto& target_neuron = neurons[synapse.target_neuron_idx];
            
            // Formel: Evidenz-Skalierungsfaktor = Ci * Cj
            double evidence_scaling = source_neuron.confidence * target_neuron.confidence;

            // Formel: ΔV_ij(t) = ΔV*R(t)*E_ij(t)*(Ci*Cj) - β_effektiv*V_ij
            double delta_trust = learning_rate * reward * synapse.eligibility_trace * evidence_scaling;

            synapse.trust_value += delta_trust;
            synapse.trust_value *= (1.0 - trust_decay);
            synapse.trust_value = std::max(0.0, std::min(1.0, synapse.trust_value));
        }
    }
}

size_t Network::get_total_neurons() const {
    std::lock_guard<std::mutex> lock(network_mutex);
    return neurons.size();
}

Neuron Network::get_neuron_copy(int idx) const {
    std::lock_guard<std::mutex> lock(network_mutex);
    if (idx < static_cast<int>(neurons.size())) {
        return neurons[idx];
    }
    throw std::out_of_range("Neuron index out of range.");
}

void Network::print_network_state() const {
    std::lock_guard<std::mutex> lock(network_mutex);
    std::cout << "Neurons: " << neurons.size() << ", Synapses: " << synapses.size() << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < neurons.size(); ++i) {
        const auto& neuron = neurons[i];
        std::cout << "  N" << i << " (D:" << neuron.id.domain_id << ",G:" << neuron.id.group_id << ",I:" << neuron.id.index_id << ")"
                  << " | Activity: " << neuron.activity
                  << " | Confidence: " << neuron.confidence
                  << " | Fired: " << (neuron.fired_last_cycle ? "Yes" : "No")
                  << std::endl;
    }
    print_synapse_trust();
}

void Network::print_synapse_trust() const {
    std::cout << "Synapse Trust Values [V_ij]: ";
    for (const auto& synapse : synapses) {
        std::cout << " (N" << synapse.source_neuron_idx << "->N" << synapse.target_neuron_idx << "): " << synapse.trust_value;
    }
    std::cout << std::endl;
}