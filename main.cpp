#include <iostream>
#include <vector>
#include <iomanip>
#include "Network.h"

int main() {
    // 1. Netzwerk initialisieren
    std::cout << "step 0\n" << std::endl;
    Network network;
    std::cout << "step 1\n" << std::endl;
    // 2. Neuronen zum Netzwerk hinzufügen
    // Sensor-Neuronen (z.B. N0, N1)
    network.add_neuron({0, 0, 0}); // ID: Domäne 0, Gruppe 0, Index 0
    std::cout << "step 2\n" << std::endl;
    network.add_neuron({0, 0, 1}); // ID: Domäne 0, Gruppe 0, Index 1

    // Abstraktes Neuron (z.B. N2)
    network.add_neuron({0, 1, 2}); // ID: Domäne 0, Gruppe 1, Index 2
    std::cout << "step 3\n" << std::endl;
    // 3. Synapsen (Verbindungen) erstellen
    // Verbindungen von den Sensor-Neuronen zum abstrakten Neuron
    network.add_synapse(0, 2); // N0 -> N2
    std::cout << "step 4\n" << std::endl;
    network.add_synapse(1, 2); // N1 -> N2

    std::cout << "Initial Network State:" << std::endl;
    network.print_network_state();
    std::cout << "\nStarting simulation for 50 cycles...\n" << std::endl;
    
    // 4. Simulationsschleife
    const int num_cycles = 50;
    for (int i = 0; i < num_cycles; ++i) {
        std::cout << "--- Cycle " << std::setw(2) << i + 1 << " ---" << std::endl;

        // Externe Signale an Sensor-Neuronen anlegen
        // Fall 1: Klares, hoch-konfidentes Signal
        if (i % 10 < 5) {
             std::cout << "Input: High-confidence, coherent signal" << std::endl;
             network.set_neuron_state(0, 0.9, 0.95); // Hohe Aktivität, hohe Konfidenz
             network.set_neuron_state(1, 0.85, 0.95); // Hohe Aktivität, hohe Konfidenz
        } 
        // Fall 2: Mehrdeutiges, niedrig-konfidentes Signal
        else {
            std::cout << "Input: Low-confidence, noisy signal" << std::endl;
            network.set_neuron_state(0, 0.9, 0.3); // Hohe Aktivität, niedrige Konfidenz
            network.set_neuron_state(1, 0.4, 0.4); // Geringere Aktivität, niedrige Konfidenz
        }

        // Netzwerkzyklus ausführen und Homöostase-Daten erhalten
        HomeostasisData data = network.network_cycle_step();

        // Belohnung anwenden, um das Lernen zu simulieren
        // Eine Belohnung stärkt Verbindungen, die zu diesem Zustand beigetragen haben.
        // Die Stärkung ist abhängig von der Konfidenz.
        if (network.get_neuron_copy(2).fired_last_cycle) {
            double reward = 1.0;
             std::cout << "Neuron 2 fired. Applying reward: " << reward << std::endl;
            network.apply_reward(reward);
        }

        // Status ausgeben
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Global Threshold: " << data.theta_global
                  << " | Volatility: " << data.volatility
                  << " | Beta: " << data.beta
                  << " | Active Neurons: " << static_cast<int>(data.A_current * network.get_total_neurons()) << "/" << network.get_total_neurons()
                  << std::endl;
        
        network.print_synapse_trust();
        std::cout << std::endl;
    }

    std::cout << "\n--- Final Network State ---" << std::endl;
    network.print_network_state();

    return 0;
}