#pragma once
#include <vector>

// Datenstruktur zur Rückgabe von Statusinformationen
struct HomeostasisData {
    double A_current;
    double A_global;
    double theta_global;
    double volatility;
    double beta;
    double error;
};

// Implementiert die globale Homöostase aus Kapitel 2
// basierend auf dem bereitgestellten Pseudocode.
class GlobalHomeostasis {
public:
    GlobalHomeostasis();

    // Hauptfunktion: Aktualisiert die Homöostase für einen Zeitschritt
    HomeostasisData update_homeostasis(int current_active_neurons, int total_neurons);
    
    // Gibt den aktuellen globalen Schwellwert zurück
    double get_current_threshold() const;

private:
    // Parameter (vgl. 2.1.3 Implementierungsdetails)
    double A_target;
    double alpha;
    double beta_min;
    double beta_max;
    double k_damping;
    double theta_min;
    double theta_max;
    int window_size;

    // Zustandsvariablen
    double A_global;
    double theta_global;
    std::vector<double> activity_history;

    // Private Hilfsfunktionen
    double calculate_volatility();
    double calculate_adaptive_beta(double volatility);
};