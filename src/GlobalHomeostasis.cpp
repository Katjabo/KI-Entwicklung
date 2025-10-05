#include "GlobalHomeostasis.h"
#include <cmath>
#include <numeric>
#include <algorithm>

GlobalHomeostasis::GlobalHomeostasis() {
    // Parameter initialisieren (Kapitel 2.1.3)
    A_target = 0.20;
    alpha = 0.03;
    beta_min = 0.05;
    beta_max = 0.3;
    k_damping = 2.0;
    theta_min = 0.1;
    theta_max = 0.9;
    window_size = 10;

    // Zustandsvariablen initialisieren
    A_global = A_target;
    theta_global = 0.5;
}

double GlobalHomeostasis::calculate_volatility() {
    if (activity_history.size() < 2) {
        return 0.0;
    }
    
    double sum = std::accumulate(activity_history.begin(), activity_history.end(), 0.0);
    double mean = sum / activity_history.size();
    
    double sq_sum = 0.0;
    for(const auto& val : activity_history) {
        sq_sum += (val - mean) * (val - mean);
    }
    double variance = sq_sum / activity_history.size();
    
    return std::sqrt(variance);
}

double GlobalHomeostasis::calculate_adaptive_beta(double volatility) {
    // Formel 2b: β(t) = β_min + (β_max - β_min) * exp(-k * σ_A(t))
    double exponential_factor = std::exp(-k_damping * volatility);
    double adaptive_beta = beta_min + (beta_max - beta_min) * exponential_factor;
    return std::max(beta_min, std::min(beta_max, adaptive_beta));
}

HomeostasisData GlobalHomeostasis::update_homeostasis(int current_active_neurons, int total_neurons) {
    // 1. Aktuelle Aktivität berechnen
    double A_current = (total_neurons > 0) ? static_cast<double>(current_active_neurons) / total_neurons : 0.0;

    // 2. Aktivitätsverlauf aktualisieren
    activity_history.push_back(A_current);
    if (activity_history.size() > window_size) {
        activity_history.erase(activity_history.begin());
    }

    // 3. Volatilität und adaptives β berechnen
    double volatility = calculate_volatility();
    double beta = calculate_adaptive_beta(volatility);

    // 4. Exponentiell geglättete globale Aktivität aktualisieren (Formel 2a)
    A_global = beta * A_current + (1 - beta) * A_global;

    // 5. Schwellwert anpassen (P-Regler, Formel 4)
    double error = A_global - A_target;
    theta_global += alpha * error;

    // 6. Schwellwert begrenzen (Formel 5)
    theta_global = std::max(theta_min, std::min(theta_max, theta_global));

    return {A_current, A_global, theta_global, volatility, beta, error};
}

double GlobalHomeostasis::get_current_threshold() const {
    return theta_global;
}