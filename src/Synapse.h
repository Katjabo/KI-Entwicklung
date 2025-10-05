#pragma once

// Repräsentiert eine dynamische Synapse Sij gemäß Kapitel 1.3
class Synapse {
public:
    int source_neuron_idx;
    int target_neuron_idx;

    // A_ij: Aktivierungsstärke (kurzfristig). Hier als übertragener Wert im Zyklus berechnet.
    // V_ij: Vertrauenswert (langfristig). Wird durch Lernen angepasst.
    double trust_value = 0.6;

    // E_ij: Eligibility Trace (Kausalitätsgedächtnis).
    double eligibility_trace = 0.0;

    Synapse(int source_idx, int target_idx);
};