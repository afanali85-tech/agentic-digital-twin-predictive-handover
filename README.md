# Agentic Digital Twin-Assisted Predictive Handover

This repository contains the code for the paper:

**Agentic Digital Twin-Assisted Multimodal Predictive Handover**

The project develops a site-specific Digital Twin (DT) and a three-agent decision framework for predictive handover in a five-gNB KFUPM deployment.

## Overview

The Digital Twin generates synchronized:

- Ray-traced RF path-power features;
- Link-validity information;
- UE mobility features;
- Mesh-derived geometric and virtual-LiDAR features.

The forecasting and handover pipeline then:

1. Builds trajectory-level training and testing folds;
2. Trains multiple future-link forecasting models;
3. Selects and evaluates handover actions;
4. Applies dynamic programming with Time-to-Trigger (DP–TTT);
5. Validates selected actions using realized ray-path availability.

## Agentic Framework

The proposed framework contains three cooperating logical agents:

### Forecasting Agent

Selects or evaluates forecasting tools for future link-quality prediction.

The implemented forecasting models are:

- Persistence;
- Deterministic Gated Recurrent Unit (GRU);
- Independent probabilistic GRU;
- Joint probabilistic GRU;
- Hurdle-aware joint probabilistic GRU.

### Planning Agent

Uses finite-horizon dynamic programming to compare possible `stay` and handover sequences. It considers:

- Future predicted link quality;
- Forecast uncertainty;
- Handover cost;
- Execution-failure risk;
- Time-to-Trigger persistence.

Only the first action of the optimized sequence is executed.

### Adaptation Agent

Validates the selected action using the realized ray-traced link-validity mask and updates the shared decision state used by later decisions.

The three agents are task-specific logical modules rather than separate large neural networks.

## Repository Structure

```text
agentic-digital-twin-predictive-handover/
├── notebooks/
│   ├── 01_build_loto_folds_hurdle_targets.ipynb
│   ├── 02_train_evaluate_improved_forecasters.ipynb
│   └── 03_improved_agentic_policy_and_report.ipynb
├── figures/
├── results/
├── README.md
├── requirements.txt
└── LICENSE
