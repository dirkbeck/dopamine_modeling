This repository contains all code required to reproduce the figures in “Prediction error correlates in the striosome-dopamine circuit emerge from information gain” (Beck & Friedman, in submission). All data is included as Excel files. No external downloads are needed. All figures other than those produced from the code here are schematic illustrations created in Adobe Illustrator.

---

## Installation

1. Clone the repository and enter its folder:  
   ```bash
   git clone https://github.com/dirkbeck/dopamine_modeling.git
   cd dopamine_modeling
   ```

2. (Recommended) Create and activate a Python virtual environment:  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install Python dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

---

## Directory Structure & Script-Figure Map

Below is a map from each subdirectory and script to the figure(s) it generates.

**Associative_learning_correlations**  
- `associative_learning_RPE_pIG_correlation.py` → Figs 2d-g  
- `associative_learning_RPE_pIG_large_cost_and_novelty.py` → Figs 3b, d, f  
- `theoretical_3D_curve.py` → Fig 3g  
- `theoretical_2D_curve.py` → Fig S3a  
- `experimental_data_fit_Kutlu_et_al_utility.py` → Fig S3b  
- `experimental_data_fit_Stauffer_et_al.py` → Fig S3c  
- `novelty_colormap.py` → Fig S3d  
- `foraging_with_large_costs_RPE_pIG_relationship.py` → Fig S3f  
- `open_gym_comparison.py` → Figs S3g-i  
- `experimental_data_fit_Kutlu_et_al_novelty.py` → Fig S3j  
- `experimental_data_fit_Fiorillo_et_al.py` → Fig S3k  

**Ramping_and_state_value**  
- `basic_ramping_and_teleport.py` → Figs 4b, d  
- `probabilistic_task_ramping.py` → Figs 4e, f  
- `dopamine_and_state_value_correlation.py` → Fig S4  

**Action_initiation**  
- `policy_shift_action_initiation.py` → Fig 5a  
- `hierarchical_melody_note_primative_simulation.py` → Fig 5d  
- `clusters_of_brackets.py` → Fig S7  
- `sensivity_by_baseline_policy_IG.py` → Fig S8  

**Decision_making_biases**  
- `risk_impulsivity_and_conflict.py` → Figs 6a-c  
- `Ushapeperformance.py` → Fig 6d  
- `decision_making_manifold.py` → Fig S9  

**Disorders**  
- `coupled_infoIG_RL_sim.py` → Figs 7g, h  
- `conceptual_disorder_policy_IG_axis.py` → Fig S10  

**Circuit_design**  
- `ACG_simulations_main_execution.py` → Fig 8b, Figs S11c-e  
  - Imports: `agents.py`, `environment.py`, `layers.py`, `plotting_functions.py`  
- `striosome_dopamine_control.py` → Fig 8e, Fig S12b  
- `d1_d2_both_necessary.py` → Fig 8g, Figs S13c-d  
- `distribution_across_neurons.py` → Figs 8i-j  
- `d1_d2_lead_to_dual_param_control.py` → Fig S13a  
- `four_stream_policies_over_time.py` → Fig S14c  

**Naturalistic_foraging**  
- `naturalistic_rodent_foraging.py` → Fig S15  

**Multi_cue_learning**  
- `chunking_vs_TD_learning_blocking_task.py` → Fig S5d  
- `chunking_vs_TD_learning_two_cue_task.py` → Figs S5f, g, i, j  

**Weber_law_surprise**  
- `Webers_law_and_policyIG.py` → Figs S6b-c  

---

## Dependencies & Data Files

Most scripts depend only on packages listed in `requirements.txt`. Two categories have extra dependencies:

1. **Excel data files** (in `Associative_learning_correlations`):  
   - `Kutlu_et_al_data_approximations.xlsx`  
     • used by  
       - `experimental_data_fit_Kutlu_et_al_utility.py` (Fig S3b)  
       - `experimental_data_fit_Kutlu_et_al_novelty.py` (Fig S3j)  
   - `Stauffer_et_al_data_approximations.xlsx`  
     • used by  
       - `experimental_data_fit_Stauffer_et_al.py` (Fig S3c)  
   - `Fiorillo_et_al_data_approximations.xlsx`  
     • used by  
       - `experimental_data_fit_Fiorillo_et_al.py` (Fig S3k)  

2. **Local package imports** (in `Circuit_design`):  
   - Scripts import modules from `Circuit_design/`:  
     `agents.py`, `environment.py`, `layers.py`, `plotting_functions.py`  

All other scripts are self-contained and require no extra data or modules beyond `requirements.txt`.

---

## How to Run Every Script

From the repository root, execute:

```bash
python path/to/subdirectory/script_name.py
```

Below are special instructions for scripts with extra dependencies:

1. Experimental‐data scripts (require Excel files):  
   ```bash
   python Associative_learning_correlations/experimental_data_fit_Kutlu_et_al_utility.py
   python Associative_learning_correlations/experimental_data_fit_Kutlu_et_al_novelty.py
   python Associative_learning_correlations/experimental_data_fit_Stauffer_et_al.py
   python Associative_learning_correlations/experimental_data_fit_Fiorelli_et_al.py
   ```

2. Main Circuit_design simulation (imports local modules):  
   ```bash
   python Circuit_design/ACG_simulations_main_execution.py
   ```

3. All other scripts (including the rest of Circuit_design):  
   ```bash
   python <other_subdirectory>/<script>.py
   ```
---

## Contact

For questions or issues, please email **dirkwadebeck@gmail.com**
