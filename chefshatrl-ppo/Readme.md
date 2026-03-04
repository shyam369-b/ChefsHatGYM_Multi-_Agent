# Chef’s Hat RL Project (PPO + Experiments)


## 📌 Project Overview
This project implements a Reinforcement Learning (RL) agent using the Proximal Policy Optimization (PPO) algorithm to play the Chef’s Hat environment. The goal is to train an agent that maximizes cumulative reward through interaction with the environment.

---

## 🚀 Assigned Variant
Variant: PPO-based learning agent | Opponent Modelling Variant: Investigate the impact of different opponent 
behaviours, including training against varied baselines, explicit opponent modelling, or analysis 
of non-stationarity. 

- Algorithm: PPO (Policy Gradient)
- Environment: Chef’s Hat Gym
- Objective: Learn optimal strategy to maximize rewards

---

## ⚙️ Installation Guide

### 1. Install Python
Python 3.9 is recommended.

Check version:
    python --version

---

### 2. Create Virtual Environment
    py -3.9 -m venv venv
    venv\\Scripts\\activate

---

### 3. Install Dependencies
    pip install -r requirements.txt

If requirements.txt is not available:
    pip install torch numpy matplotlib gymnasium

---

### 4. Install Environment
    pip install chefshatgym

---

### 5. Set Python Path (As the library installation and build is not supported installing the env locally) 
    $env:PYTHONPATH=".;ChefsHatGYM\src"   

---

### 6. Editable ENV
    pip install -e ./ChefsHatGYM

---

## ▶️ Running the Project

### Train PPO Agent
    python training/train_ppo.py

Model will be saved in:
    results/models/ppo.pth

---

### Train Opponent Model Agent
    python training/train_ppo.py

Model will be saved in:
    results/models/opponent_model.pth

---

### Run Experiments
    python experiments/run_all_experiments.py

This runs evaluation episodes and logs performance.

---

### Plotting the Metrics results
    python experiments/plot_results.py

This runs evaluation episodes and logs performance.

---

## 📊 Experimental Outputs

Results are stored in:

    results/
        │
        ├── models/
        │   └── ppo.pth
        │   └── opponent_model.pth
        │
        ├── logs/
        │   └── metrics.json
        │   └── genai.npy
        │   └── ppo.npy
        │   └── random.npy
        │
        ├── plots/
        │   └── rewards_raw.png
        │   └── rewards_smmooth.png
        │   └── comparision.png

---

## 📈 Experiments Conducted

1. PPO Baseline Training
   - Agent trained using PPO
   - Rewards tracked per episode

2. Evaluation Runs
   - Model tested after training
   - Performance logged and compared

---

## 📊 Interpreting Results

- Increasing rewards → Learning is successful
- Stable rewards → Model has converged
- Fluctuating rewards → Needs tuning

---

## ⚠️ Common Errors

1. cannot unpack non-iterable int
   Fix:
       action, log_prob = agent.select_action(state)

2. store_reward not found
   Fix:
       Remove call or implement method correctly

3. tuple/float errors
   Fix:
       action = action[0] if isinstance(action, tuple) else action

---

## 📌 Summary
This project demonstrates:
- PPO implementation
- RL training pipeline
- Experiment tracking and evaluation


