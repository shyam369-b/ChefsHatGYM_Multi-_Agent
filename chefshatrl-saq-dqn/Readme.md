# 🧠 Reinforcement Learning for Chef’s Hat Game

## 📌 Project Overview
This project focuses on training reinforcement learning agents to play the Chef’s Hat card game environment. The goal is to learn optimal strategies using modern RL algorithms and evaluate their performance.

## Variant : Opponent Modelling Variant: 
-Investigate the impact of different opponent 
behaviours, including training against varied baselines, explicit opponent modelling, or analysis 
of non-stationarity.

We implemented and compared two algorithms:
- Deep Q-Network (DQN)
- Soft Actor-Critic (SAC)

---

## ⚙️ Environment
The environment is based on a simplified version of the Chef’s Hat game.

- State: Numerical representation of game state
- Actions: Discrete action space
- Reward: Based on game progression and decisions
- Episode ends after fixed steps

---

## 🧠 Algorithms Used

### 1. Deep Q-Network (DQN)
- Value-based method
- Uses Q-learning with neural networks
- Experience replay buffer
- Target network stabilization

### 2. Soft Actor-Critic (SAC)
- Policy-based + value-based hybrid
- Uses entropy regularization
- Improves exploration and stability
- Uses twin critics

---

## 📊 Training Details

- Episodes: 500
- Batch size: 64
- Replay buffer: 100,000
- Optimizer: Adam
- Discount factor: 0.99

---

## 📈 Evaluation Metrics

We evaluated agents using:

- Average Reward
- Win Rate
- Learning Curve
- Stability across episodes

---

## 📊 Results

| Algorithm | Avg Reward | Win Rate |
|----------|-----------|----------|
| DQN      | ~40       | Low      |
| SAC      | ~70       | High     |

### Observations:
- SAC significantly outperforms DQN
- SAC shows stable learning behavior
- DQN struggles with exploration

---

## 🧪 Experiments Conducted

- Hyperparameter tuning
- Replay buffer size variation
- Training stability comparison
- Algorithm comparison (DQN vs SAC)

---

## Create Virtual Environment

```bash
py -3.9 -m venv venv
venv\Scripts\activate
$env:PYTHONPATH=".;ChefsHatGYM\src"
```


## Steps to run the project

```bash
python training/train_dqn.py      
python training/train_sac.py     
python evaluation/evaluate_agents.py
python demo/play_trained_agent.py 
```
## 🎮 Demo

To run trained agent:

```bash
python demo/play_trained_agent.py