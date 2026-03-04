import streamlit as st
import numpy as np
import torch
import os
import json
from PIL import Image

from env.make_env import make_env
from agents.ppo_agent import PPOAgent
from genai.opponent_model import OpponentModel

st.set_page_config(page_title="Chef's Hat RL Dashboard", layout="wide")

st.title("🎩 Chef’s Hat RL - Research Dashboard")

# =========================
# LOAD METRICS
# =========================
if os.path.exists("results/logs/metrics.json"):
    with open("results/logs/metrics.json") as f:
        metrics = json.load(f)
else:
    metrics = None


# =========================
# METRICS DISPLAY
# =========================
st.header("📊 Performance Metrics")

if metrics:
    col1, col2, col3 = st.columns(3)

    col1.metric("PPO Avg Reward", round(metrics["ppo"]["mean_reward"], 2))
    col2.metric("Random Avg Reward", round(metrics["random"]["mean_reward"], 2))
    col3.metric("GenAI Avg Reward", round(metrics["genai"]["mean_reward"], 2))

    st.write("### Detailed Metrics")
    st.json(metrics)
else:
    st.warning("⚠ Run experiments first to generate metrics")


# =========================
# PLOTS
# =========================
st.header("📈 Learning Curves")

if os.path.exists("results/plots/rewards_raw.png"):
    st.image("results/plots/rewards_raw.png", caption="Raw Rewards")

if os.path.exists("results/plots/rewards_smooth.png"):
    st.image("results/plots/rewards_smooth.png", caption="Smoothed Rewards")


# =========================
# GAME SECTION
# =========================
st.header("🎮 Play vs AI")

env = make_env()
state = env.reset()

state_dim = len(state)
action_dim = 5

agent = PPOAgent(state_dim, action_dim)

if os.path.exists("results/models/ppo.pth"):
    agent.load("results/models/ppo.pth")

# Load opponent model
opponent_model = OpponentModel(state_dim, action_dim)
if os.path.exists("results/models/opponent_model.pth"):
    opponent_model.load_state_dict(torch.load("results/models/opponent_model.pth"))
    opponent_model.eval()


# =========================
# SESSION STATE
# =========================
if "state" not in st.session_state:
    st.session_state.state = state

if "done" not in st.session_state:
    st.session_state.done = False


# =========================
# ACTION BUTTONS
# =========================
st.write("### Choose Your Action")

cols = st.columns(action_dim)

for i in range(action_dim):
    if cols[i].button(f"Play {i}"):
        if not st.session_state.done:
            next_state, reward, done, _ = env.step(i)

            st.session_state.state = next_state
            st.session_state.done = done

            st.write(f"🤖 AI Reward: {reward}")

            # Opponent prediction visualization
            state_tensor = torch.tensor(next_state, dtype=torch.float32)
            probs = torch.softmax(opponent_model(state_tensor), dim=-1)

            st.write("🧠 Opponent Prediction:", probs.detach().numpy())

            if done:
                st.success("🏁 Game Finished!")


# =========================
# RESET BUTTON
# =========================
if st.button("🔄 Reset Game"):
    st.session_state.state = env.reset()
    st.session_state.done = False
    st.success("Game Reset!")