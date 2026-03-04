import streamlit as st
import torch
import numpy as np
import time
import sys
import os



# ✅ Absolute path to ChefshatGym source
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GYM_PATH = os.path.join(BASE_DIR, "ChefsHatGYM", "src")

if GYM_PATH not in sys.path:
    sys.path.append(GYM_PATH)

# ✅ NOW import
from ChefsHatGym.env.ChefsHatEnv import ChefsHatEnv
from agents.network import ActorCritic


import matplotlib.pyplot as plt


# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Chef's Hat RL", layout="wide")

st.title("🎩 Chef's Hat - RL Game Dashboard")

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    model = ActorCritic(state_dim=4, action_dim=5)
    model.load_state_dict(torch.load("results/models/ppo_chefshat.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -------------------------------
# INIT ENV
# -------------------------------
if "env" not in st.session_state:
    st.session_state.env = ChefsHatEnv()
    st.session_state.state = st.session_state.env.reset()
    st.session_state.done = False
    st.session_state.rewards = []
    st.session_state.mode = "AI vs AI"

env = st.session_state.env

# -------------------------------
# MODE SELECT
# -------------------------------
mode = st.sidebar.selectbox(
    "Select Mode",
    ["AI vs AI", "Human vs AI"]
)

st.session_state.mode = mode

# -------------------------------
# RESET BUTTON
# -------------------------------
if st.sidebar.button("🔄 Reset Game"):
    st.session_state.state = env.reset()
    st.session_state.done = False
    st.session_state.rewards = []

# -------------------------------
# CARD UI
# -------------------------------
st.subheader("🃏 Game State")

st.markdown(f"""
<div style="
    padding:20px;
    border-radius:10px;
    background-color:#1e1e1e;
    color:white;
    text-align:center;
    font-size:20px;">
    Current State: {st.session_state.state}
</div>
""", unsafe_allow_html=True)

# -------------------------------
# HUMAN VS AI
# -------------------------------
def human_turn():
    st.subheader("🎮 Your Move")

    cols = st.columns(5)
    action = None

    for i in range(5):
        if cols[i].button(f"Action {i}"):
            action = i

    return action


def ai_action(state):
    state_tensor = torch.FloatTensor([state])
    logits, _ = model(state_tensor)
    probs = torch.softmax(logits, dim=-1)
    return torch.argmax(probs).item()


# -------------------------------
# GAME STEP
# -------------------------------
def step_game(action):
    next_state, reward, done, _ = env.step(action)

    st.session_state.state = next_state
    st.session_state.done = done
    st.session_state.rewards.append(reward)


# -------------------------------
# GAME LOOP UI
# -------------------------------
if not st.session_state.done:

    if mode == "Human vs AI":
        action = human_turn()

        if action is not None:
            step_game(action)

    else:
        if st.button("▶ Run AI Step"):
            action = ai_action(st.session_state.state)
            step_game(action)

else:
    st.success("🏆 Game Finished!")

# -------------------------------
# GENAI OPPONENT VISUALIZATION
# -------------------------------
st.subheader("🧠 Opponent Behaviour")

if len(st.session_state.rewards) > 0:
    st.write("Last Reward:", st.session_state.rewards[-1])

    if st.session_state.rewards[-1] > 0:
        st.success("Opponent Strategy: Aggressive")
    else:
        st.warning("Opponent Strategy: Defensive")

# -------------------------------
# REWARD GRAPH
# -------------------------------
st.subheader("📊 Reward Over Time")

if len(st.session_state.rewards) > 0:
    fig, ax = plt.subplots()
    ax.plot(st.session_state.rewards)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Reward")
    ax.set_title("Learning Curve")

    st.pyplot(fig)
else:
    st.info("No rewards yet")
