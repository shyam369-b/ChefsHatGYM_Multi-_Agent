import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm

from genai.dataset import collect_dataset
from genai.opponent_model import OpponentModel


# =========================
# CONFIG
# =========================
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3
MODEL_PATH = "results/models/opponent_model.pth"


# =========================
# TRAIN FUNCTION
# =========================
def train():
    print("🚀 Collecting dataset...")
    states, actions = collect_dataset(num_episodes=300)

    # Convert to numpy (safety)
    states = np.array(states)
    actions = np.array(actions)

    state_dim = states.shape[1]
    action_dim = int(actions.max()) + 1

    print(f"✅ State dim: {state_dim}, Action dim: {action_dim}")
    print(f"📊 Dataset size: {len(states)}")

    # Convert to tensors
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)

    dataset_size = len(states)

    # Model
    model = OpponentModel(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Create folder
    os.makedirs("results/models", exist_ok=True)

    # =========================
    # TRAINING LOOP
    # =========================
    for epoch in range(EPOCHS):
        permutation = torch.randperm(dataset_size)

        total_loss = 0

        for i in range(0, dataset_size, BATCH_SIZE):
            indices = permutation[i:i + BATCH_SIZE]

            batch_states = states[indices]
            batch_actions = actions[indices]

            # Forward
            preds = model(batch_states)

            # FIX: ensure correct shape
            loss = criterion(preds, batch_actions)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, (dataset_size // BATCH_SIZE))
        print(f"📉 Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    # =========================
    # SAVE MODEL
    # =========================
    torch.save(model.state_dict(), MODEL_PATH)

    print(f"✅ Opponent model saved at: {MODEL_PATH}")


if __name__ == "__main__":
    train()