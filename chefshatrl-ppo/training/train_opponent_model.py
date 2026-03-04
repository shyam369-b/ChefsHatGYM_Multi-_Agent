import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from genai.dataset import collect_dataset
from genai.opponent_model import OpponentModel


EPOCHS = 15
BATCH_SIZE = 64
LR = 1e-3


def train():
    states, actions = collect_dataset(num_episodes=300)

    state_dim = states.shape[1]
    action_dim = int(actions.max()) + 1

    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)

    model = OpponentModel(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    os.makedirs("results/models", exist_ok=True)

    dataset_size = len(states)

    for epoch in range(EPOCHS):
        perm = torch.randperm(dataset_size)
        total_loss = 0

        for i in range(0, dataset_size, BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]

            batch_states = states[idx]
            batch_actions = actions[idx]

            logits = model(batch_states)

            loss = criterion(logits, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (dataset_size // BATCH_SIZE)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "results/models/opponent_model.pth")

    print("✅ Opponent model trained and saved!")


if __name__ == "__main__":
    train()