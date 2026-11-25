from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity):
        self.transitions = deque(maxlen=capacity)

    def __len__(self):
        return len(self.transitions)

    def add(self, state, action, reward, next_state, done):
        current = {k: v.copy() for k, v in state.items()}
        following = {k: v.copy() for k, v in next_state.items()}
        self.transitions.append((current, action, reward, following, done))

    def sample(self, size, rng, device):
        rows = rng.sample(list(self.transitions), size)
        states = {}
        next_states = {}
        for key in ("x", "damage", "mask"):
            dtype = torch.bool if key == "mask" else torch.float32
            states[key] = torch.as_tensor(np.stack([r[0][key] for r in rows]), dtype=dtype, device=device)
            next_states[key] = torch.as_tensor(np.stack([r[3][key] for r in rows]), dtype=dtype, device=device)
        actions = torch.tensor([r[1] for r in rows], dtype=torch.long, device=device)
        rewards = torch.tensor([r[2] for r in rows], dtype=torch.float32, device=device)
        dones = torch.tensor([r[4] for r in rows], dtype=torch.bool, device=device)
        return states, actions, rewards, next_states, dones
