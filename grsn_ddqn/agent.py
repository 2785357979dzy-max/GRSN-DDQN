import copy
import random

import numpy as np
import torch
from torch import nn

from .model import GRSN
from .replay import ReplayBuffer


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def observation_tensors(state, device):
    return {key: torch.as_tensor(value, dtype=torch.bool if key == "mask" else torch.float32,
                                  device=device).unsqueeze(0) for key, value in state.items()}


def ddqn_targets(online, target, next_states, rewards, dones, gamma):
    with torch.no_grad():
        active = next_states["mask"].any(dim=1)
        actions = online(**next_states).argmax(dim=1)
        estimates = target(**next_states).gather(1, actions[:, None]).squeeze(1)
        estimates = torch.where(active & ~dones, estimates, torch.zeros_like(estimates))
        return rewards + gamma * estimates


class DDQNAgent:
    def __init__(self, network, normal_x, config, device="cpu"):
        self.config = config
        self.device = torch.device(device)
        self.rng = random.Random(config.seed)
        self.online = GRSN(network, normal_x, config).to(self.device)
        self.target = copy.deepcopy(self.online).requires_grad_(False)
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.loss_function = nn.SmoothL1Loss()
        self.replay = ReplayBuffer(config.replay_capacity)
        self.steps = 0

    def act(self, state, epsilon=0.0):
        candidates = np.flatnonzero(state["mask"])
        if not candidates.size:
            raise ValueError("No damaged roads remain")
        if self.rng.random() < epsilon:
            return int(self.rng.choice(candidates))
        with torch.no_grad():
            return int(self.online(**observation_tensors(state, self.device)).argmax(dim=1).item())

    def observe(self, state, action, reward, next_state, done):
        self.replay.add(state, action, reward, next_state, done)
        self.steps += 1
        loss_value = None
        if len(self.replay) >= self.config.batch_size:
            states, actions, rewards, following, dones = self.replay.sample(
                self.config.batch_size, self.rng, self.device
            )
            targets = ddqn_targets(self.online, self.target, following, rewards, dones, self.config.gamma)
            predictions = self.online(**states).gather(1, actions[:, None]).squeeze(1)
            loss = self.loss_function(predictions, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            loss_value = float(loss.item())
        if self.steps % self.config.target_update_steps == 0:
            self.target.load_state_dict(self.online.state_dict())
        return loss_value
