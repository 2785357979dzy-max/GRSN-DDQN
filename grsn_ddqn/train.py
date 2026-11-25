import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .agent import DDQNAgent, seed_everything
from .config import Config
from .environment import RecoveryEnvironment
from .inputs import load_manifest


def train(network, scenarios, config, output, device="cpu"):
    selected = [s for s in scenarios if s["split"] == "train"]
    if len(selected) != 9:
        raise ValueError("The paper training protocol requires nine training scenarios")
    if any(not (s["damage"] > 0).any() for s in selected):
        raise ValueError("Every training scenario must contain damaged roads")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new or empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed)
    env = RecoveryEnvironment(network, config.repair_length_unit_m)
    agent = DDQNAgent(network, env.normal_x, config, device)
    budget = config.episodes_per_scenario * sum(int((s["damage"] > 0).sum()) for s in selected)
    (output / "config.json").write_text(json.dumps(config.as_dict(), indent=2), encoding="utf-8")
    fields = ["episode", "scenario", "steps", "epsilon", "mean_loss", "efficiency", "resilience"]
    episode = 0
    with (output / "training.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for cycle in range(config.episodes_per_scenario):
            for scenario in selected:
                state = env.reset(scenario["damage"])
                losses = []
                done = False
                while not done:
                    progress = min(agent.steps / max(budget - 1, 1), 1.0)
                    epsilon = config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)
                    action = agent.act(state, epsilon)
                    following, reward, done, info = env.step(action)
                    loss = agent.observe(state, action, reward, following, done)
                    if loss is not None:
                        losses.append(loss)
                    state = following
                episode += 1
                row = dict(episode=episode, scenario=scenario["id"], steps=agent.steps, epsilon=epsilon,
                           mean_loss=float(np.mean(losses)) if losses else "", efficiency=info["efficiency"],
                           resilience=info["resilience"])
                writer.writerow(row)
                stream.flush()
                print(json.dumps(row, ensure_ascii=False), flush=True)
            checkpoint = {"format_version": 1, "model": agent.online.state_dict(),
                          "config": config.as_dict(), "episodes": episode, "steps": agent.steps}
            temporary = output / "model.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(output / "model.pt")
    return agent


def main():
    parser = argparse.ArgumentParser(description="Train GRSN-DDQN using nine prepared damage scenarios")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = Config.load(args.config) if args.config else Config()
    network, scenarios = load_manifest(args.manifest)
    train(network, scenarios, config, args.output, args.device)


if __name__ == "__main__":
    main()
