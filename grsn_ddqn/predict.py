import argparse
import json
from pathlib import Path

import torch

from .agent import observation_tensors
from .config import Config
from .environment import RecoveryEnvironment
from .inputs import load_manifest
from .model import GRSN


def predict(network, damage, checkpoint, device="cpu"):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    config = Config(**saved["config"])
    env = RecoveryEnvironment(network, config.repair_length_unit_m)
    model = GRSN(network, env.normal_x, config).to(device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    state = env.reset(damage)
    rows = [{"road_index": None, "elapsed_time": 0.0, "efficiency": float(env.x.mean()), "resilience": 0.0}]
    with torch.no_grad():
        while state["mask"].any():
            action = int(model(**observation_tensors(state, device)).argmax(dim=1).item())
            state, reward, done, info = env.step(action)
            rows.append({"road_index": action, "reward": reward, **info})
    return {"sequence": [r["road_index"] for r in rows[1:]], "curve": rows,
            "resilience": float(env.resilience), "elapsed_time": float(env.elapsed_time)}


def main():
    parser = argparse.ArgumentParser(description="Generate a road recovery sequence with a trained GRSN-DDQN model")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose an output file that does not already exist")
    network, scenarios = load_manifest(args.manifest)
    matches = [s for s in scenarios if s["id"] == args.scenario]
    if not matches:
        parser.error("The requested scenario is not present in the manifest")
    result = predict(network, matches[0]["damage"], args.checkpoint, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {len(result['sequence'])} repair decisions to {args.output}")


if __name__ == "__main__":
    main()
