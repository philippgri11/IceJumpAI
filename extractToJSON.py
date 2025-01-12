
import json

from stable_baselines3 import PPO, SAC

model = SAC.load("models/SAC_1736095942/2200000.zip")
weights = model.policy.state_dict()

# Konvertiere die Gewichtungen in JSON
with open("policy_weights.json", "w") as f:
    json.dump({k: v.tolist() for k, v in weights.items()}, f)