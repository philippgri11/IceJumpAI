from stable_baselines3 import PPO, A2C, DQN, SAC
from PythonAgent import IceJumpEnv
import os
import time

import numpy as np

myTime = int(time.time())
modelName = "PPO"
models_dir = "models/"+modelName+"_"+str(myTime)
log_dir = "logs/"

if not os.path.exists(models_dir) :
    os.makedirs(models_dir)

if not os.path.exists(log_dir) :
    os.makedirs(log_dir)

# Umgebung erstellen
env = IceJumpEnv(player_index=0, render_mode="human")
env.reset(42)


# Modell erstellen und trainieren
model = PPO("MlpPolicy", env, verbose=1, device="cpu", tensorboard_log=log_dir, ent_coef=0.01, learning_rate=1e-5, n_steps=2048, clip_range=0.1)
#model = SAC("MlpPolicy", env, batch_size=1024, device="cuda", verbose=1, tensorboard_log=log_dir)
#model = SAC.load("models/SAC_1736154619/3400000.zip", env=env, batch_size=1024, device="auto", verbose=1, tensorboard_log=log_dir)

TIMESTAMP = 10000
for i in range(1,2):
    model.learn(total_timesteps=TIMESTAMP, reset_num_timesteps=False, tb_log_name=modelName+"_"+str(myTime))
    # Modell speichern
    if i % 10 == 0:
        model.save(f"{models_dir}/{TIMESTAMP*i}")

# Auswertung: Wir spielen ein paar Episoden durch, um den Durchschnittsreward zu messen
n_eval_episodes = 10
episode_rewards = []

env = model.get_env()

for i in range(n_eval_episodes):
    obs = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        env.render()
        # Aktion vorhersagen (ohne Lernmodus, nur inferieren)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        total_reward += reward
    episode_rewards.append(total_reward)

mean_reward = np.mean(episode_rewards)
std_reward = np.std(episode_rewards)

print(f"Auswertung über {n_eval_episodes} Episoden:")
print(f"Durchschnittsreward: {mean_reward} ± {std_reward}")

# Optional: Umwelt schließen

env.close()

# Jetzt haben wir unser Modell trainiert, gespeichert und einfach ausgewertet.
# Das Modell kann später mit model = PPO.load("icejump_model", env=env) erneut geladen werden.