import pygame
from stable_baselines3 import PPO, A2C, DQN, SAC
from PythonAgent import IceJumpEnv
import os
import time

import numpy as np

training = True

myTime = int(time.time())
modelName = "DQN"
models_dir = "models/"+modelName+"_"+str(myTime)
log_dir = "logs/"

if training:
    if not os.path.exists(models_dir) :
        os.makedirs(models_dir)

    if not os.path.exists(log_dir) :
        os.makedirs(log_dir)

# Umgebung erstellen
env = IceJumpEnv(player_index=0, render_mode="human")
env.reset(42)


# Modell erstellen und trainieren
#model = PPO("MlpPolicy", env, verbose=1, device="cuda", tensorboard_log=log_dir, ent_coef=0.01, learning_rate=1e-3, n_steps=2048, clip_range=0.1)
#model = SAC("MlpPolicy", env, batch_size=1024, device="auto", verbose=1, tensorboard_log=log_dir)
'''
model = DQN(
    "MlpPolicy",
    env,
    batch_size=4096,               # Größere Batch-Größe
    train_freq=128,                # Weniger häufiges Training
    gradient_steps=64,             # Mehr Gradientenschritte pro Aktualisierung
    buffer_size=1000000,           # Größerer Replay-Buffer
    #ent_coef="auto",               # Dynamische Entropie-Anpassung
    learning_rate=1e-3,            # Schnellere Lernrate
    tau=0.02,                      # Schnelleres Update des Zielnetzwerks
    device="cuda",                 # Nutzung der GPU
    verbose=1,
    tensorboard_log=log_dir        # Logging für Tensorboard
)
'''
model = DQN.load(
    "models/DQN_1738416413/11000000.zip",
    env,
    batch_size=4096,               # Größere Batch-Größe
    train_freq=128,                # Weniger häufiges Training
    gradient_steps=64,             # Mehr Gradientenschritte pro Aktualisierung
    buffer_size=1000000,           # Größerer Replay-Buffer
    ent_coef="auto",               # Dynamische Entropie-Anpassung
    #learning_rate=1e-3,            # Schnellere Lernrate
    tau=0.02,                      # Schnelleres Update des Zielnetzwerks
    device="cuda",                 # Nutzung der GPU
    verbose=1,
    tensorboard_log=log_dir        # Logging für Tensorboard
)

TIMESTAMP = 10000
index = 0
for i in range(1,3000):
    model.learn(total_timesteps=TIMESTAMP, reset_num_timesteps=False, tb_log_name=modelName+"_"+str(myTime))
    index = i
    # Modell speichern
    if i % 10 == 0:
        model.save(f"{models_dir}/{TIMESTAMP*i}")

model.save(f"{models_dir}/{TIMESTAMP*index}")

env.setupWindow()

# Auswertung: Wir spielen ein paar Episoden durch, um den Durchschnittsreward zu messen
n_eval_episodes = 100
episode_rewards = []

env = model.get_env()

for i in range(n_eval_episodes):
    obs = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        # Pygame-Events verarbeiten
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                pygame.quit()
                exit()  # Beendet das Programm bei Fenster-Schließen

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