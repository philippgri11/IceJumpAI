from stable_baselines3 import PPO
import numpy as np

from PythonAgent import IceJumpEnv

# Umgebung erstellen
env = IceJumpEnv(player_index=0)

# Modell erstellen und trainieren
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Modell speichern
model.save("icejump_model_III")

# Auswertung: Wir spielen ein paar Episoden durch, um den Durchschnittsreward zu messen
n_eval_episodes = 10
episode_rewards = []

for i in range(n_eval_episodes):
    obs = env.reset()
    done = False
    total_reward = 0.0
    while not done:
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