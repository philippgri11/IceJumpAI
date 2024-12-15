from stable_baselines3 import PPO
from PythonAgent import IceJumpEnv

# Umgebung erstellen
env = IceJumpEnv(player_index=0)

# Modell laden
model = PPO.load("icejump_model", env=env)

# Nun können Sie das geladene Modell verwenden, um Aktionen vorherzusagen
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    # hier könnte man z. B. den reward sammeln oder andere Auswertungen machen

env.close()