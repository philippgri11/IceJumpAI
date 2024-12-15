from py4j.java_gateway import JavaGateway, CallbackServerParameters
from stable_baselines3 import PPO
import numpy as np

from PythonAgent import IceJumpEnv

class PythonModel(object):
    class Java:
        implements = ["apoIcejump.ai.PythonModelInterface"]

    def __init__(self):
        # Umgebung erstellen
        self.env = IceJumpEnv(player_index=0)

        # Modell laden
        self.model = PPO.load("icejump_model_III", env=self.env)

    def predictAction(self, obs):
        # obs ist ein Java-Array, in ein Numpy-Array konvertieren
        obs = np.array(obs, dtype=np.float32)

        # Aktion vorhersagen
        action, _ = self.model.predict(obs, deterministic=True)
        # model ist hier self.model, weil wir auf Instanzvariable zugreifen

        # Da 'action' oft ein Array ist, und wir hier vielleicht nur einen einzelnen float brauchen:
        # Wenn die Action-Shape (1,) ist, dann action[0] zurückgeben.
        # Ansonsten, wenn das Model die richtigen Dimensionen liefert, einfach action zurückgeben.
        return float(action[0]) if isinstance(action, np.ndarray) else float(action)

# Gateway mit Callback-Server starten
gateway = JavaGateway(callback_server_parameters=CallbackServerParameters())
my_bot = gateway.entry_point

# Python-Modellobjekt erstellen
model = PythonModel()

# Modell an den Bot übergeben
my_bot.setPythonModel(model)

print("Python-Modell wurde an den Bot übergeben. Der Bot kann nun predictAction aufrufen!")