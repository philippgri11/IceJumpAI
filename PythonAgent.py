import gym
import json
import numpy as np
from gym import spaces
from py4j.java_gateway import JavaGateway

class IceJumpEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    MAX_BLOCKS = 20
    MAX_GOODIES = 5
    MAX_BIRDS = 5

    # Annahmen für Normalisierung
    GAME_WIDTH = 640.0
    GAME_HEIGHT = 480.0
    MAX_HITS = 10.0
    MAX_GOODIE_TYPE = 5.0  # Angenommen Goodie-Typen: 0 bis 5

    def __init__(self, player_index=0):
        super(IceJumpEnv, self).__init__()
        # Py4J Gateway
        self.gateway = JavaGateway()
        self.entry_point = self.gateway.entry_point

        self.player_index = player_index

        # Aktionen: float-Wert für die horizontale Geschwindigkeit
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Beobachtungsraum:
        # Spieler: 6 Werte
        # Blocks: 20 * 3 = 60 Werte
        # Goodies: 5 * 3 = 15 Werte
        # Birds: 5 * 2 = 10 Werte
        # Gesamt: 6 + 60 + 15 + 10 = 91 Werte
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(91,), dtype=np.float32)

        self.done = False
        self.last_y = None
        self.time_step = 0

    def reset(self):
        self.entry_point.startGame()
        self.done = False
        self.time_step = 0
        obs = self._get_obs()
        self.last_y = obs[1]  # y-Koordinate des eigenen Spielers
        return obs

    def step(self, action):
        # Aktion durchführen
        vec_x = float(action[0])
        self.entry_point.setPlayerAction(self.player_index, vec_x)
        self.time_step += 1

        # Einen Schritt simulieren
        self.entry_point.step()

        # Aktuellen Zustand abfragen
        obs = self._get_obs()
        self.done = self.entry_point.isGameOver()

        state_json = json.loads(self.entry_point.getState())
        winner = state_json.get("winner", None)

        reward = 0.1 * self.time_step

        if self.done and winner is not None:
            player_name = state_json["players"][self.player_index]["name"]
            if winner == player_name:
                reward += 1000.0
            else:
                reward = 0.0

        info = {}
        return obs, reward, self.done, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def _normalize_pos_x(self, x):
        return x / self.GAME_WIDTH

    def _normalize_pos_y(self, y):
        return y / self.GAME_HEIGHT

    def _normalize_hits(self, hits):
        return hits / self.MAX_HITS

    def _normalize_goodie_type(self, gtype):
        return gtype / self.MAX_GOODIE_TYPE

    def _get_obs(self):
        state_str = self.entry_point.getState()
        state_json = json.loads(state_str)
        players = state_json["players"]

        # Spielerzustand normalisieren
        x0 = self._normalize_pos_x(players[0]["x"])
        y0 = self._normalize_pos_y(players[0]["y"])
        v0 = 1.0 if players[0]["visible"] else 0.0
        x1 = self._normalize_pos_x(players[1]["x"])
        y1 = self._normalize_pos_y(players[1]["y"])
        v1 = 1.0 if players[1]["visible"] else 0.0

        obs_players = np.array([x0, y0, v0, x1, y1, v1], dtype=np.float32)

        # Blocks
        blocks = state_json.get("blocks", [])
        obs_blocks = []
        for i, block in enumerate(blocks[:self.MAX_BLOCKS]):
            bx = self._normalize_pos_x(block["x"])
            by = self._normalize_pos_y(block["y"])
            hits = self._normalize_hits(block["hits"])
            obs_blocks.extend([bx, by, hits])
        # Auffüllen mit Nullen
        while len(obs_blocks) < self.MAX_BLOCKS * 3:
            obs_blocks.append(0.0)

        # Goodies
        goodies = state_json.get("goodies", [])
        obs_goodies = []
        for i, goodie in enumerate(goodies[:self.MAX_GOODIES]):
            gx = self._normalize_pos_x(goodie["x"])
            gy = self._normalize_pos_y(goodie["y"])
            gtype = self._normalize_goodie_type(goodie["type"])
            obs_goodies.extend([gx, gy, gtype])
        while len(obs_goodies) < self.MAX_GOODIES * 3:
            obs_goodies.append(0.0)

        # Birds
        birds = state_json.get("birds", [])
        obs_birds = []
        for i, bird in enumerate(birds[:self.MAX_BIRDS]):
            bx = self._normalize_pos_x(bird["x"])
            by = self._normalize_pos_y(bird["y"])
            obs_birds.extend([bx, by])
        while len(obs_birds) < self.MAX_BIRDS * 2:
            obs_birds.append(0.0)

        obs = np.concatenate([obs_players,
                              np.array(obs_blocks, dtype=np.float32),
                              np.array(obs_goodies, dtype=np.float32),
                              np.array(obs_birds, dtype=np.float32)])
        return obs