from dataclasses import dataclass

import numpy as np
from stable_baselines3 import DQN

from game.ai.ai import AI

from game.constants import GAME_WIDTH, GAME_HEIGHT

import game_state

@dataclass
class TrainedBot(AI):

    under = False
    left = 1
    MAX_BLOCKS = 1
    name = "DQN-Bot 01.03.2025"

    def __init__(self):
        super().__init__(self.name)
        self.model = DQN.load(
            "speichern/20250301_all.zip"
        )


    def think(self, index: int):
        action, _ = self.model.predict(self._get_obs(index), deterministic=True)
        self.vecX = float(action-1) * 0.16

    def _normalize_pos_x(self, x):
        return x / GAME_WIDTH

    def _normalize_pos_y(self, y):
        return y / GAME_HEIGHT

    def _get_obs(self, index: int):
        aiPlayer = game_state.level_instance.getPlayerOne()
        enemyPlayer = game_state.level_instance.getPlayerTwo()
        if index == 1:
            aiPlayer = game_state.level_instance.getPlayerTwo()
            enemyPlayer = game_state.level_instance.getPlayerOne()

        width = self._normalize_pos_x(aiPlayer.width)

        x0 = self._normalize_pos_x(aiPlayer.x)
        y0 = self._normalize_pos_y(aiPlayer.y)
        vecX0 = self._normalize_pos_y(aiPlayer.vecX)
        vecY0 = self._normalize_pos_y(aiPlayer.vecY)
        x1 = self._normalize_pos_x(enemyPlayer.x)
        y1 = self._normalize_pos_y(enemyPlayer.y)
        vecX1 = self._normalize_pos_y(enemyPlayer.vecX)
        vecY1 = self._normalize_pos_y(enemyPlayer.vecY)
        underLeft = 0
        if ((x0 + width >= x1) and (x0 <= x1 + width)) or self.under:
            self.under = True
            if self.under:
                distance = 35
                if abs(aiPlayer.x - enemyPlayer.x) > distance:
                    self.under = False
            if self.left == 1 and aiPlayer.x < 50:
                self.left = 2
            if self.left == 2 and aiPlayer.x + aiPlayer.width > GAME_WIDTH - 50:
                self.left = 1
            underLeft = self.left

        obs_blocks = []
        sorted_blocks = sorted(game_state.level_instance.getBlocks(), key=lambda block: abs(block.x - aiPlayer.x))
        for i, block in enumerate(sorted_blocks[:self.MAX_BLOCKS]):
            bx = self._normalize_pos_x(block.x)
            by = self._normalize_pos_y(block.y)
            hits = block.hits
            obs_blocks.extend([bx, by, hits])
        # Auffüllen mit Nullen
        while len(obs_blocks) < self.MAX_BLOCKS * 3:
            obs_blocks.append(-1.0)

        obs_players = np.array([x0, y0, x1, y1, vecX0, vecY0, vecX1, vecY1, underLeft], dtype=np.float32)

        return np.concatenate([obs_players,
                               np.array(obs_blocks[:3*self.MAX_BLOCKS], dtype=np.float32)])