from dataclasses import dataclass
from game.constants import ICEBLOCK_MAX_CHANGE_Y, GAME_WIDTH, GAME_HEIGHT, ICEBLOCK_WIDTH, WATER_HEIGHT, ICEBLOCK_HEIGHT, ICEBLOCK_DECREASE_VEC_X

from game.entities.entity import Entity
import game_state

@dataclass
class Block(Entity):
    normalSpeedVecX: float = 0
    normalSpeedVecY: float = 0

    hits: int = 3

    def __init__(self):

        x: float = game_state.rand.randint(0, GAME_WIDTH - ICEBLOCK_WIDTH)
        y: float = GAME_HEIGHT - WATER_HEIGHT - 32
        vecX: float = game_state.rand.random() * 0.01 - 0.005
        if abs(vecX) < 0.001:
            if vecX < 0:
                vecX = -0.001
            else:
                vecX = 0.001

        vecY: float = game_state.rand.random() * 0.005 - 0.0025

        width: float = ICEBLOCK_WIDTH
        height: float = ICEBLOCK_HEIGHT

        super().__init__(x, y, width, height, vecX, vecY)

    def think(self, delta):
        self.x = self.x + self.vecX * delta
        if self.x < 0 or self.x + self.width > GAME_WIDTH:
            self.vecX = -self.vecX
            if self.x < 0:
                self.x = 0
            elif self.x + self.width > GAME_WIDTH:
                self.x = GAME_WIDTH - self.width

        oldVecX = self.vecX
        if abs(self.vecX - self.normalSpeedVecX) != 0:
            if self.vecX > 0:
                self.vecX = self.vecX - ICEBLOCK_DECREASE_VEC_X
            else:
                self.vecX = self.vecX + ICEBLOCK_DECREASE_VEC_X

        if abs(self.vecX) <= abs(self.normalSpeedVecX) or (oldVecX < 0 and self.vecX > 0) or (oldVecX > 0 and self.vecX < 0):
            if oldVecX < 0:
                self.vecX = -abs(self.normalSpeedVecX)
            else:
                self.vecX = abs(self.normalSpeedVecX)

        self.y = self.y + self.vecY * delta
        if abs(self.startY - self.y) > ICEBLOCK_MAX_CHANGE_Y:
            self.vecY = -self.vecY

        if self.hits <= 0:
            self.visible = False