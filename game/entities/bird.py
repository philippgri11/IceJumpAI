from dataclasses import dataclass
import random
from game.constants import GAME_WIDTH, GAME_HEIGHT, ICEBLOCK_WIDTH, WATER_HEIGHT, ICEBLOCK_HEIGHT

from game.entities.entity import Entity

@dataclass
class Bird(Entity):
    normalSpeedVecX: float = 0
    normalSpeedVecY: float = 0

    hits: int = 3

    def __init__(self):
        rand = random.Random(42)

        x: int = rand.randint(0, GAME_WIDTH - ICEBLOCK_WIDTH)
        y: int = GAME_HEIGHT - WATER_HEIGHT - 32
        vecX: float = rand.random() * 0.01 - 0.005
        if abs(vecX) < 0.001:
            if vecX < 0:
                vecX = -0.001
            else:
                vecX = 0.001

        vecY: float = rand.random() * 0.005 - 0.0025

        width: float = ICEBLOCK_WIDTH
        height: float = ICEBLOCK_HEIGHT

        super().__init__(x, y, width, height, vecX, vecY)

    def init(self):
        self.x = self.startX
        self.y = self.startY