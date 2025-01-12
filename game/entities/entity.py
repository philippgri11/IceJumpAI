from dataclasses import dataclass

from game.constants import ABOVE_PIXEL
import pygame


@dataclass
class Entity:
    x: float
    y: float
    startX: float
    startY: float
    vecX: float
    vecY: float
    width: float
    height: float
    visible: bool

    def __init__(self, x: float, y: float, width: float, height: float, vecX: float, vecY: float):
        self.x = x
        self.y = y
        self.startX = x
        self.startY = y
        self.width = width
        self.height = height
        self.vecX = vecX
        self.vecY = vecY
        self.visible = True

    def init(self):
        self.x = self.startX
        self.y = self.startY
        self.vecX = 0
        self.vecY = 0
        self.visible = True

    def think(self, delta: int):
        self.x += delta * self.vecX
        self.y += delta * self.vecY

    def getRec(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def intersects(self, entity):
        return self.getRec().colliderect(entity.getRec())

    def isAboveEntity(self, entity):
        if self.y + self.height < entity.y + ABOVE_PIXEL:
            return True
        return False

    def isDownEntity(self, entity):
        if entity.y + entity.height < self.y + ABOVE_PIXEL:
            return True
        return False
