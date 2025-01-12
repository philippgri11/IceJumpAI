from dataclasses import dataclass

from game.ai.ai import AI
from game.entities.entity import Entity
from game.constants import PLAYER_HIT_BLOCK_VEC_Y, PLAYER_HIT_BLOCK_DIFFERENCE_VEC_Y, ICEBLOCK_HIT_VEC_X, GAME_SUDDEN_DEATH_TIME, WATER_HEIGHT, PLAYER_WIDTH, PLAYER_HEIGHT, PLAYER_HIT_PLAYER_VEC_Y, PLAYER_MAX_VEC_X_OVER_ENEMY, GAME_HEIGHT, PLAYER_DECREASE_Y, PLAYER_MAX_VEC_Y, GAME_WIDTH
import game_state
import pygame


@dataclass
class Player(Entity):
    name: str = "Human"
    ai: AI = None
    index: int = 0

    def __init__(self, name: str, x: float = 0, y: float = 0, index: int = 0):
        self.name = name

        self.index = index

        width: float = PLAYER_WIDTH
        height: float = PLAYER_HEIGHT

        print(x, y, width, height)
        super().__init__(x, y, width, height, 0, 0)

    def getName(self):
        if self.ai is None:
            return "Human"
        return self.ai.name

    def setAI(self, ai: AI):
        self.ai = ai
        if self.ai is None:
            self.name = "Human"
        else:
            self.name = self.ai.name

    def thinkAI(self):
        if self.ai is not None:
            self.ai.think(self.index)
            self.vecX = self.ai.vecX

    def think(self, delta: float):
        self.thinkPlayerCollision(delta)
        self.vecY = self.vecY + PLAYER_DECREASE_Y * delta
        if self.vecY > PLAYER_MAX_VEC_Y:
            self.vecY = PLAYER_MAX_VEC_Y
        speed = 1
        self.x = self.x + self.vecX * delta * speed
        self.y = self.y + self.vecY * delta


        if self.x < 0:
            self.x = 0
        elif (self.x + self.width) > GAME_WIDTH:
            self.x = GAME_WIDTH - self.width

        if self.y > GAME_HEIGHT - WATER_HEIGHT:
            self.visible = False

        self.thinkBlockCollision(delta)

    def thinkBlockCollision(self, delta: float):
        self.blockCheck(game_state.level_instance.getBlocks(), game_state.level_instance.time)
        pass

    def blockCheck(self, blocks, time: int):
        for index, value in enumerate(blocks):
            block = value
            if self.intersects(block):
                if (self.isAboveEntity(block)):
                    if time <= GAME_SUDDEN_DEATH_TIME:
                        block.hits -= 1
                    else:
                        block.hits = 0
                    self.y = block.y - self.height

                    difference = block.x + block.width/2 - self.x - self.width/2
                    block.vecX = difference * ICEBLOCK_HIT_VEC_X

                    moreHeight = 1
                    if abs(difference) < self.width/4:
                        self.vecY = PLAYER_HIT_BLOCK_VEC_Y * moreHeight
                    else:
                        self.vecY = PLAYER_HIT_BLOCK_VEC_Y * moreHeight + (abs(difference) - self.width/4) * PLAYER_HIT_BLOCK_DIFFERENCE_VEC_Y

    def thinkPlayerCollision(self, delta: float):
        enemy = game_state.level_instance.getEnemy(self.index)
        if self.intersects(enemy):
            if self.isAboveEntity(enemy):
                self.y = enemy.y - self.height
                moreHeight = 1
                self.vecY = PLAYER_HIT_PLAYER_VEC_Y * moreHeight
                if enemy.vecY < 0:
                    enemy.vecY = 0
            elif self.isDownEntity(enemy):
                enemy.y = self.y - enemy.height
                moreHeight = 1
                enemy.vecY = PLAYER_HIT_PLAYER_VEC_Y * moreHeight
                if self.vecY < 0:
                    self.vecY = 0
            else:
                if self.x < enemy.x:
                    enemy.x = self.x + enemy.width
                else:
                    enemy.x = self.x - self.width
        elif self.y < enemy.y:
            if enemy.getRec().colliderect(pygame.Rect(self.x - 3, 0, self.width + 6, GAME_HEIGHT)):
                if abs(self.vecX) > PLAYER_MAX_VEC_X_OVER_ENEMY:
                    if self.vecX < 0:
                        self.vecX = -PLAYER_MAX_VEC_X_OVER_ENEMY
                    else:
                        self.vecX = PLAYER_MAX_VEC_X_OVER_ENEMY
