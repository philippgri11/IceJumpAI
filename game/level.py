import random
import time as timer

from game.ai.Easy import Easy
from game.ai.Hard import Hard
from game.ai.Middle import Middle
from game.entities.block import Block
from game.entities.player import Player
from game.constants import GAME_SUDDEN_DEATH_TIME, GAME_WIDTH, PLAYER_WIDTH, GAME_HEIGHT

import numpy as np
import game_state

class Level:
    blocks = np.empty((1,), dtype=Block)
    playerOne: Player = None
    playerTwo: Player = None
    time: int = 0

    def __init__(self):
        game_state.level_instance = self
        if self.playerOne is None:
            self.playerOne = Player("Human", GAME_WIDTH*1/4 - PLAYER_WIDTH/2, GAME_HEIGHT/2 - 50, 0)
        if self.playerTwo is None:
            self.playerTwo = Player("Second", GAME_WIDTH*3/4 - PLAYER_WIDTH/2, GAME_HEIGHT/2 - 100, 1)
            self.playerTwo.setAI(Hard())

    def init(self):
        self.createBlocks()
        self.playerOne.init()
        self.playerTwo.init()
        self.time = 0

    def changePlayer(self):
        if self.playerOne.ai is None:
            self.playerOne.setAI(Hard())
            self.playerTwo.setAI(None)
            return 1
        else:
            self.playerOne.setAI(None)
            self.playerTwo.setAI(Hard())
            return 0

    def createBlocks(self):
        rand = random.Random((int)(timer.time()))
        count: int = rand.randint(0, 30) + 10
        block_array = []
        for i in range(count):
            arr = Block()
            block_array.append(arr)
        self.blocks = np.array(block_array)

    def getBlocks(self):
        return self.blocks

    def getPlayerOne(self):
        return self.playerOne

    def getPlayerTwo(self):
        return self.playerTwo

    def getEnemy(self, index: int):
        if index == 1:
            return self.playerOne
        return self.playerTwo

    def isSuddenDeath(self):
        if self.time > GAME_SUDDEN_DEATH_TIME:
            return True
        else:
            return False

    def think(self, delta: int):
        self.playerOne.thinkAI()
        self.playerTwo.thinkAI()

        self.playerOne.think(delta)
        self.playerTwo.think(delta)

        self.thinkBlocks(delta)

        self.time += delta

    def thinkBlocks(self, delta: int):
        for index, value in enumerate(self.blocks):
            block = self.blocks[index]
            block.think(delta)

        indices_to_delete = [i for i, value in enumerate(self.blocks) if not value.visible]
        self.blocks = np.delete(self.blocks, indices_to_delete, axis=0)

    def isLevelOver(self):
        if self.playerTwo.visible and self.playerOne.visible:
            return False
        return True