from dataclasses import dataclass

from game.ai.ai import AI

from game.constants import PLAYER_MAX_VEC_X
import game_state

@dataclass
class Easy(AI):

    def think(self, index: int):
        player = game_state.level_instance.getPlayerOne()
        if index != 0: player = game_state.level_instance.getPlayerTwo()

        # falls es noch einen Eisblock gibt, dann
        if len(game_state.level_instance.getBlocks()) > 0:
            block = self.getMinBlock(player)
            # gib die Differenz zwischen beiden Entit�ten
            difference: float = player.x + player.width/2 - block.x - block.width/2

            # falls die Differenz größer als 3 ist, dann bewege den Spieler mit der Hälfte der Maximaleschwindigkeit nach links, falls Dif kleiner als -3 dann rechts, ansonsten beweg dich nicht
            if difference > 3:
                self.vecX = -PLAYER_MAX_VEC_X/2
            elif difference < -3:
                self.vecX = PLAYER_MAX_VEC_X/2
            else:
                self.vecX = 0

    def getMinBlock(self, player):
        result: int = -1
        for index, value in enumerate(game_state.level_instance.getBlocks()):
            if result < 0 or abs(value.x - player.x) < abs(game_state.level_instance.getBlocks()[result].x - player.x):
                result = index
        return game_state.level_instance.getBlocks()[result]