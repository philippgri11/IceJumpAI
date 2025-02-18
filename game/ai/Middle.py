from dataclasses import dataclass

from game.ai.ai import AI

from game.constants import PLAYER_MAX_VEC_X
import game_state

@dataclass
class Middle(AI):

    name = "Middle"

    def think(self, index: int):
        player = game_state.level_instance.getPlayerOne()
        enemy =  game_state.level_instance.getPlayerTwo()
        if index != 0:
            player = game_state.level_instance.getPlayerTwo()
            enemy = game_state.level_instance.getPlayerOne()

        difference: float = player.x - enemy.x

        if player.x + player.width > enemy.x and player.x < enemy.x + enemy.width:
            if player.y + player.height/2 < enemy.y:
                if difference > 3:
                    self.vecX = -PLAYER_MAX_VEC_X*3/4
                elif difference < -3:
                    self.vecX = PLAYER_MAX_VEC_X*3/4
                else:
                    self.vecX = 0
                return

        if player.vecY < 0.1:
            if player.x + player.width - 5 > enemy.x and player.x + 5 < enemy.x + enemy.width:
                if player.y < enemy.y:
                    if player.x - enemy.x > 4.0:
                        self.vecX = -PLAYER_MAX_VEC_X*3/4
                        return

                    if player.x - enemy.x < -4.0:
                        self.vecX = PLAYER_MAX_VEC_X*3/4
                        return
                else:
                    if player.x < enemy.x:
                        self.vecX = -PLAYER_MAX_VEC_X
                    else:
                        self.vecX = PLAYER_MAX_VEC_X
                    return
            elif player.x - enemy.x > 4.0:
                self.vecX = -PLAYER_MAX_VEC_X
                return

            elif player.x - enemy.x < -4.0:
                self.vecX = PLAYER_MAX_VEC_X
                return

        # falls es noch einen Eisblock gibt, dann
        if len(game_state.level_instance.getBlocks()) > 0:
            block = self.getMinBlock(player)
            # gib die Differenz zwischen beiden Entit�ten
            difference: float = player.x + player.width/2 - block.x - block.width/2

            # falls die Differenz größer als 3 ist, dann bewege den Spieler mit der Hälfte der Maximaleschwindigkeit nach links, falls Dif kleiner als -3 dann rechts, ansonsten beweg dich nicht
            if difference > 3:
                self.vecX = -PLAYER_MAX_VEC_X
            elif difference < -3:
                self.vecX = PLAYER_MAX_VEC_X
            else:
                self.vecX = 0

    def getMinBlock(self, player):
        result: int = -1
        for index, value in enumerate(game_state.level_instance.getBlocks()):
            if result < 0 or abs(value.x - player.x) < abs(game_state.level_instance.getBlocks()[result].x - player.x):
                result = index
        return game_state.level_instance.getBlocks()[result]