from game.level import Level

level = Level()

def entry_point():
    init_game()

def init_game():
    level.init()

def isGameOver():
    return level.isLevelOver()

def startGame():
    level.init()

def changePlayers():
    return level.changePlayer()

def changeAI():
    level.playerAIChose += 1
    if level.playerAIChose > 3:
        level.playerAIChose = 0


def step():
    level.think(10)

def setPlayerAction(index, vecX):
    if index == 0:
        level.getPlayerOne().vecX = vecX
    else:
        level.getPlayerTwo().vecX = vecX

def getLevel():
    return level

def getPlayer(index: int):
    if index == 0:
        return level.playerOne
    return level.playerTwo

def winnerName():
    if isGameOver():
        if not level.getPlayerOne().visible:
            return level.getPlayerTwo().getName()
        if not level.getPlayerTwo().visible:
            return level.getPlayerOne().getName()
        return "Draw"
    return None