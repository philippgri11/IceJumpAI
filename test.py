from game.level import Level

import game.main as myMain


level = myMain.entry_point()

index = 0  # Initialisiere den Zähler

while index < 10000:
    # Deine Logik hier
    #print(f"Durchlauf: {index + 1}")
    if myMain.isGameOver():
        break
    myMain.step()

    # Zähler erhöhen
    index += 1

print("Level vorbei nach x Schritten: ", index, myMain.level.playerOne, myMain.level.playerTwo)