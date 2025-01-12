import time

from typing import Optional

import numpy as np
import gymnasium as gym

import game.main as myMain

import pygame

class IceJumpEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    MAX_BLOCKS = 40
    MAX_GOODIES = 5
    MAX_BIRDS = 5

    # Annahmen für Normalisierung
    GAME_WIDTH = 640.0
    GAME_HEIGHT = 480.0
    MAX_HITS = 3.0
    MAX_GOODIE_TYPE = 5.0  # Angenommen Goodie-Typen: 0 bis 5

    def __init__(self, player_index=0, render_mode=None):
        super(IceJumpEnv, self).__init__()
        # Py4J Gateway
        #self.gateway = JavaGateway()
        self.entry_point = myMain
        self.entry_point.entry_point()

        # Render-Modus speichern
        self.render_mode = render_mode

        self.player_index = player_index

        # Aktionen: float-Wert für die horizontale Geschwindigkeit
        self.action_space = gym.spaces.Box(low=-0.16, high=0.16, shape=(1,), dtype=np.float32)
        #self.action_space = gym.spaces.Discrete(3, start=0, seed=42)

        self.won = 0
        self.lose = 0
        # Beobachtungsraum:
        # Spieler: 10 Werte
        # Blocks: 40 * 3 = 120 Werte
        # Goodies: 5 * 3 = 15 Werte
        # Birds: 5 * 2 = 10 Werte
        # Gesamt: 10 + 120 + 15 + 10 = 155 Werte
        #self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32)

        self.done = False
        self.last_y = None
        self.time_step = 0
        self.sumVec = 0

    def setupWindow(self):
        # Initialisierung
        pygame.init()
        self.font = pygame.font.Font(None, 20)

        # Bildschirm erstellen
        self.screen = pygame.display.set_mode((640, 480))

    def reset(self, seed: Optional[int] = None):
        super().reset(seed=seed)

        self.entry_point.startGame()

        self.player_index = self.entry_point.changePlayers()
        if self.player_index > 1:
            self.player_index = 0
        self.done = False
        self.time_step = 0
        obs = self._get_obs()
        self.last_y = obs[1]  # y-Koordinate des eigenen Spielers
        info = {}
        self.sumVec = 0
        #print("reset")
        return obs, info

    def step(self, action):
        reward = 0

        # Aktion durchführen
        vec_x = float(action[0])
        #vec_x = float(action-1) * 0.16
        #print(vec_x, action)
        self.sumVec += vec_x
        self.entry_point.setPlayerAction(self.player_index, vec_x)
        self.time_step += 1

        # Einen Schritt simulieren
        self.entry_point.step()

        # Aktuellen Zustand abfragen
        obs = self._get_obs()
        self.done = self.entry_point.isGameOver()

        winner = myMain.winnerName()

        enemyIndex = 1
        if self.player_index == 1:
            enemyIndex = 0
        myPlayer = myMain.getPlayer(self.player_index)
        myEnemy = myMain.getPlayer(enemyIndex)
        width = self._normalize_pos_x(myPlayer.width)
        height = width

        found = False

        bwidth = 0
        blocks = myMain.getLevel().getBlocks()
        for i, block in enumerate(blocks[:self.MAX_BLOCKS]):
            bwidth = self._normalize_pos_x(block.width)
            break
        # ist der Spieler über dem nächsten Eisblock?
        if (obs[0] + width > obs[4]) and (obs[0] <= obs[4] + bwidth) and (obs[1] + height/2 < obs[5]) :
            found = True

        # Schrittweise Belohnung für Zeit
        reward += 0.1 * self.time_step

        # Belohnung für Stabilität (auf Eisblock bleiben)
        if found:
            reward += 10
        else:
            reward -= 10  # Sofortige Strafe bei gefährlichem Verhalten

        # Wenn er hoch springt, soll er in Richtung Gegner gehen
        if myPlayer.vecY < 0:
            # Bestrafung für Entfernung vom Gegner
            distance = abs(myPlayer.x - myEnemy.x)
            reward -= min(distance * 0.05, 5.0)  # Proportionale Strafe, maximal 5

        # Belohnung für Annäherung an den Gegner
        if vec_x > 0 and myPlayer.x < myEnemy.x:
            reward += 1
        elif vec_x < 0 and myPlayer.x > myEnemy.x:
            reward += 1

        # Belohnung, wenn über dem Gegner oder Bestrafung, wenn unter dem Gegner
        if myPlayer.x + myPlayer.width > myEnemy.x and myPlayer.x <= myEnemy.x + myPlayer.width:
            if myPlayer.y + myPlayer.height < myEnemy.y:
                reward += 100

            if myPlayer.y > myEnemy.y + myEnemy.height:
                reward -= 100

        # Auswertung, wenn jemand gewonnen hat
        if self.done and winner is not None:
            player_name = myPlayer.getName()
            if winner == player_name:
                self.won += 1
                ratio = str(self.won) + " / " + str(self.lose)
                reward += 20000
                print("OMG OMG, gewonnen gegen ", myEnemy.name, self.time_step, int(self.sumVec), int(myPlayer.x), int(myEnemy.x), ratio)
            else:
                self.lose += 1
                ratio = str(self.won) + " / " + str(self.lose)
                print("Verloren ", winner, self.player_index, self.time_step, int(self.sumVec), int(myPlayer.x), int(myEnemy.x), ratio)
                reward -= 5000  # Konstante Strafe bei Verlust

        info = {}
        #print("reward ", reward, found, self.time_step, vec_x, obs)
        #print("Step ", vec_x, action, reward, info)
        return obs, reward, self.done, False, info

    def render(self, mode='human'):
        try:
            self.screen.fill((0, 0, 0))

            for index, value in enumerate(self.entry_point.level.getBlocks()):
                block_surface = pygame.Surface((value.width, value.height), pygame.SRCALPHA)  # Transparente Oberfläche
                block_surface.fill((255, 255, 255, 128))  # white mit Alpha = 128
                self.screen.blit(block_surface, (value.x, value.y))

                pygame.draw.rect(self.screen, (255, 255, 255), (value.x, value.y, value.width, value.height), width=2)

            playerOne = self.entry_point.level.playerOne
            pygame.draw.rect(self.screen, (255, 0, 0), (playerOne.x, playerOne.y, playerOne.width, playerOne.height))

            playerTwo = self.entry_point.level.playerTwo
            pygame.draw.rect(self.screen, (0, 255, 0), (playerTwo.x, playerTwo.y, playerTwo.width, playerTwo.height))

            text = "Zeit: "+str(self.entry_point.level.time/1000)+" s"
            text_surface = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(text_surface, (290, 50))

            time.sleep(0.01)
        except pygame.error as e:
            # Behandle spezifische Pygame-Fehler
            print(f"Pygame-Fehler: {e}")
        except Exception as e:
            # Behandle allgemeine Fehler
            print(f"Ein Fehler ist aufgetreten: {e}")
        finally:
            # Bildschirm aktualisieren
            pygame.display.flip()

    def close(self):
        pygame.quit()

    def _normalize_pos_x(self, x):
        return x / self.GAME_WIDTH

    def _normalize_pos_y(self, y):
        return y / self.GAME_HEIGHT

    def _normalize_hits(self, hits):
        return hits / self.MAX_HITS

    def _normalize_goodie_type(self, gtype):
        return gtype / self.MAX_GOODIE_TYPE

    def _get_obs(self):
        aiPlayer = self.entry_point.level.getPlayerOne()
        enemyPlayer = self.entry_point.level.getPlayerTwo()
        if self.player_index == 1:
            aiPlayer = self.entry_point.level.getPlayerTwo()
            enemyPlayer = self.entry_point.level.getPlayerOne()

        width = self._normalize_pos_x(aiPlayer.width)

        x0 = self._normalize_pos_x(aiPlayer.x)
        y0 = self._normalize_pos_y(aiPlayer.y)
        vecX0 = self._normalize_pos_y(aiPlayer.vecX)
        vecY0 = self._normalize_pos_y(aiPlayer.vecY)
        x1 = self._normalize_pos_x(enemyPlayer.x)
        y1 = self._normalize_pos_y(enemyPlayer.y)
        vecX1 = self._normalize_pos_y(enemyPlayer.vecX)
        vecY1 = self._normalize_pos_y(enemyPlayer.vecY)

        index = -1
        for i, block in enumerate(self.entry_point.level.getBlocks()):
            bx = self._normalize_pos_x(block.x)
            bwidth = self._normalize_pos_x(block.width)

            if (index < 0) or ((x0 + width > bx) and (x0 < bx + bwidth)) :
                if index < 0 or (abs(self._normalize_pos_x(self.entry_point.level.getBlocks()[index].x) - x0) > abs(bx - x0)):
                    index = i

        bx = -1
        by = -1
        bhit = -1
        if index >= 0:
            bx = self._normalize_pos_x(self.entry_point.level.getBlocks()[index].x)
            by = self._normalize_pos_y(self.entry_point.level.getBlocks()[index].y)
            bhit = self._normalize_pos_y(self.entry_point.level.getBlocks()[index].hits)
        obs = np.array([x0, y0, x1, y1, bx, by, bhit, vecX0, vecY0, vecX1, vecY1], dtype=np.float32)
        return obs