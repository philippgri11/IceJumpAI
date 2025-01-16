import time

from typing import Optional

import numpy as np
import gymnasium as gym

import game.main as myMain

import pygame
import math

from game.constants import GAME_WIDTH, GAME_HEIGHT, WATER_HEIGHT


class IceJumpEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    MAX_BLOCKS = 3
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
        self.waveX = 0
        self.waveY = 0
        # Beobachtungsraum:
        # Spieler: 8 Werte
        # Blocks: 40 * 3 = 120 Werte
        # Goodies: 5 * 3 = 15 Werte
        # Birds: 5 * 2 = 10 Werte
        # Gesamt: 10 + 120 + 15 + 10 = 155 Werte
        #self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(17,), dtype=np.float32)

        self.done = False
        self.last_y = None
        self.time_step = 0
        self.sumVec = 0

        self.block_surface = self.create_rounded_rect_surface(30, 30, (255, 255, 255, 128), (255, 255, 255), 3, False)
        self.playerOne_surface = self.create_rounded_rect_surface(30, 30, (255, 0, 0), (0, 0, 0), 3, True)
        self.playerTwo_surface = self.create_rounded_rect_surface(30, 30, (0, 255, 0), (0, 0, 0), 3, True)
        self.background = self.draw_vertical_gradient(GAME_WIDTH, GAME_HEIGHT, (100, 150, 200), (160, 225, 254))
        self.waveSurface = self.create_wave_surface(GAME_WIDTH + 100, WATER_HEIGHT + 32, (0, 182, 221))
        self.waveSurfaceBack = self.create_wave_surface(GAME_WIDTH + 130, WATER_HEIGHT + 32, (40, 208, 254))

    def setupWindow(self):
        # Initialisierung
        pygame.init()
        self.font = pygame.font.Font(None, 25)

        # Bildschirm erstellen
        self.screen = pygame.display.set_mode((GAME_WIDTH, GAME_HEIGHT))

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

        addConstantX = self._normalize_pos_x(5)
        #blocks = myMain.getLevel().getBlocks()
        for i in range(8, 8 + 3 * self.MAX_BLOCKS, 3):
            bx = obs[i]
            by = obs[i+1]
            # ist der Spieler über einem Eisblock?
            if (obs[0] + width - addConstantX > bx) and (obs[0] + addConstantX <= bx + width) and (obs[1] + height/2 < by) :
                found = True
                break


        # Schrittweise Belohnung für Zeit
        reward += 0.1 * self.time_step

        if myPlayer.x < 50 or myPlayer.x + myPlayer.width >= GAME_WIDTH - 50:
            if myPlayer.x < 50:
                reward -= 50 * abs(myPlayer.x - 50)
            else:
                reward -= 50 * abs(-myPlayer.x - myPlayer.width + GAME_WIDTH)

        # Belohnung, wenn über dem Gegner oder Bestrafung, wenn unter dem Gegner
        if myPlayer.x + myPlayer.width > myEnemy.x and myPlayer.x <= myEnemy.x + myPlayer.width:
            if myPlayer.y + myPlayer.height < myEnemy.y:
                reward += 20000 - abs(myPlayer.x - myEnemy.x) * 500

            if myPlayer.y > myEnemy.y + myEnemy.height:
                reward -= 5000
        else:
            # Bestrafung für Entfernung vom Gegner
            distance = abs(myPlayer.x - myEnemy.x)
            reward -= min(distance * 0.05, 5.0)  # Proportionale Strafe, maximal 5

            # Wenn er sinkt springt, soll er auf einen Eisblock
            if myPlayer.vecY >= 0:
                # Belohnung für Stabilität (auf Eisblock bleiben)
                if found:
                    reward += 200
                else:
                    reward -= 5000  # Sofortige Strafe bei gefährlichem Verhalten
            # Wenn er nach oben springt, soll er sich dem Gegner annähern
            else:
                # Belohnung für Annäherung an den Gegner
                if vec_x > 0 and myPlayer.x < myEnemy.x:
                    reward += 10
                elif vec_x < 0 and myPlayer.x > myEnemy.x:
                    reward += 10

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

    # Abgerundeten Eisblock vorzeichnen
    def create_rounded_rect_surface(self, width, height, color, colorBorder, radius, eyes):
        surface = pygame.Surface((width, height), pygame.SRCALPHA)  # Transparente Surface
        pygame.draw.rect(surface, color, (0, 0, width, height), border_radius=radius)
        pygame.draw.rect(surface, colorBorder, (0, 0, width, height), width=1, border_radius=radius)

        if eyes is True:
            pygame.draw.rect(surface, colorBorder, (width/2 - 4, height/2 - 6, 3, 9), border_radius=radius)
            pygame.draw.rect(surface, colorBorder, (width/2 + 1, height/2 - 6, 3, 9), border_radius=radius)

        return surface


    def draw_vertical_gradient(self, width, height, color_top, color_bottom):
        """Zeichnet einen vertikalen Farbverlauf von `color_top` nach `color_bottom`."""
        gradient_surface = pygame.Surface((width, height), pygame.SRCALPHA)

        # Schrittweise Farbmischung
        for y in range(height):
            # Interpolation der Farben
            r = color_top[0] + (color_bottom[0] - color_top[0]) * y // height
            g = color_top[1] + (color_bottom[1] - color_top[1]) * y // height
            b = color_top[2] + (color_bottom[2] - color_top[2]) * y // height

            pygame.draw.line(gradient_surface, (r, g, b), (0, y), (width, y))
        return gradient_surface

    def create_wave_surface(self, wave_width, wave_height, color, wave_amplitude=6):
        """
        Erstellt eine Surface mit einer Sinuswelle.
        """
        wave_surface = pygame.Surface((wave_width, wave_height), pygame.SRCALPHA)
        points = []

        # Berechnung der Wellenpunkte
        for i in range(wave_width):
            y = int(math.sin(math.radians(i << 3)) * wave_amplitude) + 20
            points.append((i, y))

        # Schließen der Fläche
        points.append((wave_width, wave_height))  # Unten rechts
        points.append((0, wave_height))           # Unten links

        # Zeichne die Welle auf die Surface
        pygame.draw.polygon(wave_surface, color, points)
        return wave_surface

    def render(self, mode='human'):
        try:
            # Hintergrund mit Farbverlauf zeichnen
            self.screen.blit(self.background, (0, 0))
            #self.draw_vertical_gradient(self.screen, (100, 150, 200), (160, 225, 254))

            self.screen.blit(self.waveSurfaceBack, (self.waveX*3/4 - 90, GAME_HEIGHT - WATER_HEIGHT - 37))

            for index, value in enumerate(self.entry_point.level.getBlocks()):
                self.screen.blit(self.block_surface, (value.x, value.y))

            playerOne = self.entry_point.level.playerOne
            self.screen.blit(self.playerOne_surface, (playerOne.x, playerOne.y))

            playerTwo = self.entry_point.level.playerTwo
            self.screen.blit(self.playerTwo_surface, (playerTwo.x, playerTwo.y))

            text = "Zeit: "+str(self.entry_point.level.time/1000)+" s"
            text_surface = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(text_surface, (GAME_WIDTH // 2 - 60, 10))

            text = "Spieler 1: "+str(self.entry_point.level.playerOne.getName())+""
            text_surface = self.font.render(text, True, (128, 0, 0))
            self.screen.blit(text_surface, (10, 10))

            text = "Spieler 2: "+str(self.entry_point.level.playerTwo.getName())+""
            text_surface = self.font.render(text, True, (0, 128, 0))
            self.screen.blit(text_surface, (GAME_WIDTH - 10 - text_surface.get_width(), 10))

            self.waveX += 0.1
            if self.waveX >= 90:
                self.waveX = 0

            self.screen.blit(self.waveSurface, (-self.waveX, GAME_HEIGHT - WATER_HEIGHT - 32))

            #pygame.draw.rect(self.screen, (0, 182, 221), (0, GAME_HEIGHT - WATER_HEIGHT, GAME_WIDTH, WATER_HEIGHT))  # Rechteck zeichnen

            time.sleep(0.005)
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

        obs_blocks = []
        sorted_blocks = sorted(self.entry_point.level.getBlocks(), key=lambda block: abs(block.x - aiPlayer.x))
        for i, block in enumerate(sorted_blocks[:self.MAX_BLOCKS]):
            bx = self._normalize_pos_x(block.x)
            by = self._normalize_pos_y(block.y)
            hits = block.hits
            obs_blocks.extend([bx, by, hits])
        # Auffüllen mit Nullen
        while len(obs_blocks) < self.MAX_BLOCKS * 3:
            obs_blocks.append(-1.0)

        obs_players = np.array([x0, y0, x1, y1, vecX0, vecY0, vecX1, vecY1], dtype=np.float32)

        return np.concatenate([obs_players,
                               np.array(obs_blocks[:9], dtype=np.float32)])