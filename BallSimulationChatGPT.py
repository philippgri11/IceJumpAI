#!/usr/bin/env python3
import pygame
import math
import sys
import random

# --- Parameter und Konstanten ---
WIDTH, HEIGHT = 800, 600
FPS = 60

# Parameter für das Polygon (Container)
NUM_SIDES = 15         # z. B. 10 für ein 10‑Eck
ROTATION_SPEED = 1.5   # Radiant pro Sekunde

# Parameter für die Bälle
# Hier kann man die Größe (Radius) der Bälle einstellen – die Masse wird als radius^2 angenommen.
BALL1_RADIUS = 13
BALL2_RADIUS = 15

BALL_COLOR1 = (173, 216, 230)  # Hellblau
BALL_COLOR2 = (0, 255, 255)    # Cyan (zweiter Ball)

# Anfangspositionen (leicht versetzt) und -geschwindigkeiten (verschiedene Richtungen)
ball1_pos = pygame.Vector2(WIDTH // 2 - 50, HEIGHT // 2 - 100)
ball2_pos = pygame.Vector2(WIDTH // 2 + 50, HEIGHT // 2 - 100)
ball1_vel = pygame.Vector2(150, 0)
ball2_vel = pygame.Vector2(-150, 0)

# Berechne die Masse (als Näherung: Masse ~ radius^2)
ball1_mass = BALL1_RADIUS ** 2
ball2_mass = BALL2_RADIUS ** 2

# Punktestand
score1 = 0
score2 = 0
WIN_SCORE = 10

# Physikalische Konstanten
GRAVITY = pygame.Vector2(0, 500)  # Pixel pro Sekunde^2
FRICTION = 0.99                 # Dämpfung (einfache Reibung)
RESTITUTION = 0.9               # Elastizitätskoeffizient beim Aufprall (für Wandkollisionen)

# Parameter für das Polygon (Container)
POLY_CENTER = pygame.Vector2(WIDTH // 2, HEIGHT // 2)
POLY_RADIUS = 250  # Abstand vom Mittelpunkt zu den Eckpunkten

# Index der aktuell rot markierten Kante (0 <= red_edge_index < NUM_SIDES)
red_edge_index = 0

# Flags, um Mehrfachzählungen bei fortdauerndem Kontakt zu vermeiden
ball1_scored = False
ball2_scored = False

# --- Funktionen ---

def get_polygon_vertices(n_sides, center, radius, rotation):
    """
    Berechnet die Eckpunkte eines n-seitigen Vielecks.
    :param n_sides: Anzahl der Seiten
    :param center: pygame.Vector2, Mittelpunkt
    :param radius: Abstand vom Mittelpunkt zu den Eckpunkten
    :param rotation: Rotation in Radiant
    :return: Liste von pygame.Vector2
    """
    vertices = []
    start_angle = rotation - math.pi / 2  # erster Punkt oben
    for i in range(n_sides):
        angle = start_angle + i * 2 * math.pi / n_sides
        x = center.x + radius * math.cos(angle)
        y = center.y + radius * math.sin(angle)
        vertices.append(pygame.Vector2(x, y))
    return vertices

def closest_point_on_segment(p, a, b):
    """
    Berechnet den Punkt auf der Strecke a-b, der am nächsten zu p liegt.
    :param p: pygame.Vector2 (z. B. Ballmittelpunkt)
    :param a: pygame.Vector2, Startpunkt der Strecke
    :param b: pygame.Vector2, Endpunkt der Strecke
    :return: pygame.Vector2, nächster Punkt auf der Strecke
    """
    ab = b - a
    if ab.length_squared() == 0:
        return a
    t = (p - a).dot(ab) / ab.length_squared()
    t = max(0, min(1, t))
    return a + t * ab

def wall_velocity_at_point(point, center, angular_velocity):
    """
    Berechnet die Geschwindigkeit eines rotierenden Punkts (auf dem Polygon)
    am gegebenen Punkt.
    :param point: pygame.Vector2, Punkt auf dem Polygon
    :param center: pygame.Vector2, Drehzentrum
    :param angular_velocity: Drehgeschwindigkeit (Radiant pro Sekunde)
    :return: pygame.Vector2, Geschwindigkeit des Punktes
    """
    r = point - center
    return angular_velocity * pygame.Vector2(-r.y, r.x)

def handle_polygon_collision(ball_pos, ball_vel, ball_radius, vertices):
    """
    Prüft die Kollision eines Balls (Position und Geschwindigkeit) mit allen Polygonkanten.
    Wird bei Kollision die Position korrigiert und die Geschwindigkeit entsprechend reflektiert.
    :return: (neue_position, neue_velocity, collided_with_red_edge: bool)
    """
    collided_with_red = False
    for i in range(len(vertices)):
        a = vertices[i]
        b = vertices[(i + 1) % len(vertices)]
        closest = closest_point_on_segment(ball_pos, a, b)
        dist = (ball_pos - closest).length()
        if dist < ball_radius:
            # Kollisionsnormal bestimmen
            if dist != 0:
                normal = (ball_pos - closest).normalize()
            else:
                edge_dir = (b - a).normalize()
                normal = pygame.Vector2(-edge_dir.y, edge_dir.x)
            penetration = ball_radius - dist
            ball_pos += normal * penetration

            wall_vel = wall_velocity_at_point(closest, POLY_CENTER, ROTATION_SPEED)
            rel_vel = ball_vel - wall_vel
            vn = rel_vel.dot(normal)
            if vn < 0:
                rel_vel = rel_vel - (1 + RESTITUTION) * vn * normal
                ball_vel = rel_vel + wall_vel

            global red_edge_index
            if i == red_edge_index:
                collided_with_red = True
    return ball_pos, ball_vel, collided_with_red

def handle_ball_collision(pos1, vel1, mass1, radius1, pos2, vel2, mass2, radius2):
    """
    Prüft und behandelt die Kollision zwischen zwei Bällen mit unterschiedlichen Massen.
    Es erfolgt eine elastische Kollision unter Berücksichtigung des Impulserhaltungssatzes.
    :return: (neue_vel1, neue_vel2, neue_pos1, neue_pos2)
    """
    delta = pos1 - pos2
    dist = delta.length()
    min_dist = radius1 + radius2
    if dist == 0:
        # Vermeidung von Division durch 0
        return vel1, vel2, pos1, pos2
    if dist < min_dist:
        # Korrigiere die Überschneidung (positionsbasierte Korrektur nach inversen Massen)
        overlap = min_dist - dist
        inv_mass1 = 1 / mass1
        inv_mass2 = 1 / mass2
        inv_mass_sum = inv_mass1 + inv_mass2
        correction = delta.normalize() * overlap
        pos1 += correction * (inv_mass1 / inv_mass_sum)
        pos2 -= correction * (inv_mass2 / inv_mass_sum)

        # Berechne den Normalvektor
        normal = (pos1 - pos2).normalize()
        rel_vel = vel1 - vel2
        vel_along_normal = rel_vel.dot(normal)
        if vel_along_normal < 0:
            # Berechne den Impuls (e = 1 für vollkommen elastisch)
            impulse_mag = -2 * vel_along_normal / (inv_mass1 + inv_mass2)
            impulse = impulse_mag * normal
            vel1 = vel1 + impulse * inv_mass1
            vel2 = vel2 - impulse * inv_mass2
    return vel1, vel2, pos1, pos2

# --- Hauptprogramm ---
def main():
    global red_edge_index, score1, score2, ball1_scored, ball2_scored
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("2 Bälle im rotierenden Vieleck (mit Gewicht)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 36)

    poly_rotation = 0  # Aktuelle Rotation des Polygons

    running = True
    winner = None
    while running:
        dt = clock.tick(FPS) / 1000.0

        # --- Ereignisbehandlung ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Polygonrotation aktualisieren ---
        poly_rotation += ROTATION_SPEED * dt
        vertices = get_polygon_vertices(NUM_SIDES, POLY_CENTER, POLY_RADIUS, poly_rotation)

        # --- Physik-Update der Bälle ---
        # Ball 1
        global ball1_pos, ball1_vel, ball2_pos, ball2_vel, ball1_mass, ball2_mass
        ball1_vel += GRAVITY * dt
        ball1_vel *= FRICTION ** dt
        ball1_pos += ball1_vel * dt

        # Ball 2
        ball2_vel += GRAVITY * dt
        ball2_vel *= FRICTION ** dt
        ball2_pos += ball2_vel * dt

        # --- Kollision mit den Polygonkanten ---
        ball1_pos, ball1_vel, collided_red1 = handle_polygon_collision(ball1_pos, ball1_vel, BALL1_RADIUS, vertices)
        ball2_pos, ball2_vel, collided_red2 = handle_polygon_collision(ball2_pos, ball2_vel, BALL2_RADIUS, vertices)

        # Punktevergabe, falls ein Ball die rot markierte Kante berührt
        if collided_red1:
            if not ball1_scored:
                score1 += 1
                ball1_scored = True
                red_edge_index = random.randrange(NUM_SIDES)
        else:
            ball1_scored = False

        if collided_red2:
            if not ball2_scored:
                score2 += 1
                ball2_scored = True
                red_edge_index = random.randrange(NUM_SIDES)
        else:
            ball2_scored = False

        # --- Ball-Ball-Kollision unter Berücksichtigung der unterschiedlichen Massen ---
        ball1_vel, ball2_vel, ball1_pos, ball2_pos = handle_ball_collision(
            ball1_pos, ball1_vel, ball1_mass, BALL1_RADIUS,
            ball2_pos, ball2_vel, ball2_mass, BALL2_RADIUS
        )

        # --- Gewinnerprüfung ---
        if score1 >= WIN_SCORE:
            winner = "Ball 1"
            running = False
        elif score2 >= WIN_SCORE:
            winner = "Ball 2"
            running = False

        # --- Zeichnen ---
        screen.fill((0, 0, 0))  # schwarzer Hintergrund

        # Zeichne das Polygon: Alle Kanten weiß, außer der rot markierten Kante
        for i in range(len(vertices)):
            start = vertices[i]
            end = vertices[(i + 1) % len(vertices)]
            color = (255, 0, 0) if i == red_edge_index else (255, 255, 255)
            pygame.draw.line(screen, color, (int(start.x), int(start.y)), (int(end.x), int(end.y)), 2)

        # Zeichne die Bälle
        pygame.draw.circle(screen, BALL_COLOR1, (int(ball1_pos.x), int(ball1_pos.y)), BALL1_RADIUS)
        pygame.draw.circle(screen, BALL_COLOR2, (int(ball2_pos.x), int(ball2_pos.y)), BALL2_RADIUS)

        # Zeichne die Punktestände
        score_text = font.render(f"Ball 1: {score1}   Ball 2: {score2}", True, (255, 255, 255))
        screen.blit(score_text, (20, 20))

        pygame.display.flip()

    # Spielende: Gewinner anzeigen
    end_text = font.render(f"{winner} gewinnt!", True, (255, 255, 0))
    screen.fill((0, 0, 0))
    screen.blit(end_text, (WIDTH // 2 - end_text.get_width() // 2, HEIGHT // 2 - end_text.get_height() // 2))
    pygame.display.flip()
    pygame.time.wait(3000)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
