"""
view_arena.py — Visualize the homing agent environment.

Run with:
    python view_arena.py             # static mode (default)
    python view_arena.py moving      # moving mode (threat interception)

Controls:
  W / UP      = Thrust forward
  A / LEFT    = Rotate counter-clockwise
  D / RIGHT   = Rotate clockwise
  R           = Reset episode (new random positions)
  SPACE       = Toggle between manual / random-agent mode
  M           = Toggle between static / moving target mode
  ESC         = Quit
"""

import sys
import math
import pygame
import numpy as np

sys.path.insert(0, ".")

import env  # noqa: F401
import gymnasium as gym
from config import Config


# ═══════════════════════════════════════════════════════
#  COLORS
# ═══════════════════════════════════════════════════════

BG_COLOR         = (12, 12, 20)
GRID_COLOR       = (25, 25, 40)
BORDER_COLOR     = (40, 40, 70)

# Agent (interceptor) — cool blue
AGENT_COLOR      = (100, 200, 255)
AGENT_OUTLINE    = (150, 220, 255)
AGENT_THRUST     = (255, 140, 40)
AGENT_FLAME_CORE = (255, 230, 150)
AGENT_TRAIL      = (60, 120, 180)

# Static target — pinkish-red
TARGET_COLOR     = (255, 80, 100)
TARGET_RING      = (255, 120, 140)

# Threat rocket — angry red-orange
THREAT_COLOR     = (255, 70, 50)
THREAT_OUTLINE   = (255, 120, 100)
THREAT_TRAIL     = (180, 60, 40)

# Protected target — green shield
PROTECTED_COLOR  = (80, 255, 120)
PROTECTED_DIM    = (40, 150, 70)

# UI
TEXT_COLOR       = (200, 200, 220)
TEXT_DIM         = (100, 100, 130)
SUCCESS_COLOR    = (80, 255, 120)
FAIL_COLOR       = (255, 80, 80)
WARNING_COLOR    = (255, 200, 80)


# ═══════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════

WINDOW_WIDTH  = 1000
WINDOW_HEIGHT = 800
ARENA_SIZE    = 800
INFO_X        = 820
FPS           = 20
MAX_TRAIL     = 200


def world_to_screen(wx, wy):
    """Convert world coords (y-up) to screen coords (y-down)."""
    return (int(wx), int(ARENA_SIZE - wy))


# ═══════════════════════════════════════════════════════
#  DRAWING FUNCTIONS
# ═══════════════════════════════════════════════════════

def draw_rocket(surface, position, angle, color, outline_color, size=12, thrust_on=False):
    """Draw a rocket (triangle) at the given position and angle."""
    sx, sy = world_to_screen(position.x, position.y)
    render_angle = -angle  # flip for screen coords

    cos_a = math.cos(render_angle)
    sin_a = math.sin(render_angle)

    def rot(px, py):
        return (sx + int(px * cos_a - py * sin_a),
                sy + int(px * sin_a + py * cos_a))

    nose    = rot(size * 1.5, 0)
    left_w  = rot(-size, -size * 0.7)
    right_w = rot(-size,  size * 0.7)

    # Thrust flame (behind the rocket)
    if thrust_on:
        ft = rot(-size * 2.5, 0)
        fl = rot(-size * 1.2, -size * 0.3)
        fr = rot(-size * 1.2,  size * 0.3)
        pygame.draw.polygon(surface, AGENT_THRUST, [ft, fl, fr])

        it = rot(-size * 2.0, 0)
        il = rot(-size * 1.2, -size * 0.15)
        ir = rot(-size * 1.2,  size * 0.15)
        pygame.draw.polygon(surface, AGENT_FLAME_CORE, [it, il, ir])

    # Body
    pygame.draw.polygon(surface, color, [nose, left_w, right_w])
    pygame.draw.polygon(surface, outline_color, [nose, left_w, right_w], 2)


def draw_static_target(surface, position, radius, pulse_time):
    """Draw the static target as a pulsing crosshair."""
    sx, sy = world_to_screen(position.x, position.y)
    pulse = 0.5 + 0.5 * math.sin(pulse_time * 3.0)
    glow_r = int(radius * (1.3 + 0.3 * pulse))

    # Glow
    glow_surf = pygame.Surface((glow_r * 4, glow_r * 4), pygame.SRCALPHA)
    center = (glow_r * 2, glow_r * 2)
    for i in range(3):
        r = glow_r - i * 4
        alpha = int(20 + 15 * pulse) - i * 5
        pygame.draw.circle(glow_surf, (*TARGET_COLOR, max(alpha, 5)), center, r)
    surface.blit(glow_surf, (sx - glow_r * 2, sy - glow_r * 2))

    # Ring + crosshair
    pygame.draw.circle(surface, TARGET_RING, (sx, sy), int(radius), 2)
    pygame.draw.circle(surface, TARGET_COLOR, (sx, sy), 5)
    line_len = int(radius * 0.6)
    pygame.draw.line(surface, TARGET_RING, (sx - line_len, sy), (sx + line_len, sy), 1)
    pygame.draw.line(surface, TARGET_RING, (sx, sy - line_len), (sx, sy + line_len), 1)


def draw_protected_target(surface, position, radius, pulse_time):
    """Draw the protected target as a green pulsing shield."""
    sx, sy = world_to_screen(position.x, position.y)
    pulse = 0.5 + 0.5 * math.sin(pulse_time * 2.0)

    # Outer glow ring
    glow_r = int(radius * (1.2 + 0.2 * pulse))
    pygame.draw.circle(surface, PROTECTED_DIM, (sx, sy), glow_r, 2)

    # Shield shape (hexagon)
    hex_r = int(radius * 0.7)
    points = []
    for i in range(6):
        angle = math.pi / 6 + i * math.pi / 3
        px = sx + int(hex_r * math.cos(angle))
        py = sy + int(hex_r * math.sin(angle))
        points.append((px, py))
    pygame.draw.polygon(surface, PROTECTED_DIM, points, 2)
    pygame.draw.polygon(surface, (*PROTECTED_COLOR, 40), points)  # slight fill

    # Center dot
    pygame.draw.circle(surface, PROTECTED_COLOR, (sx, sy), 4)


def draw_trail(surface, trail_points, color):
    """Draw a fading trail."""
    n = len(trail_points)
    if n < 2:
        return
    for i in range(1, n):
        frac = i / n
        c = (int(color[0] * frac), int(color[1] * frac), int(color[2] * frac))
        p1 = world_to_screen(*trail_points[i - 1])
        p2 = world_to_screen(*trail_points[i])
        pygame.draw.line(surface, c, p1, p2, 1)


def draw_dashed_line(surface, start_world, end_world, color=(30, 30, 50)):
    """Draw a dashed line between two world-space points."""
    sx1, sy1 = world_to_screen(start_world.x, start_world.y)
    sx2, sy2 = world_to_screen(end_world.x, end_world.y)

    dx, dy = sx2 - sx1, sy2 - sy1
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 1:
        return

    dash, gap = 8, 6
    total = dash + gap
    for i in range(int(dist / total)):
        f1 = (i * total) / dist
        f2 = min((i * total + dash) / dist, 1.0)
        x1 = int(sx1 + dx * f1)
        y1 = int(sy1 + dy * f1)
        x2 = int(sx1 + dx * f2)
        y2 = int(sy1 + dy * f2)
        pygame.draw.line(surface, color, (x1, y1), (x2, y2), 1)


def draw_grid(surface):
    """Draw a subtle background grid."""
    for x in range(0, ARENA_SIZE + 1, 100):
        pygame.draw.line(surface, GRID_COLOR, (x, 0), (x, ARENA_SIZE), 1)
    for y in range(0, ARENA_SIZE + 1, 100):
        pygame.draw.line(surface, GRID_COLOR, (0, y), (ARENA_SIZE, y), 1)
    pygame.draw.rect(surface, BORDER_COLOR, (0, 0, ARENA_SIZE, ARENA_SIZE), 2)


def draw_info_panel(surface, font, font_small, info, control_mode,
                    target_mode, total_reward, episode):
    """Draw the info panel on the right side."""
    x = INFO_X
    y = 20

    def text(label, value, color=TEXT_COLOR, y_offset=0):
        nonlocal y
        y += y_offset
        surface.blit(font_small.render(label, True, TEXT_DIM), (x, y))
        surface.blit(font.render(str(value), True, color), (x, y + 16))
        y += 42

    # Title
    surface.blit(font.render("HOMING AGENT", True, AGENT_COLOR), (x, y))
    y += 30

    # Mode badges
    mode_label = "INTERCEPT" if target_mode == "moving" else "NAVIGATE"
    mode_color = WARNING_COLOR if target_mode == "moving" else AGENT_COLOR
    surface.blit(font_small.render(f"MODE: {mode_label}", True, mode_color), (x, y))
    y += 18

    ctrl_label = "MANUAL" if control_mode == "manual" else "RANDOM"
    ctrl_color = SUCCESS_COLOR if control_mode == "manual" else TEXT_DIM
    surface.blit(font_small.render(f"CTRL: {ctrl_label}", True, ctrl_color), (x, y))
    y += 22

    pygame.draw.line(surface, BORDER_COLOR, (x, y), (x + 160, y), 1)
    y += 10

    # Stats
    text("DISTANCE", f"{info.get('distance', 0):.1f}")
    text("SPEED", f"{info.get('speed', 0):.2f}")
    text("HEADING", f"{math.degrees(info.get('angle', 0)):.1f}\u00b0")
    text("STEP", f"{info.get('steps', 0)}")

    # Moving mode extras
    if target_mode == "moving":
        threat_d = info.get('threat_distance_to_protected', 0)
        threat_color = FAIL_COLOR if threat_d < 100 else WARNING_COLOR if threat_d < 200 else TEXT_COLOR
        text("THREAT DIST", f"{threat_d:.1f}", color=threat_color)

    pygame.draw.line(surface, BORDER_COLOR, (x, y), (x + 160, y), 1)
    y += 10

    text("EPISODE", f"{episode}")
    rew_color = SUCCESS_COLOR if total_reward > 0 else FAIL_COLOR
    text("REWARD", f"{total_reward:+.1f}", color=rew_color)

    # Controls
    y = WINDOW_HEIGHT - 200
    pygame.draw.line(surface, BORDER_COLOR, (x, y), (x + 160, y), 1)
    y += 8
    for line in [
        "W/UP     Thrust",
        "A/LEFT   Rotate L",
        "D/RIGHT  Rotate R",
        "R        Reset",
        "SPACE    Ctrl mode",
        "M        Target mode",
        "ESC      Quit",
    ]:
        surface.blit(font_small.render(line, True, TEXT_DIM), (x, y))
        y += 17


def draw_banner(surface, font, text, color):
    """Draw a centered status banner."""
    text_surf = font.render(text, True, color)
    rect = text_surf.get_rect(center=(ARENA_SIZE // 2, ARENA_SIZE // 2))
    bg = pygame.Surface((rect.width + 40, rect.height + 20), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 180))
    surface.blit(bg, (rect.x - 20, rect.y - 10))
    surface.blit(text_surf, rect)


# ═══════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════

def make_env(target_mode):
    """Create a new environment with the given target mode."""
    config = Config(TARGET_MODE=target_mode)
    return gym.make("HomingAgent-v0", config=config), config


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Homing Agent \u2014 Arena Viewer")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("Consolas", 16, bold=True)
    font_small = pygame.font.SysFont("Consolas", 12)
    font_banner = pygame.font.SysFont("Consolas", 28, bold=True)

    # Parse command-line arg for initial mode
    target_mode = sys.argv[1] if len(sys.argv) > 1 else "static"
    environment, config = make_env(target_mode)
    obs, info = environment.reset(seed=42)

    control_mode = "manual"
    agent_trail = []
    threat_trail = []
    total_reward = 0.0
    episode = 1
    episode_done = False
    done_timer = 0
    done_text = ""
    done_color = TEXT_COLOR
    sim_time = 0.0

    running = True
    while running:
        # ── Events ──
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r:
                    obs, info = environment.reset()
                    agent_trail.clear()
                    threat_trail.clear()
                    total_reward = 0.0
                    episode += 1
                    episode_done = False
                    done_timer = 0

                elif event.key == pygame.K_SPACE:
                    control_mode = "random" if control_mode == "manual" else "manual"

                elif event.key == pygame.K_m:
                    # Toggle target mode — requires new environment
                    target_mode = "static" if target_mode == "moving" else "moving"
                    environment.close()
                    environment, config = make_env(target_mode)
                    obs, info = environment.reset()
                    agent_trail.clear()
                    threat_trail.clear()
                    total_reward = 0.0
                    episode = 1
                    episode_done = False
                    done_timer = 0

        # ── Determine action ──
        if episode_done:
            done_timer += 1
            if done_timer > FPS * 2:
                obs, info = environment.reset()
                agent_trail.clear()
                threat_trail.clear()
                total_reward = 0.0
                episode += 1
                episode_done = False
                done_timer = 0
            action = np.array([0.0, 0.0], dtype=np.float32)

        elif control_mode == "manual":
            keys = pygame.key.get_pressed()
            thrust = 1.0 if (keys[pygame.K_w] or keys[pygame.K_UP]) else -1.0
            rotation = 0.0
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                rotation = 1.0
            elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                rotation = -1.0
            action = np.array([thrust, rotation], dtype=np.float32)
        else:
            action = environment.action_space.sample()

        # ── Step ──
        if not episode_done:
            obs, reward, terminated, truncated, info = environment.step(action)
            total_reward += reward
            sim_time = info.get('time', 0.0)

            world = environment.unwrapped.world

            # Agent trail
            agent_trail.append((world.agent.position.x, world.agent.position.y))
            if len(agent_trail) > MAX_TRAIL:
                agent_trail.pop(0)

            # Threat trail (moving mode)
            if target_mode == "moving":
                threat_trail.append((world.threat.position.x, world.threat.position.y))
                if len(threat_trail) > MAX_TRAIL:
                    threat_trail.pop(0)

            if terminated or truncated:
                episode_done = True
                done_timer = 0
                if info.get('reached_target'):
                    done_text = "INTERCEPTED!" if target_mode == "moving" else "TARGET REACHED!"
                    done_color = SUCCESS_COLOR
                elif info.get('defense_failed', False):
                    done_text = "DEFENSE FAILED!"
                    done_color = FAIL_COLOR
                elif info.get('out_of_bounds'):
                    done_text = "OUT OF BOUNDS"
                    done_color = FAIL_COLOR
                else:
                    done_text = "TIMEOUT"
                    done_color = WARNING_COLOR

        # ── RENDER ──
        screen.fill(BG_COLOR)
        draw_grid(screen)

        world = environment.unwrapped.world

        if target_mode == "moving":
            # ── Moving mode scene ──

            # Dashed line: threat → protected target (threat's path)
            draw_dashed_line(screen, world.threat.position,
                             world.protected_position, (60, 30, 30))

            # Dashed line: agent → threat (what agent is chasing)
            draw_dashed_line(screen, world.agent.position,
                             world.threat.position, (30, 40, 60))

            # Trails
            draw_trail(screen, threat_trail, THREAT_TRAIL)
            draw_trail(screen, agent_trail, AGENT_TRAIL)

            # Protected target (green shield)
            draw_protected_target(screen, world.protected_position,
                                  config.PROTECTED_RADIUS, sim_time)

            # Threat rocket (red)
            draw_rocket(screen, world.threat.position, world.threat.angle,
                        THREAT_COLOR, THREAT_OUTLINE, size=10)

        else:
            # ── Static mode scene ──

            # Dashed line: agent → target
            draw_dashed_line(screen, world.agent.position,
                             world.target_position)

            # Trail
            draw_trail(screen, agent_trail, AGENT_TRAIL)

            # Static target (red crosshair)
            draw_static_target(screen, world.target_position,
                               config.TARGET_RADIUS, sim_time)

        # Agent (blue rocket — always drawn)
        is_thrusting = (not episode_done) and (
            (control_mode == "manual" and (pygame.key.get_pressed()[pygame.K_w] or
                                           pygame.key.get_pressed()[pygame.K_UP]))
            or (control_mode == "random" and action[0] > 0)
        )
        draw_rocket(screen, world.agent.position, world.agent.angle,
                    AGENT_COLOR, AGENT_OUTLINE, size=12, thrust_on=is_thrusting)

        # Info panel
        draw_info_panel(screen, font, font_small, info, control_mode,
                        target_mode, total_reward, episode)

        # Banner
        if episode_done and done_timer < FPS * 2:
            draw_banner(screen, font_banner, done_text, done_color)

        pygame.display.flip()
        clock.tick(FPS)

    environment.close()
    pygame.quit()


if __name__ == "__main__":
    main()
