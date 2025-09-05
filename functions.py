import pygame
from settings import *


def draw(screen, grid):
    for y in range(grid_size):
        for x in range(grid_size):
            x_screen = x * block_size
            y_screen = y * block_size
            if grid[y][x] == 1:
                pygame.draw.rect(screen, red, pygame.Rect(x_screen, y_screen, block_size, block_size))

            elif grid[y][x] == 2:
                pygame.draw.rect(screen, green, pygame.Rect(x_screen, y_screen, block_size, block_size))


def draw_grid(screen):
    for i in range(1, grid_size):
        start_pos = (i * block_size, 0)
        end_pos = (i * block_size, screen_dim[1])
        pygame.draw.line(screen, white, start_pos, end_pos, 1)

    for i in range(1, grid_size):
        start_pos = (0, i * block_size)
        end_pos = (screen_dim[1], i * block_size)
        pygame.draw.line(screen, white, start_pos, end_pos, 1)