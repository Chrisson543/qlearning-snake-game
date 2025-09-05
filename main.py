import pygame
import sys
from settings import *
from functions import *
from snake import Snake
from NeuralNetwork import NeuralNetwork

screen = pygame.display.set_mode(screen_dim)
clock = pygame.time.Clock()

snake = Snake()
nn = NeuralNetwork(11, 8, 2)

while True:
    screen.fill(black)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    snake.update(screen)
    draw_grid(screen)
    pygame.display.update()
    clock.tick(60)