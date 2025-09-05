import numpy as np
import pygame
import random
from settings import *
from NeuralNetwork import NeuralNetwork


class Snake:
    def __init__(self):
        self.body = [[grid_size // 2, grid_size // 2]]
        self.food = self.place_food()
        self.direction = [1, 0]
        self.directions = [[0, -1], [1, 0], [0, 1], [-1, 0]]
        self.brain = NeuralNetwork(11, 8, 3)
        self.brain.load('model.npz')

        self.score = 0

        self.last_move = pygame.time.get_ticks()

    def get_next_direction_index(self, direction):
        new_direction_idx = (self.directions.index(self.direction) + direction) % len(self.directions)

        return new_direction_idx

    def set_direction(self, direction):
        next_direction_index = self.get_next_direction_index(direction)

        self.direction = self.directions[next_direction_index]

    def reset(self):
        self.body = [[grid_size // 2, grid_size // 2]]
        self.food = self.place_food()
        self.direction = [1, 0]

        self.score = 0

        self.last_move = pygame.time.get_ticks()

    def move(self, grow=False):
        head = self.body[0]
        new_head = [head[0] + self.direction[0], head[1] + self.direction[1]]

        self.body.insert(0, new_head)

        if not grow:
            self.body.pop()

    def eat_and_grow(self):
        self.body.append(self.food)
        self.food = self.place_food()

        self.score += 1

    def place_food(self):
        food = [random.randint(0, grid_size-1), random.randint(0, grid_size-1)]

        while food in self.body:
            food = [random.randint(0, grid_size-1), random.randint(0, grid_size-1)]

        return food

    def hit_wall(self, head):
        return head[0] >= grid_size or head[0] < 0 or head[1] >= grid_size or head[1] < 0

    def get_state(self):
        head = self.body[0]

        forward = [head[0] + self.direction[0], head[1] + self.direction[1]]
        left = [head[0] + self.directions[self.get_next_direction_index(-1)][0], head[1] + self.directions[self.get_next_direction_index(-1)][1]]
        right = [head[0] + self.directions[self.get_next_direction_index(1)][0], head[1] + self.directions[self.get_next_direction_index(1)][1]]

        def danger_in_direction(direction):
            return direction in self.body or self.hit_wall(direction)

        danger_ahead = danger_in_direction(forward)
        danger_left = danger_in_direction(left)
        danger_right = danger_in_direction(right)

        current_direction_index = self.directions.index(self.direction)

        moving_up = current_direction_index == 0
        moving_right = current_direction_index == 1
        moving_down = current_direction_index == 2
        moving_left = current_direction_index == 3

        food_left = self.food[0] < head[0]
        food_right = self.food[0] > head[0]
        food_above = self.food[1] < head[1]
        food_below = self.food[1] > head[1]

        state = [
            danger_ahead,
            danger_left,
            danger_right,
            moving_up,
            moving_right,
            moving_down,
            moving_left,
            food_left,
            food_right,
            food_above,
            food_below
        ]

        return state

    def collision_check(self):
        head = self.body[0]

        if head in self.body[1:] or self.hit_wall(head):
            self.reset()

        if self.food == head:
            self.score += 1
            self.food = self.place_food()
            self.move(grow=True)

    def draw(self, screen):

        pygame.draw.rect(screen, green, pygame.Rect(self.food[0] * block_size, self.food[1] * block_size, block_size, block_size))

        for i in range(len(self.body)):
            if i == 0:
                pygame.draw.rect(screen, red, pygame.Rect(self.body[i][0] * block_size, self.body[i][1] * block_size, block_size, block_size))
            else:
                pygame.draw.rect(screen, blue, pygame.Rect(self.body[i][0] * block_size, self.body[i][1] * block_size, block_size, block_size))

    def update(self, screen):
        if pygame.time.get_ticks() - self.last_move > 200:
            action = np.argmax(self.brain.forward(self.get_state()))
            direction = -1 if action == 0 else 0 if action == 1 else 1
            self.set_direction(direction)
            self.move()
            self.collision_check()
            self.last_move = pygame.time.get_ticks()
        self.draw(screen)
        self.get_state()