import math
import numpy as np
import pygame
from NeuralNetwork import NeuralNetwork
from snake import Snake
import random


class Agent:
    def __init__(self):
        self.online_nn = NeuralNetwork(11, 8, 3)
        self.target_network = NeuralNetwork(11, 8, 3)
        self.target_network.copy(self.online_nn)
        self.snake = Snake()
        self.memory = []
        self.batch_size = 64

        self.gamma = 0.99
        self.epsilon = 0.05

        self.step_counter = 0
        self.eat_counter = 0
        self.last_save = 0

    def get_reward(self, died, ate_food, prev_head_food_dist):
        reward = -0.01

        if died:
            return reward - 10

        if ate_food:
            return reward + 10

        head = self.snake.body[0]
        head_food_dist = abs(head[0] - self.snake.food[0]) + abs(head[1] - self.snake.food[1])

        if head_food_dist > prev_head_food_dist:
            reward -= 0.03
        if head_food_dist < prev_head_food_dist:
            reward += 0.03

        return reward

    def step(self, action):

        self.snake.set_direction(-1 if action == 0 else 0 if action == 1 else 1)
        prev_head_pos = self.snake.body[0]
        self.snake.move()

        prev_head_food_dist = abs(prev_head_pos[0] - self.snake.food[0]) + abs(prev_head_pos[1] - self.snake.food[1])

        done = 1 if self.snake.hit_wall(self.snake.body[0]) or self.snake.body[0] in self.snake.body[1:] else 0
        ate_food = self.snake.body[0] == self.snake.food

        reward = self.get_reward(done, ate_food, prev_head_food_dist)

        if ate_food:
            self.snake.eat_and_grow()
            self.eat_counter = self.step_counter

        if self.step_counter - self.eat_counter >= 400:
            done = 1
            reward -= 1

        next_state = np.array(self.snake.get_state())

        self.step_counter += 1

        if self.step_counter % 50000 == 0:
            self.epsilon -= 0.05 if self.epsilon > 0.05 else 0

        return reward, next_state, done

    def calculate_y(self, reward, max_next_state_q_values, done):
        return np.where(done == 0, reward + self.gamma * max_next_state_q_values, reward)

    def add_to_memory(self, state, action, reward, next_state, done):
        data = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }

        self.memory.append(data)

    def train(self):
        for _ in range(30000):
            done = False
            loss = 0

            self.eat_counter = self.step_counter

            while not done:
                state = np.array(self.snake.get_state())

                q_values = self.online_nn.forward(state)[0]
                action = random.randint(0, q_values.size-1) if random.random() < self.epsilon else np.argmax(q_values)

                reward, next_state, done = self.step(action)

                self.add_to_memory(state, action, reward, next_state, done)

                if len(self.memory) > 1000:
                    batch = random.sample(self.memory, self.batch_size)

                    batch_states = np.array([batch[i]['state'] for i in range(self.batch_size)])
                    batch_next_state = np.array([batch[i]['next_state'] for i in range(self.batch_size)])
                    batch_actions = np.array([batch[i]['action'] for i in range(self.batch_size)])
                    batch_reward = np.array([batch[i]['reward'] for i in range(self.batch_size)])
                    batch_done = np.array([batch[i]['done'] for i in range(self.batch_size)])

                    batch_q_values = self.online_nn.forward(batch_states)

                    batch_next_state_q_values = self.target_network.forward(batch_next_state)

                    batch_max_next_state_q_values = np.max(batch_next_state_q_values, axis=1)

                    y_batch = self.calculate_y(batch_reward, batch_max_next_state_q_values, batch_done)
                    y_batch_padded = batch_q_values.copy()
                    y_batch_padded[np.arange(len(batch_q_values)), batch_actions] = y_batch

                    self.online_nn.backward(batch_states, y_batch_padded, batch_q_values)
                    self.online_nn.update_parameters(1e-3)
                    loss = self.online_nn.calculate_loss(batch_q_values, y_batch_padded)

                if self.step_counter >= 1000 and self.step_counter % 1000 == 0:
                    self.target_network.copy(self.online_nn)

            print(f'iteration: {_}. score: {self.snake.score}. epsilon: {self.epsilon:.2f}. loss: {loss:.2f}. steps: {self.step_counter}')
            self.snake.reset()

            if len(self.memory) > 50000:
                self.memory = self.memory[-50000:]

            if self.step_counter - self.last_save >= 10000:
                print('save')
                self.online_nn.save('model.npz')
                self.last_save = self.step_counter


agent = Agent()
agent.online_nn.load('model.npz')
agent.train()