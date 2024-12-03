import argparse
from typing import Tuple

from src.agent import Agent
import os
import random
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from collections import deque
from operator import itemgetter

from src.agent_utils import nn_model


class VanillaDQN(Agent):
    def __init__(self, no_of_states, no_of_actions, config, do_load_model=False, model_file=None):
        
        super().__init__(config['hyperparams']['lr'])
        self.state_size = no_of_states
        self.action_size = no_of_actions

        # Hyperparameters
        self.gamma = config['hyperparams']['gamma']  # discount rate
        self.epsilon = config['hyperparams']['epsilon']  # eps-greedy exploration rate
        self.batch_size = config['hyperparams']['batch_size']  # maximum size of the batches sampled from memory
        self.epsilon_min = config['hyperparams']['epsilon_min']  # minimum eps-greedy exploration rate
        self.epsilon_decay = config['hyperparams']['epsilon_decay']  # decay of eps-greedy exploration rate
        self.alpha = config['hyperparams']['alpha']  # alpha parameter
        self.beta = config['hyperparams']['beta']  # beta parameter
        self.prior_eps = config['hyperparams']['prior_eps']  # epsilon for prioritization

        self.max_priority = 1
        self.max_weight = 1

        self.per = config['variants']['per']
        self.ddqn = config['variants']['ddqn']

        self.optimizer = tf.keras.optimizers.Adam()


        # Initialize the neural network models of weight and target weight networks
        self.model = nn_model(self.state_size, self.action_size, self.learning_rate, config['neural_nets']['model'])
        if do_load_model:
            self.model.load_weights(model_file)

        self.target_model = tf.keras.models.clone_model(self.model)  # Clone the model architecture
        self.target_model.set_weights(self.model.get_weights())  # Copy weights from self.model

        # Define times at which target weights are synchronized
        self.target_model_time = 8

        self.max_size = 2000

        # Maximal size of memory buffer
        self.memory = deque(maxlen=self.max_size)

    def select_action(self, state, training=True) -> Tuple[int, np.array]:
        """ToDO Use Tensorflow decorator"""
        # Ensure the use of exploration during training
        if training and tf.random.uniform(()) <= self.epsilon:
            # Take a random step
            return tf.random.uniform(shape=(), minval=0, maxval=self.action_size, dtype=tf.int32), None

        state = tf.expand_dims(state, axis=0)  # Adds a batch dimension
        q_values = self.model(state)

        # Use tf.argmax to select the action with the highest q-value
        action = tf.argmax(q_values, axis=1).numpy()[0]  # Convert to a scalar
        return action, q_values



    def process_step(self, episode: int, step: int, state, action, reward, next_state, prob, done):
        self.record(state, action, reward, next_state, done)

    def process_episode(self, episode):
        self._update_epsilon()
        self.update_weights(episode)


   # Here newly observed transitions are stored in the experience replay buffer
    def record(self, state, action, reward, next_state, done):
        if(self.per):
            self.memory.append(((state, action, reward, next_state, done),self.max_priority))
        else:
            self.memory.append((state, action, reward, next_state, done))

    # update epsilon as long as its larger than our minvalue
    def _update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_weights(self, t):  # t for time step at which target model will be updated
        # Check whether there are enough samples for the minibatch
        # Step 13: Initialize accumulated weight change (used for gradient updates)
        if len(self.memory) < self.batch_size:
            return

        if(self.per):
            minibatch, indices = self.sample_batch()
        else:
            # Sample a minibatch from memory
            minibatch = random.sample(self.memory, self.batch_size)

        # Extract components from the minibatch using TensorFlow
        states = tf.stack([state for state, _, _, _, _ in minibatch])
        # Extract and convert actions to ensure they are integers
        actions = [int(action) for _, action, _, _, _ in minibatch]  # Ensure actions are integers
        actions = tf.constant(actions, dtype=tf.int32)        
        rewards = tf.constant([reward for _, _, reward, _, _ in minibatch], dtype=tf.float32)
        next_states = tf.stack([next_state for _, _, _, next_state, _ in minibatch])
        dones = tf.stack([done for _, _, _, _, done in minibatch])
            
       # Compute Q-value targets
        if self.ddqn:
            next_q_values_online = tf.convert_to_tensor(self.model.predict(next_states, verbose=0), dtype=tf.float32)
            next_q_values_target = tf.convert_to_tensor(self.target_model.predict(next_states, verbose=0), dtype=tf.float32)
            best_next_actions = tf.argmax(next_q_values_online, axis=1)
            max_next_q_values = tf.gather(next_q_values_target, best_next_actions, batch_dims=1)
        else:
            next_q_values = tf.convert_to_tensor(self.target_model.predict(next_states, verbose=0), dtype=tf.float32)
            max_next_q_values = tf.reduce_max(next_q_values, axis=1)

        if(self.per):
            weights = self.update_priorities()  # Update priorities based on errors
            weights_tensor = tf.convert_to_tensor(weights, dtype=tf.float32)
        # Calculate the targets
        targets = rewards + (1 - tf.cast(dones, tf.float32)) * self.gamma * max_next_q_values

        # Perform gradient descent using gradient tape
        with tf.GradientTape() as tape:
            # Predict Q-values for current states
            q_values = self.model(states, training=True)
            # Gather the Q-values for the actions taken
            q_values_actions = tf.gather(q_values, actions, batch_dims=1)
            if(self.per):
                loss = tf.reduce_sum(weights_tensor * (targets - q_values_actions)** 2)
            else:
                # Compute the loss
                loss = tf.reduce_mean(tf.square(targets - q_values_actions))
        # Compute gradients of the loss with respect to the model's trainable variables
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # Apply the gradients to the model
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Regularly update target network weights
        if t % self.target_model_time == 0:
            self.target_model.set_weights(self.model.get_weights())

    def sample_batch(self):
        sum_of_prios = sum(prio**self.alpha for _, prio in self.memory)
        tuples = [tup for tup, _ in self.memory]
        prio = [(prio**self.alpha/sum_of_prios) for _, prio in self.memory]
        # A better solution: select the index directly by considering the weights
        # Select the indices directly considering the weights
        indices = random.choices(range(len(tuples)), weights=prio, k=self.batch_size)
        # Use a list comprehension to get the tuples corresponding to the selected indices
        picked_tuple = [tuples[i] for i in indices]
        return picked_tuple, indices
        
    def update_priorities(self, indices, priorities: np.ndarray):
        priorities = tf.abs(priorities)
        weights = []
        for idx, priority in zip(indices, priorities):
            if(priority>self.max_priority):
                self.max_priority = priority
            updated_weight = ((len(self.memory) * priority)**(-self.beta))/self.weights_max
            if updated_weight > self.weights_max:
                self.weights_max = updated_weight
            weights.append(updated_weight)
            # Assuming self.memory is a list of tuples, where each tuple is (item, priority)
            item, old_priority = self.memory[idx]
            self.memory[idx] = (item, priority)
        return weights

    # update epsilon as long as its larger than our minvalue
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
