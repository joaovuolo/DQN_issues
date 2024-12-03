from typing import Tuple
from src.agent import Agent, nn_model
import random
import numpy as np
import tensorflow as tf
from collections import deque

class VanillaDQN(Agent):
    def __init__(self, no_of_states, no_of_actions, config):
        
        super().__init__(config['algorithm'], config['hyperparams']['lr'])
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

        self.per = config['variants']['per']
        self.ddqn = config['variants']['ddqn']

        # Initialize the neural network models of weight and target weight networks
        self.model = nn_model(self.state_size, self.action_size, self.learning_rate, config['neural_nets']['model'])

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
            minibatch, indices = self.sample_batch(self.beta)
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

        # Get q_values that the primary model predicts for current states
        target_q_values = self.model.predict(states, verbose=0)

        # Convert predictions to tensors
        target_q_values = tf.Variable(target_q_values, dtype=tf.float32)

        if(self.ddqn):
            # Get Q-values that the online and target models predict for next states
            next_q_values_online = self.model.predict(next_states, verbose=0)  # Online model for action selection
            next_q_values_target = self.target_model.predict(next_states, verbose=0)  # Target model for value evaluation
            
            next_q_values_online = tf.convert_to_tensor(next_q_values_online, dtype=tf.float32)
            next_q_values_target = tf.convert_to_tensor(next_q_values_target, dtype=tf.float32)

            # Action selection using the online model
            best_next_actions = tf.argmax(next_q_values_online, axis=1)

            # Action evaluation using the target model
            max_next_q_values = tf.gather(next_q_values_target, best_next_actions, batch_dims=1)
        else:
            # Get q_values that the target model predicts for next states
            next_q_values = self.target_model.predict(next_states, verbose=0)

            # Convert predictions to tensors
            next_q_values = tf.convert_to_tensor(next_q_values, dtype=tf.float32)

            # Calculate the targets
            max_next_q_values = tf.reduce_max(next_q_values, axis=1)

        targets = rewards + (1-tf.cast(dones, tf.float32))*self.gamma * max_next_q_values
        if(self.per):
            self.update_priorities(indices,targets)

        # Update the target Q-values for the selected actions
        for i, a in enumerate(actions):
            target_q_values[i, a].assign(targets[i])  # Use TensorFlow to assign values
        
        # Train the model to fit the updated target Q-values
        self.model.fit(states, target_q_values, epochs=1, verbose=0)

        # Regularly update target network weights
        if t % self.target_model_time == 0:
            self.target_model.set_weights(self.model.get_weights())

    # Here newly observed transitions are stored in the experience replay buffer
    def record(self, state, action, reward, next_state, done):
        if(self.per):
            self.memory.append(((state, action, reward, next_state, done),self.max_priority))
        else:
            self.memory.append((state, action, reward, next_state, done))

    def sample_batch(self, beta: float = 0.4):
        sum_of_weights = sum(weight for _, weight in self.memory)
        tuples = [tup for tup, _ in self.memory]
        prio = [(weight/sum_of_weights)**beta for _, weight in self.memory]
        # A better solution: select the index directly by considering the weights
        # Select the indices directly considering the weights
        indices = random.choices(range(len(tuples)), weights=prio, k=self.batch_size)
        # Use a list comprehension to get the tuples corresponding to the selected indices
        picked_tuple = [tuples[i] for i in indices]
        return picked_tuple, indices
        
    def update_priorities(self, indices, priorities: np.ndarray):
        priorities = tf.abs(priorities)
        for idx, priority in zip(indices, priorities):
            # Assuming self.memory is a list of tuples, where each tuple is (item, priority)
            item, old_priority = self.memory[idx]
            self.memory[idx] = (item, priority)

    # update epsilon as long as its larger than our minvalue
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay