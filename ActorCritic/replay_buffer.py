import random
from collections import deque, namedtuple
import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        """Simple replay buffer with deque

        Args:
            buffer_size (int): maximum buffer length
            batch_size (int): batch size to be sampled
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)
        self.experience = namedtuple(
            'Experience', ['state', 'action', 'reward', 'next_state', 'done'])

    def add(self, state, action, reward, next_state, done):
        """Add a sample to buffer
        """
        self.buffer.append(self.experience(
            state, action, reward, next_state, done))

    def sample(self):
        """Sample a batch from buffer

        Returns:
            tuple: tuple of ndarrays size of (batch_size, )
        """
        experiences = random.sample(self.buffer, self.batch_size)

        states = np.vstack(
            [i.state for i in experiences if experiences is not None])
        actions = np.vstack(
            [i.action for i in experiences if experiences is not None])
        rewards = np.vstack(
            [i.reward for i in experiences if experiences is not None])
        next_states = np.vstack(
            [i.next_state for i in experiences if experiences is not None])
        dones = np.vstack(
            [i.done for i in experiences if experiences is not None])

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.buffer)


class ReplayBufferEff:
    def __init__(self, buffer_size=1000, batch_size=64):
        """Effeicent replay buffer with numpy array indexing

        Args:
            buffer_size (int, optional): maximum buffer length. Defaults to 1000.
            batch_size (int, optional): batch size to be sampled. Defaults to 64.
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.idx = 0
        self.size = 0

        self.state_memory = np.empty(buffer_size, dtype=np.ndarray)
        self.action_memory = np.empty(buffer_size, dtype=np.ndarray)
        self.reward_memory = np.empty(buffer_size, dtype=np.ndarray)
        self.next_state_memory = np.empty(buffer_size, dtype=np.ndarray)
        self.done_memory = np.empty(buffer_size, dtype=np.ndarray)

    def add(self, state, action, reward, next_state, done):
        """Add a sample to buffer
        """
        self.state_memory[self.idx] = state
        self.action_memory[self.idx] = action
        self.reward_memory[self.idx] = reward
        self.next_state_memory[self.idx] = next_state
        self.done_memory[self.idx] = done

        self.idx += 1
        self.idx %= self.buffer_size

        self.size += 1
        self.size = min(self.buffer_size, self.size)

    def sample(self):
        """Sample a batch

        Returns:
            tuple: tuple of ndarray size of (batch_size, )
        """
        experience_idx = np.random.choice(
            self.size, self.batch_size, replace=False)

        states = np.vstack(self.state_memory[experience_idx])
        actions = np.vstack(self.action_memory[experience_idx])
        rewards = np.vstack(self.reward_memory[experience_idx])
        next_states = np.vstack(self.next_state_memory[experience_idx])
        dones = np.vstack(self.done_memory[experience_idx])

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return self.size


class PrioritizedReplay:
    def __init__(self, buffer_size, batch_size, alpha=0.6, beta=0.1, seed=0):
        """Prioritized replay buffer

        Args:
            buffer_size (int): maximum buffer length
            batch_size (int): batch size to be sampled
            alpha (float, optional): weight to blend uniform probability of priority [0,1]. Defaults to 0.6.
            beta (float, optional): degree of correction to calculate importance sampling weights [0,1]. Defaults to 0.1.
            seed (int, optional): seed for numpy random. Defaults to 0.
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        np.random.seed(seed)

        self.eps = 1e-6
        self.alpha = alpha
        self.beta = beta
        self.beta_incr_rate = 0.99992

        self.buffer = np.empty((self.buffer_size, 2), dtype=object)
        # self.experience = np.zeros(5, dtype=object)
        self.idx = 0
        self.n_entries = 0
        self.next_idx = 0

        self.priority_idx = 0  # idx to store priority in buffer
        self.sample_idx = 1    # idx to store sample in buffer

        self.new_added_priority_idx = 0

    def add(self, state, action, reward, next_state, done):
        """add a sample to buffer
        """
        experience = np.empty(5, dtype=object)
        experience[0] = np.array(state)
        experience[1] = np.array(action)
        experience[2] = np.array(reward)
        experience[3] = np.array(next_state)
        experience[4] = np.array(done)

        priority = 1.0
        if self.n_entries > 0:
            priority = self.buffer[:self.n_entries, self.priority_idx].max()

        self.buffer[self.next_idx, self.priority_idx] = priority
        self.buffer[self.next_idx, self.sample_idx] = experience

        self.n_entries = min(self.n_entries + 1, self.buffer_size)
        self.next_idx = (self.next_idx + 1) % self.buffer_size

    def update_priorities(self, priorities, idxs):
        """update new priorities current buffer

        Args:
            priorities (ndarray): new calculated priorities
            idxs (ndarray): indexes of priorities to place in buffer
        """
        self.buffer[idxs, self.priority_idx] = np.abs(priorities)

    def sample(self):
        """sample a batch

        Returns:
            tuple: tuple of ndarray of a batch
        """
        self.beta = min(1.0, self.beta * self.beta_incr_rate**-1)

        enteries = self.buffer[:self.n_entries]
        priorities = (enteries[:, self.priority_idx] + self.eps) ** self.alpha
        probs = np.array(priorities / np.sum(priorities), dtype=np.float64)
        # exp = np.random.choice(self.buffer, self.batch_size, p=probs, replace=False)

        weights = (self.n_entries * probs) ** -self.beta
        normalized_weights = weights / weights.max()

        exp_idxs = np.random.choice(
            self.n_entries, size=self.batch_size, replace=False, p=probs)
        experiences = np.array([enteries[idx] for idx in exp_idxs])

        exp_idxs = np.vstack(exp_idxs)
        weights = np.vstack(normalized_weights[exp_idxs])
        states = np.vstack([experiences[i, self.sample_idx][0]
                            for i in range(len(experiences))])
        actions = np.vstack([experiences[i, self.sample_idx][1]
                             for i in range(len(experiences))])
        rewards = np.vstack([experiences[i, self.sample_idx][2]
                             for i in range(len(experiences))])
        next_states = np.vstack([experiences[i, self.sample_idx][3]
                                 for i in range(len(experiences))])
        dones = np.vstack([experiences[i, self.sample_idx][4]
                           for i in range(len(experiences))])

        return (exp_idxs, weights, states, actions, rewards, next_states, dones)

    def __len__(self):
        return self.n_entries
