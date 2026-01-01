import random
from collections import deque
import numpy as np

class ReplayBuffer:
    
    def __init__(self, capacity):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """
        Add experience to buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        experience = (state.copy(), action, reward, next_state.copy(), done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """
        Sample random batch of experiences
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Batch of experiences
        """
        if len(self.buffer) < batch_size:
            return None
        
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Get current buffer size"""
        return len(self.buffer)