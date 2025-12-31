"""
Deep Q-Network (DQN) Implementation using only NumPy
Authors:
    Memar Gedefaw 	    UGR/7591/15 
    Nathnael Merihun	UGR/0789/15
    Tsion Feleke	    UGR/1689/15
    Yeabisra Bizuwork	UGR/4173/15
Course: Introduction to Artificial Intelligence
Addis Ababa University
"""

import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import sys
import time

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.001):
        """
        Initialize neural network with given architecture
        
        Args:
            input_size: Dimension of input state
            hidden_sizes: List of hidden layer sizes
            output_size: Number of actions (output Q-values)
            learning_rate: Learning rate for gradient descent
        """
        self.lr = learning_rate
        
        # Initialize layers
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.weights = []
        self.biases = []
        
        # He initialization for ReLU layers
        for i in range(len(self.layer_sizes) - 1):
            # He initialization: sqrt(2/fan_in)
            fan_in = self.layer_sizes[i]
            scale = np.sqrt(2.0 / fan_in)
            weight = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * scale
            bias = np.zeros((1, self.layer_sizes[i+1]))
            
            self.weights.append(weight)
            self.biases.append(bias)
        
        # Store intermediate values for backpropagation
        self.activations = []
        self.z_values = []
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input state (can be batch or single state)
            
        Returns:
            Q-values for each action
        """
        # Ensure x is 2D
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        self.activations = [x]
        self.z_values = []
        
        # Hidden layers with ReLU activation
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            a = np.maximum(0, z)  # ReLU activation
            self.z_values.append(z)
            self.activations.append(a)
        
        # Output layer (linear activation)
        z_out = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z_out)
        self.activations.append(z_out)
        
        return self.activations[-1]
    
    def backward(self, state, action, target):
        """
        Backward pass (gradient descent) for a single sample
        
        Args:
            state: Input state
            action: Action index taken
            target: TD target value
        """
        # Ensure inputs are 2D
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        
        # Forward pass to populate activations
        q_values = self.forward(state)
        
        # Get predicted Q-value for the taken action
        q_pred = q_values[0, action]
        
        # Compute gradient at output
        d_loss = 2 * (q_pred - target)  # Derivative of MSE
        
        # Initialize gradients
        d_weights = [np.zeros_like(w) for w in self.weights]
        d_biases = [np.zeros_like(b) for b in self.biases]
        
        # Backpropagate through output layer
        d_output = np.zeros_like(q_values)
        d_output[0, action] = d_loss
        
        # Gradient for output layer
        d_weights[-1] = self.activations[-2].T @ d_output
        d_biases[-1] = np.sum(d_output, axis=0, keepdims=True)
        
        # Backpropagate through hidden layers
        d_current = d_output
        
        for l in range(len(self.weights) - 2, -1, -1):
            # Gradient through activation (ReLU)
            d_z = d_current @ self.weights[l+1].T
            d_a = d_z * (self.z_values[l] > 0)  # ReLU derivative
            
            # Gradient for weights and biases
            d_weights[l] = self.activations[l].T @ d_a
            d_biases[l] = np.sum(d_a, axis=0, keepdims=True)
            
            d_current = d_a
        
        # Gradient clipping to prevent explosion
        for i in range(len(d_weights)):
            np.clip(d_weights[i], -1.0, 1.0, out=d_weights[i])
            np.clip(d_biases[i], -1.0, 1.0, out=d_biases[i])
        
        # Update weights
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * d_weights[i]
            self.biases[i] -= self.lr * d_biases[i]
    
    def batch_backward(self, states, actions, targets):
        """
        Batch gradient descent for multiple samples
        
        Args:
            states: Batch of states
            actions: Batch of actions
            targets: Batch of TD targets
        """
        batch_size = len(states)
        
        # Initialize gradients
        d_weights = [np.zeros_like(w) for w in self.weights]
        d_biases = [np.zeros_like(b) for b in self.biases]
        
        # Process each sample in batch
        total_loss = 0
        for i in range(batch_size):
            # Forward pass
            q_values = self.forward(states[i])
            q_pred = q_values[0, actions[i]]
            
            # Compute loss
            loss = (q_pred - targets[i]) ** 2
            total_loss += loss
            d_loss = 2 * (q_pred - targets[i])
            
            # Get gradients for this sample
            # Gradient for output layer
            d_output = np.zeros_like(q_values)
            d_output[0, actions[i]] = d_loss
            
            d_weights[-1] += self.activations[-2].T @ d_output / batch_size
            d_biases[-1] += np.sum(d_output, axis=0, keepdims=True) / batch_size
            
            # Backpropagate through hidden layers
            d_current = d_output
            
            for l in range(len(self.weights) - 2, -1, -1):
                d_z = d_current @ self.weights[l+1].T
                d_a = d_z * (self.z_values[l] > 0)
                
                d_weights[l] += self.activations[l].T @ d_a / batch_size
                d_biases[l] += np.sum(d_a, axis=0, keepdims=True) / batch_size
                
                d_current = d_a
        
        # Gradient clipping
        for i in range(len(d_weights)):
            np.clip(d_weights[i], -1.0, 1.0, out=d_weights[i])
            np.clip(d_biases[i], -1.0, 1.0, out=d_biases[i])
        
        # Update weights
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * d_weights[i]
            self.biases[i] -= self.lr * d_biases[i]
        
        return total_loss / batch_size
    
    def predict(self, state):
        """
        Get Q-values for a state
        
        Args:
            state: Input state
            
        Returns:
            Q-values for each action
        """
        return self.forward(state)[0]
    
    def copy_weights(self, other_network):
        """
        Copy weights from another network
        
        Args:
            other_network: Network to copy weights from
        """
        for i in range(len(self.weights)):
            self.weights[i] = other_network.weights[i].copy()
            self.biases[i] = other_network.biases[i].copy()


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


class DQNAgent:
    
    def __init__(self, state_size, action_size, config=None):
        """
        Initialize DQN agent
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            config: Configuration dictionary (optional)
        """
        self.state_size = state_size
        self.action_size = action_size
        
        # Default configuration
        default_config = {
            'hidden_sizes': [128, 64],
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'buffer_capacity': 10000,
            'batch_size': 64,
            'target_update_freq': 100,
            'train_freq': 4,
            'learning_starts': 1000
        }
        
        # Merge with provided config
        self.config = default_config if config is None else {**default_config, **config}
        
        # Initialize networks
        self.policy_net = NeuralNetwork(
            state_size, 
            self.config['hidden_sizes'], 
            action_size, 
            self.config['learning_rate']
        )
        
        self.target_net = NeuralNetwork(
            state_size, 
            self.config['hidden_sizes'], 
            action_size, 
            self.config['learning_rate']
        )
        
        # Copy weights to target network
        self.target_net.copy_weights(self.policy_net)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(self.config['buffer_capacity'])
        
        # Initialize exploration rate
        self.epsilon = self.config['epsilon_start']
        
        # Training counters
        self.steps_done = 0
        self.episodes_done = 0
        self.update_counter = 0
        
        # Statistics
        self.loss_history = []
        self.reward_history = []
        self.epsilon_history = []
        
    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: Whether in training mode (uses epsilon)
            
        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.action_size)
        else:
            # Exploit: action with highest Q-value
            q_values = self.policy_net.predict(state)
            return np.argmax(q_values)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.config['epsilon_end'], 
                          self.epsilon * self.config['epsilon_decay'])
    
    def store_experience(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """
        Perform one training step
        
        Returns:
            Loss value if training occurred, else None
        """
        # Check if we have enough samples
        if len(self.memory) < self.config['batch_size']:
            return None
        
        # Sample batch from replay buffer
        batch = self.memory.sample(self.config['batch_size'])
        if batch is None:
            return None
        
        states, actions, rewards, next_states, dones = batch
        
        # Compute Q-values for next states using target network
        next_q_values = np.zeros(self.config['batch_size'])
        for i in range(self.config['batch_size']):
            q_vals = self.target_net.predict(next_states[i])
            next_q_values[i] = np.max(q_vals)
        
        # Compute TD targets
        targets = rewards + self.config['gamma'] * next_q_values * (1 - dones)
        
        # Train policy network
        loss = self.policy_net.batch_backward(states, actions, targets)
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.config['target_update_freq'] == 0:
            self.target_net.copy_weights(self.policy_net)
        
        return loss
    
    def train(self, env, num_episodes, max_steps=500, render=False, verbose=True):
        """
        Train the DQN agent
        
        Args:
            env: Environment to train on
            num_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            render: Whether to render the environment
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training statistics
        """
        episode_rewards = []
        episode_lengths = []
        losses = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            state = state.astype(np.float32)
            
            total_reward = 0
            episode_loss = 0
            steps_in_episode = 0
            
            for step in range(max_steps):
                if render and episode % 10 == 0:
                    env.render()
                
                # Select action
                action = self.select_action(state, training=True)
                
                # Take action
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = next_state.astype(np.float32)
                done = terminated or truncated
                
                # Store experience
                self.store_experience(state, action, reward, next_state, done)
                
                # Train if enough experiences collected
                if self.steps_done >= self.config['learning_starts'] and \
                   self.steps_done % self.config['train_freq'] == 0:
                    loss = self.train_step()
                    if loss is not None:
                        episode_loss += loss
                        losses.append(loss)
                
                # Update state and counters
                state = next_state
                total_reward += reward
                steps_in_episode += 1
                self.steps_done += 1
                
                if done:
                    break
            
            # Decay epsilon
            self.decay_epsilon()
            
            # Record statistics
            episode_rewards.append(total_reward)
            episode_lengths.append(steps_in_episode)
            self.reward_history.append(total_reward)
            self.epsilon_history.append(self.epsilon)
            
            # Log progress
            if verbose and (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode + 1}/{num_episodes}, "
                      f"Reward: {total_reward:.1f}, "
                      f"Avg Reward (last 10): {avg_reward:.1f}, "
                      f"Epsilon: {self.epsilon:.3f}, "
                      f"Steps: {self.steps_done}")
        
        self.episodes_done += num_episodes
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'losses': losses,
            'epsilon_history': self.epsilon_history
        }
    
    def test(self, env, num_episodes=10, max_steps=500, render=False, verbose=True):
        """
        Test the trained agent
        
        Args:
            env: Environment to test on
            num_episodes: Number of test episodes
            max_steps: Maximum steps per episode
            render: Whether to render the environment
            verbose: Whether to print results
            
        Returns:
            Dictionary with test statistics
        """
        test_rewards = []
        test_lengths = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            state = state.astype(np.float32)
            
            total_reward = 0
            steps_in_episode = 0
            
            for step in range(max_steps):
                if render:
                    env.render()
                
                # Always exploit during testing
                action = self.select_action(state, training=False)
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = next_state.astype(np.float32)
                done = terminated or truncated
                
                state = next_state
                total_reward += reward
                steps_in_episode += 1
                
                if done:
                    break
            
            test_rewards.append(total_reward)
            test_lengths.append(steps_in_episode)
            
            if verbose:
                print(f"Test Episode {episode + 1}/{num_episodes}, "
                      f"Reward: {total_reward:.1f}, "
                      f"Length: {steps_in_episode}")
        
        stats = {
            'test_rewards': test_rewards,
            'test_lengths': test_lengths,
            'mean_reward': np.mean(test_rewards),
            'std_reward': np.std(test_rewards),
            'max_reward': np.max(test_rewards),
            'min_reward': np.min(test_rewards)
        }
        
        if verbose:
            print(f"\nTest Results:")
            print(f"Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
            print(f"Max Reward: {stats['max_reward']:.2f}")
            print(f"Min Reward: {stats['min_reward']:.2f}")
        
        return stats
    
    def save_model(self, filename):
        """
        Save model weights to file
        
        Args:
            filename: Path to save file
        """
        model_data = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'config': self.config,
            'policy_weights': self.policy_net.weights,
            'policy_biases': self.policy_net.biases,
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }
        np.save(filename, model_data, allow_pickle=True)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """
        Load model weights from file
        
        Args:
            filename: Path to load file from
        """
        model_data = np.load(filename, allow_pickle=True).item()
        
        # Create new networks with loaded architecture
        self.policy_net = NeuralNetwork(
            model_data['state_size'],
            model_data['config']['hidden_sizes'],
            model_data['action_size'],
            model_data['config']['learning_rate']
        )
        self.target_net = NeuralNetwork(
            model_data['state_size'],
            model_data['config']['hidden_sizes'],
            model_data['action_size'],
            model_data['config']['learning_rate']
        )
        
        # Load weights
        self.policy_net.weights = model_data['policy_weights']
        self.policy_net.biases = model_data['policy_biases']
        self.target_net.copy_weights(self.policy_net)
        
        # Load other parameters
        self.epsilon = model_data['epsilon']
        self.steps_done = model_data['steps_done']
        self.config = model_data['config']
        
        print(f"Model loaded from {filename}")


def plot_training_results(stats, window=10):
    """
    Plot training results
    
    Args:
        stats: Dictionary with training statistics
        window: Window size for moving average
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot episode rewards
    episode_rewards = stats['episode_rewards']
    axes[0, 0].plot(episode_rewards, alpha=0.6, label='Episode Reward')
    
    # Calculate moving average
    moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    axes[0, 0].plot(range(window-1, len(episode_rewards)), moving_avg, 
                   'r-', linewidth=2, label=f'{window}-Episode Moving Avg')
    
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Training: Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot episode lengths
    axes[0, 1].plot(stats['episode_lengths'], alpha=0.6)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].set_title('Training: Episode Lengths')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot losses
    if stats['losses']:
        axes[1, 0].plot(stats['losses'], alpha=0.6)
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot epsilon decay
    axes[1, 1].plot(stats['epsilon_history'])
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Epsilon')
    axes[1, 1].set_title('Exploration Rate (Epsilon) Decay')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def efficiency_analysis(agent, env, num_test_runs=5):
    """
    Analyze efficiency of the trained agent
    
    Args:
        agent: Trained DQN agent
        env: Environment
        num_test_runs: Number of test runs for analysis
    """
    print("=" * 60)
    print("EFFICIENCY ANALYSIS")
    print("=" * 60)
    
    # Test performance
    test_stats = agent.test(env, num_episodes=num_test_runs, render=False, verbose=False)
    
    print(f"\n1. Performance Metrics:")
    print(f"   Average Reward: {test_stats['mean_reward']:.2f}")
    print(f"   Reward Std Dev: {test_stats['std_reward']:.2f}")
    print(f"   Consistency Score: {test_stats['mean_reward']/test_stats['std_reward']:.2f} "
          f"(higher is better)")
    
    # Sample efficiency
    print(f"\n2. Sample Efficiency:")
    print(f"   Total Training Steps: {agent.steps_done}")
    print(f"   Total Episodes: {agent.episodes_done}")
    print(f"   Steps per Episode: {agent.steps_done/max(1, agent.episodes_done):.1f}")
    
    # Exploration efficiency
    print(f"\n3. Exploration Analysis:")
    print(f"   Final Epsilon: {agent.epsilon:.4f}")
    print(f"   Target Epsilon: {agent.config['epsilon_end']:.4f}")
    print(f"   Exploration Decayed: {100 * (1 - agent.epsilon/agent.config['epsilon_start']):.1f}%")
    
    # Memory usage analysis
    print(f"\n4. Memory Usage:")
    print(f"   Replay Buffer Usage: {len(agent.memory)}/{agent.config['buffer_capacity']} "
          f"({100*len(agent.memory)/agent.config['buffer_capacity']:.1f}%)")
    
    # Training efficiency
    if agent.loss_history:
        recent_losses = agent.loss_history[-100:] if len(agent.loss_history) >= 100 else agent.loss_history
        avg_recent_loss = np.mean(recent_losses) if recent_losses else 0
        print(f"\n5. Training Convergence:")
        print(f"   Recent Average Loss: {avg_recent_loss:.4f}")
        print(f"   Loss Trend: {'Converging' if avg_recent_loss < 0.5 else 'Still Learning'}")
    
    # Success criteria for CartPole
    success_threshold = 195  # OpenAI's threshold for "solving" CartPole
    success_rate = 100 * np.mean([r >= success_threshold for r in test_stats['test_rewards']])
    
    print(f"\n6. Success Rate:")
    print(f"   Success Threshold: {success_threshold}+ reward")
    print(f"   Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print(f"   ✓ Agent successfully solved the environment!")
    elif success_rate >= 50:
        print(f"   ~ Agent shows promising performance")
    else:
        print(f"   ✗ Agent needs more training")
    
    print("=" * 60)
    
    return test_stats


def run_complete_dqn_example():
    """
    Complete example of training and testing DQN on CartPole
    """
    # Note: We'll create a mock environment since we can't import gym in this pure NumPy code
    # In practice, you would import gym and use: env = gym.make('CartPole-v1')
    
    print("DQN Implementation using only NumPy")
    print("=" * 60)
    
    # For demonstration, we'll simulate a simple environment
    class MockCartPoleEnv:
        """Mock environment for demonstration"""
        def __init__(self):
            self.state_size = 4
            self.action_size = 2
            self.max_steps = 500
            
        def reset(self):
            state = np.random.randn(4).astype(np.float32) * 0.05
            return state, {}
        
        def step(self, action):
            # Simulate dynamics
            done = np.random.random() < 0.01  # 1% chance of termination per step
            reward = 1.0 if not done else 0.0
            next_state = np.random.randn(4).astype(np.float32) * 0.05
            
            # Simulate truncation
            truncated = False
            
            return next_state, reward, done, truncated, {}
        
        def render(self):
            pass
    
    # Create mock environment
    env = MockCartPoleEnv()
    
    # Initialize DQN agent
    print("\nInitializing DQN Agent...")
    agent_config = {
        'hidden_sizes': [128, 64],
        'learning_rate': 0.001,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'buffer_capacity': 10000,
        'batch_size': 32,  # Smaller for faster training
        'target_update_freq': 100,
        'train_freq': 4,
        'learning_starts': 100
    }
    
    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        config=agent_config
    )
    
    # Train the agent
    print("\nTraining DQN Agent...")
    print("=" * 60)
    
    training_stats = agent.train(
        env=env,
        num_episodes=50,  # Small number for quick demonstration
        max_steps=200,
        render=False,
        verbose=True
    )
    
    # Test the agent
    print("\n" + "=" * 60)
    print("Testing DQN Agent...")
    print("=" * 60)
    
    test_stats = agent.test(
        env=env,
        num_episodes=10,
        max_steps=200,
        render=False,
        verbose=True
    )
    
    # Analyze efficiency
    efficiency_analysis(agent, env, num_test_runs=10)
    
    # Plot results (commented out for pure NumPy environment)
    # plot_training_results(training_stats)
    
    # Save model
    agent.save_model("dqn_model.npy")
    
    # Demonstrate loading
    print("\n" + "=" * 60)
    print("Demonstrating Model Loading...")
    print("=" * 60)
    
    loaded_agent = DQNAgent(env.state_size, env.action_size)
    loaded_agent.load_model("dqn_model.npy")
    
    # Test loaded agent
    loaded_test_stats = loaded_agent.test(
        env=env,
        num_episodes=5,
        render=False,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("Implementation Complete!")
    print("=" * 60)
    
    return agent, training_stats, test_stats


def benchmark_dqn():
    """Benchmark different DQN configurations"""
    print("DQN Configuration Benchmark")
    print("=" * 60)
    
    # Different configurations to test
    configs = {
        'Baseline': {
            'hidden_sizes': [64, 32],
            'learning_rate': 0.001,
            'batch_size': 32
        },
        'Deep Network': {
            'hidden_sizes': [128, 64, 32],
            'learning_rate': 0.0005,
            'batch_size': 64
        },
        'Fast Learning': {
            'hidden_sizes': [128, 64],
            'learning_rate': 0.01,
            'batch_size': 16,
            'epsilon_decay': 0.98
        }
    }
    
    # Mock environment
    class SimpleEnv:
        def __init__(self):
            self.state_size = 4
            self.action_size = 2
            
        def reset(self):
            return np.random.randn(4).astype(np.float32) * 0.05, {}
        
        def step(self, action):
            done = np.random.random() < 0.05
            reward = 1.0 if not done else 0.0
            next_state = np.random.randn(4).astype(np.float32) * 0.05
            return next_state, reward, done, False, {}
    
    env = SimpleEnv()
    
    results = {}
    for config_name, config in configs.items():
        print(f"\nTesting {config_name} configuration...")
        
        # Merge with default config
        full_config = {
            'hidden_sizes': [128, 64],
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'buffer_capacity': 5000,
            'batch_size': 32,
            'target_update_freq': 100,
            'train_freq': 4,
            'learning_starts': 100
        }
        full_config.update(config)
        
        # Create and train agent
        agent = DQNAgent(env.state_size, env.action_size, full_config)
        
        # Short training for benchmarking
        stats = agent.train(env, num_episodes=20, max_steps=100, verbose=False)
        
        # Test
        test_stats = agent.test(env, num_episodes=10, verbose=False)
        
        results[config_name] = {
            'mean_reward': test_stats['mean_reward'],
            'std_reward': test_stats['std_reward'],
            'final_epsilon': agent.epsilon,
            'steps_trained': agent.steps_done
        }
        
        print(f"  Average Reward: {test_stats['mean_reward']:.2f}")
        print(f"  Training Steps: {agent.steps_done}")
    
    # Display benchmark results
    print("\n" + "=" * 60)
    print("Benchmark Results Summary:")
    print("=" * 60)
    
    for config_name, result in results.items():
        print(f"\n{config_name}:")
        print(f"  Avg Reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
        print(f"  Steps: {result['steps_trained']}")
        print(f"  Final Epsilon: {result['final_epsilon']:.3f}")
    
    return results


if __name__ == "__main__":
    print("DQN Implementation")
    print("=" * 60)
    
    # Run complete example
    agent, training_stats, test_stats = run_complete_dqn_example()
    
    '''
    # TO run benchmark
    print("\n" + "=" * 60)
    print("Running Configuration Benchmark...")
    print("=" * 60)
    benchmark_results = benchmark_dqn()
    '''
