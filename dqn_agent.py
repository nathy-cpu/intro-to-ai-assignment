import numpy as np
from neural_network import NeuralNetwork
from replay_buffer import ReplayBuffer

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