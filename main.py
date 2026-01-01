import numpy as np
from dqn_agent import DQNAgent
from utils import plot_training_results, efficiency_analysis, benchmark_configurations

def create_mock_environment():
    """Create a mock environment for testing"""
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
    
    return MockCartPoleEnv()

def run_demo():
    """Run a complete DQN demonstration"""
    print("DQN Implementation using only NumPy")
    print("=" * 60)
    
    # Create mock environment
    env = create_mock_environment()
    
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
    
    # Plot results
    try:
        plot_training_results(training_stats)
    except:
        print("Note: Plotting requires matplotlib. Install with 'pip install matplotlib'")
    
    # Save model
    agent.save_model("dqn_model.npy")
    
    return agent, training_stats, test_stats

def run_benchmark():
    """Run configuration benchmarks"""
    print("\n" + "=" * 60)
    print("Running Configuration Benchmark...")
    print("=" * 60)
    
    env = create_mock_environment()
    
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
    
    results = benchmark_configurations(configs, env, episodes=20, steps=100)
    
    # Display benchmark results
    print("\n" + "=" * 60)
    print("Benchmark Results Summary:")
    print("=" * 60)
    
    for config_name, result in results.items():
        print(f"\n{config_name}:")
        print(f"  Avg Reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
        print(f"  Training Time: {result['train_time']:.2f}s")
        print(f"  Steps: {result['steps_trained']}")
        print(f"  Final Epsilon: {result['final_epsilon']:.3f}")
    
    return results

if __name__ == "__main__":
    print("DQN Implementation")
    print("=" * 60)
    
    # Run complete example
    agent, training_stats, test_stats = run_demo()
    
    # Uncomment to run benchmark
    # benchmark_results = run_benchmark()