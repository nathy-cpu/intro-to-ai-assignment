import numpy as np
import matplotlib.pyplot as plt
import time

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
    print(f"   Consistency Score: {test_stats['mean_reward']/max(test_stats['std_reward'], 1e-6):.2f} "
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

def benchmark_configurations(configs, env, episodes=20, steps=100):
    """
    Benchmark different DQN configurations
    
    Args:
        configs: Dictionary of configuration names to config dicts
        env: Environment
        episodes: Number of episodes per benchmark
        steps: Maximum steps per episode
        
    Returns:
        Dictionary of benchmark results
    """
    results = {}
    
    for config_name, config in configs.items():
        print(f"\nBenchmarking {config_name} configuration...")
        
        # Create agent with this config
        agent = DQNAgent(env.state_size, env.action_size, config)
        
        # Train
        start_time = time.time()
        train_stats = agent.train(env, num_episodes=episodes, max_steps=steps, verbose=False)
        train_time = time.time() - start_time
        
        # Test
        test_stats = agent.test(env, num_episodes=5, verbose=False)
        
        results[config_name] = {
            'mean_reward': test_stats['mean_reward'],
            'std_reward': test_stats['std_reward'],
            'train_time': train_time,
            'steps_trained': agent.steps_done,
            'final_epsilon': agent.epsilon,
            'train_stats': train_stats
        }
        
        print(f"  Average Reward: {test_stats['mean_reward']:.2f}")
        print(f"  Training Time: {train_time:.2f}s")
        print(f"  Training Steps: {agent.steps_done}")
    
    return results