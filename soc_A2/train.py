import numpy as np
import matplotlib.pyplot as plt
from env import ProbeEnv
from agent import ProbeAgent

def train_agent(episodes=20000):
    env = ProbeEnv()
    agent = ProbeAgent(epsilon_decay=0.9995) 
    
    rewards_history = []
    success_history = []
    
    print("Beginning atmospheric reinforcement optimization training loop...")
    for episode in range(1, episodes + 1):
        raw_state = env.reset()
        state = agent.discretize_state(raw_state)
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_raw_state, reward, done, info = env.step(action)
            next_state = agent.discretize_state(next_raw_state)
            
            agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
        agent.decay_epsilon()
        
        rewards_history.append(total_reward)
        success_history.append(1.0 if info["outcome"] == "success" else 0.0)
        
        if episode % 1000 == 0:
            recent_success = np.mean(success_history[-100:]) * 100
            print(f"Episode {episode:5d}/{episodes} | Epsilon: {agent.epsilon:.3f} | Window Success Rate: {recent_success:.1f}%")
            
    np.save("q_table.npy", agent.q_table)
    print("Training finished. Saving Q-table matrix plots...")
    
    window = 200
    r_avg = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
    s_avg = np.convolve(success_history, np.ones(window)/window, mode='valid') * 100
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    axs[0].plot(r_avg, color='blue')
    axs[0].set_title('Moving Average of Episode Total Reward')
    axs[0].set_ylabel('Reward Units')
    axs[0].grid(True)
    
    axs[1].plot(s_avg, color='green')
    axs[1].set_title('Moving Average of Landing Success Rate (%)')
    axs[1].set_xlabel('Episodes')
    axs[1].set_ylabel('Success Rate %')
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.savefig("learning_curve.png")
    plt.show()

if __name__ == "__main__":
    train_agent()