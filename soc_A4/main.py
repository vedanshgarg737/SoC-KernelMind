import numpy as np
import matplotlib.pyplot as plt
import torch

from core.generator import WorkloadGenerator
from core.simulator import BaselineSimulators
from env.scheduler_env import DirectSchedulerEnvironment
from models.dqn_agent import DoubleDQNAgent
from utils.metrics import calculate_metrics

def main():
    # Set Seed for Reproducibility
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 1. Universal Evaluation Test Set
    test_workloads = [WorkloadGenerator.generate_uniform(num_processes=20, seed=100 + i) for i in range(20)]

    # 2. Evaluate Baselines
    baseline_results = {"FCFS": [], "SJF": [], "RR": []}
    for w in test_workloads:
        _, m_fcfs = BaselineSimulators.simulate_fcfs(w)
        _, m_sjf = BaselineSimulators.simulate_sjf(w)
        _, m_rr = BaselineSimulators.simulate_rr(w)
        
        baseline_results["FCFS"].append(m_fcfs)
        baseline_results["SJF"].append(m_sjf)
        baseline_results["RR"].append(m_rr)

    fcfs_mean_wait = np.mean([m["mean_wait_time"] for m in baseline_results["FCFS"]])
    sjf_mean_wait = np.mean([m["mean_wait_time"] for m in baseline_results["SJF"]])
    rr_mean_wait = np.mean([m["mean_wait_time"] for m in baseline_results["RR"]])

    fcfs_fairness = np.mean([m["jain_fairness"] for m in baseline_results["FCFS"]])
    sjf_fairness = np.mean([m["jain_fairness"] for m in baseline_results["SJF"]])
    rr_fairness = np.mean([m["jain_fairness"] for m in baseline_results["RR"]])

    # 3. Train Direct Neural Scheduler Agent
    env = DirectSchedulerEnvironment(queue_window_size=10)
    agent = DoubleDQNAgent(feature_dim=3, queue_size=10, lr=5e-4)

    episodes = 400
    epsilon = 1.0
    eps_decay = 0.992
    min_eps = 0.05
    
    train_wait_history = []
    q_magnitude_history = []

    print("--- Starting Direct Neural Scheduler Training ---")
    for ep in range(episodes):
        workload = WorkloadGenerator.generate_uniform(num_processes=20, seed=ep)
        state, mask = env.reset(workload)
        done = False
        ep_q_vals = []

        while not done:
            action = agent.select_action(state, mask, epsilon)
            (next_state, next_mask), reward, done, _ = env.step(action)

            agent.replay_buffer.push(state, mask, action, reward, next_state, next_mask, done)
            agent.train_step(batch_size=32)

            # Track Q-value magnitudes for diagnostics
            with torch.no_grad():
                st_t = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(agent.device)
                mk_t = torch.tensor(np.array(mask), dtype=torch.bool).unsqueeze(0).to(agent.device)
                q_v = agent.policy_net(st_t, mk_t)
                valid_q = q_v[mk_t]
                if len(valid_q) > 0:
                    ep_q_vals.append(valid_q.abs().mean().item())

            state, mask = next_state, next_mask

        epsilon = max(min_eps, epsilon * eps_decay)
        if ep % 10 == 0:
            agent.update_target_network()

        # Track Episode Evaluation
        ep_metrics = calculate_metrics(env.completed_processes)
        train_wait_history.append(ep_metrics["mean_wait_time"])
        q_magnitude_history.append(np.mean(ep_q_vals) if ep_q_vals else 0.0)

        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}/{episodes} | Mean Wait: {ep_metrics['mean_wait_time']:.2f} | Mean |Q|: {q_magnitude_history[-1]:.2f} | Epsilon: {epsilon:.2f}")

    # 4. Final Zero-Exploration Evaluation
    agent_test_metrics = []
    for w in test_workloads:
        # env.reset() returns (state, mask)
        state, mask = env.reset(w)
        done = False
        while not done:
            action = agent.select_action(state, mask, epsilon=0.0)
            
            # env.step() returns ((next_state, next_mask), reward, done, info)
            (next_state, next_mask), _, done, _ = env.step(action)
            state, mask = next_state, next_mask
            
        agent_test_metrics.append(calculate_metrics(env.completed_processes))

    agent_mean_wait = np.mean([m["mean_wait_time"] for m in agent_test_metrics])
    agent_fairness = np.mean([m["jain_fairness"] for m in agent_test_metrics])

    # 5. Output Summary Table
    print("\n================ FINAL COMPARATIVE BENCHMARK ================")
    print(f"{'Policy':<15} | {'Mean Wait Time':<15} | {'Jain Fairness Index':<20}")
    print("-" * 55)
    print(f"{'FCFS':<15} | {fcfs_mean_wait:<15.2f} | {fcfs_fairness:<20.4f}")
    print(f"{'SJF':<15} | {sjf_mean_wait:<15.2f} | {sjf_fairness:<20.4f}")
    print(f"{'Round Robin':<15} | {rr_mean_wait:<15.2f} | {rr_fairness:<20.4f}")
    print(f"{'Neural Agent':<15} | {agent_mean_wait:<15.2f} | {agent_fairness:<20.4f}")
    print("=" * 55)

    # 6. Generate Figures
    # Plot 1: Convergence
    plt.figure(figsize=(10, 5))
    window = 15
    smoothed_wait = np.convolve(train_wait_history, np.ones(window)/window, mode='valid')
    plt.plot(smoothed_wait, label="Neural Agent (Moving Avg)", color="blue")
    plt.axhline(y=fcfs_mean_wait, color='red', linestyle='--', label="FCFS Baseline")
    plt.axhline(y=sjf_mean_wait, color='green', linestyle='--', label="SJF Baseline")
    plt.axhline(y=rr_mean_wait, color='orange', linestyle='--', label="RR Baseline")
    plt.title("KernelMind: Direct Neural Scheduler Training Convergence")
    plt.xlabel("Episode")
    plt.ylabel("Mean Wait Time (Ticks)")
    plt.legend()
    plt.grid(True)
    plt.savefig("convergence_plot.png")
    plt.close()

    # Plot 2: Pareto Tradeoff Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter([fcfs_mean_wait], [fcfs_fairness], color='red', s=120, label='FCFS', zorder=5)
    plt.scatter([sjf_mean_wait], [sjf_fairness], color='green', s=120, label='SJF', zorder=5)
    plt.scatter([rr_mean_wait], [rr_fairness], color='orange', s=120, label='Round Robin', zorder=5)
    plt.scatter([agent_mean_wait], [agent_fairness], color='blue', marker='*', s=250, label='Neural Agent', zorder=6)
    
    plt.title("Pareto Tradeoff: Mean Wait Time vs. Jain Fairness")
    plt.xlabel("Mean Wait Time (Lower is Better)")
    plt.ylabel("Jain Fairness Index (Higher is Better)")
    plt.legend()
    plt.grid(True)
    plt.savefig("tradeoff_scatter.png")
    plt.close()
    print("Plots saved successfully.")

if __name__ == "__main__":
    main()