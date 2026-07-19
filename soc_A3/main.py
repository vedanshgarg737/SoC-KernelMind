import numpy as np
import matplotlib.pyplot as plt
import copy
from core.process import generate_workload
from core.simulator import run_fcfs, run_sjf, run_rr
from core.environment import MetaSchedulerEnv
from agent.q_learning import TabularQLearner
from utils.metrics import calculate_metrics

def evaluate_baseline_static(baseline_func, evaluation_workloads, **kwargs):
    means, p90s, jains = [], [], []
    for wl in evaluation_workloads:
        completed = baseline_func(wl, **kwargs)
        m, p, j = calculate_metrics(completed)
        means.append(m)
        p90s.append(p)
        jains.append(j)
    return np.mean(means), np.mean(p90s), np.mean(jains)

def evaluate_random_agent(evaluation_workloads, env):
    means, p90s, jains = [], [], []
    for wl in evaluation_workloads:
        state = env.reset(wl)
        done = False
        while not done:
            action = np.random.randint(env.action_space_size)
            state, _, done = env.step(action)
        m, p, j = calculate_metrics(env.completed)
        means.append(m)
        p90s.append(p)
        jains.append(j)
    return np.mean(means), np.mean(p90s), np.mean(jains)

def evaluate_trained_agent(agent, evaluation_workloads, env):
    means, p90s, jains = [], [], []
    for wl in evaluation_workloads:
        state = env.reset(wl)
        done = False
        while not done:
            action = agent.choose_action(state, exploit_only=True)
            state, _, done = env.step(action)
        m, p, j = calculate_metrics(env.completed)
        means.append(m)
        p90s.append(p)
        jains.append(j)
    return np.mean(means), np.mean(p90s), np.mean(jains)

if __name__ == "__main__":
    np.random.seed(42)
    NUM_TRAIN_EPISODES = 30000
    NUM_EVAL_WORKLOADS = 500
    TAX_OVERHEAD = 1
    
    print("⚡ Generating validation workloads...")
    eval_workloads = [generate_workload(10, "uniform") for _ in range(NUM_EVAL_WORKLOADS)]
    
    print("🚀 Computing Baseline Heuristics Performance Benchmark Profiles...")
    rand_env = MetaSchedulerEnv(context_switch_tax=TAX_OVERHEAD)
    m_rand, p_rand, j_rand = evaluate_random_agent(eval_workloads, rand_env)
    m_fcfs, p_fcfs, j_fcfs = evaluate_baseline_static(run_fcfs, eval_workloads)
    m_sjf, p_sjf, j_sjf = evaluate_baseline_static(run_sjf, eval_workloads)
    m_rrs, p_rrs, j_rrs = evaluate_baseline_static(run_rr, eval_workloads, quantum=2, context_switch_tax=TAX_OVERHEAD)
    m_rrl, p_rrl, j_rrl = evaluate_baseline_static(run_rr, eval_workloads, quantum=8, context_switch_tax=TAX_OVERHEAD)
    
    # Init environment and training pipeline structures
    env = MetaSchedulerEnv(context_switch_tax=TAX_OVERHEAD)
    agent = TabularQLearner(state_dims=env.state_dims, action_size=env.action_space_size)
    
    moving_avg_history = []
    window_size = 500
    recent_performance = []
    
    print(f"\n⚙️ Commencing Training Pipeline Execution Loop ({NUM_TRAIN_EPISODES} Tracks)...")
    for ep in range(1, NUM_TRAIN_EPISODES + 1):
        wl = generate_workload(10, "uniform")
        state = env.reset(wl)
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            
        m, _, _ = calculate_metrics(env.completed)
        recent_performance.append(m)
        if len(recent_performance) > window_size:
            recent_performance.pop(0)
        moving_avg_history.append(np.mean(recent_performance))
        
        agent.decay_epsilon(ep, NUM_TRAIN_EPISODES)
        
        if ep % 5000 == 0:
            print(f" -> Completed Episode Block {ep}/{NUM_TRAIN_EPISODES} | Moving Mean Window Wait: {moving_avg_history[-1]:.2f}")

    # Evaluate Final Performance Matrix
    m_agent, p_agent, j_agent = evaluate_trained_agent(agent, eval_workloads, env)
    
    print("\n" + "="*70)
    print("📊 FINAL COMPARATIVE PERFORMANCE HIGHLIGHTS MATRIX")
    print("="*70)
    print(f"{'Scheduler Strategy Pattern':<30} | {'Mean Wait':<10} | {'P90 Wait':<10} | {'Jain Index':<10}")
    print("-"*70)
    print(f"{'Random Agent Baseline':<30} | {m_rand:<10.2f} | {p_rand:<10.2f} | {j_rand:<10.3f}")
    print(f"{'FCFS Standard Static':<30} | {m_fcfs:<10.2f} | {p_fcfs:<10.2f} | {j_fcfs:<10.3f}")
    print(f"{'SJF Non-Preemptive Ideal':<30} | {m_sjf:<10.2f} | {p_sjf:<10.2f} | {j_sjf:<10.3f}")
    print(f"{'Round-Robin Short (q=2)':<30} | {m_rrs:<10.2f} | {p_rrs:<10.2f} | {j_rrs:<10.3f}")
    print(f"{'Round-Robin Long (q=8)':<30} | {m_rrl:<10.2f} | {p_rrl:<10.2f} | {j_rrl:<10.3f}")
    print(f"{'KernelMind RL Meta-Scheduler':<30} | {m_agent:<10.2f} | {p_agent:<10.2f} | {j_agent:<10.3f}")
    print("="*70)

    # Secondary Challenge Execution Frame: The Workload Stress Test
    print("\n🚨 Initiating Secondary Objective Stress Test Validation Pipeline: I/O Storm...")
    storm_workloads = [generate_workload(10, "io_storm") for _ in range(NUM_EVAL_WORKLOADS)]
    ms_sjf, ps_sjf, js_sjf = evaluate_baseline_static(run_sjf, storm_workloads)
    ms_rrs, ps_rrs, js_rrs = evaluate_baseline_static(run_rr, storm_workloads, quantum=2, context_switch_tax=TAX_OVERHEAD)
    ms_agent, ps_agent, js_agent = evaluate_trained_agent(agent, storm_workloads, env)
    
    print("-"*70)
    print(f"{'SJF Under Storm Conditions':<30} | {ms_sjf:<10.2f} | {ps_sjf:<10.2f} | {js_sjf:<10.3f}")
    print(f"{'RR-Short Under Storm Conditions':<30} | {ms_rrs:<10.2f} | {ps_rrs:<10.2f} | {js_rrs:<10.3f}")
    print(f"{'KernelMind Agent Under Storm':<30} | {ms_agent:<10.2f} | {ps_agent:<10.2f} | {js_agent:<10.3f}")
    print("="*70)

    # Build Visualization Matrix Map
    plt.figure(figsize=(12, 6))
    plt.plot(moving_avg_history, label='KernelMind Agent (Moving Avg Window = 500)', color='purple', lw=2)
    plt.axhline(y=m_rand, color='red', linestyle='--', alpha=0.7, label='Random Agent Baseline')
    plt.axhline(y=m_fcfs, color='orange', linestyle='--', alpha=0.7, label='FCFS Baseline')
    plt.axhline(y=m_sjf, color='green', linestyle='--', alpha=0.7, label='SJF Non-Preemptive Baseline')
    plt.axhline(y=m_rrs, color='blue', linestyle='--', alpha=0.7, label='RR-Short (q=2) Baseline')
    plt.title("KernelMind Meta-Scheduler Architecture Convergence Trajectory Analysis Graph")
    plt.xlabel("Training Runs (Episodes)")
    plt.ylabel("System Mean Process Wait Latency (Ticks)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.savefig("convergence_plot.png", dpi=300)
    print("\n💾 Convergence visual plot saved successfully to workspace disk filesystem as 'convergence_plot.png'.")