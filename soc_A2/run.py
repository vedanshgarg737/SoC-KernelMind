import os
import time
import numpy as np
from env import ProbeEnv
from agent import ProbeAgent

def render_probe_ascii(h, max_h, v, action, wind, step_count):
    os.system('clear' if os.name == 'posix' else 'cls')
    term_lines = 25
    
    if h > 150.0:
        display_max = max_h
        zoom_str = "[ CAMERA: WIDE ANGLE (1000m) ]"
    else:
        display_max = 150.0
        zoom_str = "[ CAMERA: TARGET APPROACH (150m) ]"
        
    pos = int((h / display_max) * term_lines)
    pos = max(0, min(term_lines, pos))
    
    wind_strs = ["Calm", "Gusty", "Adrian Gale"]
    thrust_str = "[####] ON" if action == 1 else "[    ] OFF"
    
    print(f" T+{step_count*0.1:5.1f}s | ALT: {h:6.1f}m | VEL: {v:7.1f}m/s | THRUST: {thrust_str} | WIND: {wind_strs[wind]}")
    print(f" {zoom_str}")
    print("-" * 75)
    
    for i in range(term_lines, -1, -1):
        if i == pos:
            print("        /\\         ")
            print("       /--\\   <-- spin-drive")
            if action == 1:
                print("       /WW\\   (THRUSTING)")
        elif i == 0:
            print("============================== [ TAUMOEBA TARGET ] ======================")
        else:
            if i % 5 == 0:
                print(f"  {int((i/term_lines)*display_max):4d}m |")
            else:
                print("       |")
    time.sleep(0.06)

def run_evaluation():
    env = ProbeEnv()
    agent = ProbeAgent()
    
    try:
        agent.q_table = np.load("q_table.npy")
    except FileNotFoundError:
        print("Error: q_table.npy missing. Run train.py first.")
        return
        
    raw_state = env.reset()
    state = agent.discretize_state(raw_state)
    done = False
    step_count = 0
    
    while not done:
        action = agent.choose_action(state, evaluation=True)
        render_probe_ascii(env.h, 1000.0, env.v, action, env.wind_idx, step_count)
        
        raw_state, reward, done, info = env.step(action)
        state = agent.discretize_state(raw_state)
        step_count += 1
        
    render_probe_ascii(env.h, 1000.0, env.v, 0, env.wind_idx, step_count)
    print("\n" + "="*40)
    print(f"MISSION TERMINATED | Final Status: {info['outcome'].upper()}")
    print(f"Final Altitude: {env.h:.2f} m | Final Impact Speed: {env.v:.2f} m/s")
    print("="*40)

if __name__ == "__main__":
    run_evaluation()