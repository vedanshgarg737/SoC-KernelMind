import numpy as np

class ProbeEnv:
    def __init__(self):
        self.m = 1000.0                
        self.g = 13.7                  
        self.R_adrian = 10700000.0     
        self.F_thrust_max = 25000.0    
        self.dt = 0.1                  
        
        self.wind_multipliers = {0: 1.0, 1: 1.5, 2: 2.5}
        
        self.P_wind = np.array([
            [0.80, 0.15, 0.05],  
            [0.20, 0.60, 0.20],  
            [0.05, 0.35, 0.60]   
        ])
        
        self.max_steps = 2000          
        self.reset()

    def reset(self):
        self.h = 1000.0                
        self.v = 0.0                   
        self.wind_idx = 0              
        self.step_count = 0
        return (self.h, self.v, self.wind_idx)

    def step(self, action):
        self.step_count += 1
        
        self.wind_idx = np.random.choice([0, 1, 2], p=self.P_wind[self.wind_idx])
        k_drag_eff = 2.0 * self.wind_multipliers[self.wind_idx]
        
        F_gravity = -self.m * self.g * (1.0 - (self.h / self.R_adrian))
        F_thrust = self.F_thrust_max if action == 1 else 0.0
        F_drag = k_drag_eff * (self.v ** 2) * np.sign(-self.v)
        
        F_net = F_gravity + F_thrust + F_drag
        
        a = F_net / self.m
        self.v += a * self.dt
        self.h += self.v * self.dt
        
        done = False
        reward = 0.0
        info = {"outcome": "running"}
        
        if action == 1:
            reward -= 1.5  
        else:
            reward -= 0.1  
            
        if self.h <= 0:
            done = True
            self.h = 0.0
            if self.v >= -3.0:  
                reward += 1000.0
                info["outcome"] = "success"
            else:
                reward -= 500.0 + (abs(self.v) * 20.0)
                info["outcome"] = "crushed"
                
        elif self.h > 1200.0:
            done = True
            reward -= 1000.0
            info["outcome"] = "runaway"
        elif self.step_count >= self.max_steps:
            done = True
            reward -= 1000.0
            info["outcome"] = "timeout"
            
        return (self.h, self.v, self.wind_idx), reward, done, info