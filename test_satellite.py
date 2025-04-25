import pygame
import numpy as np
from gymnasium import spaces
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import time
import os

# Import the environment
from satellite_navigation import SatelliteEnv

def test_model():
    print("Loading saved model and starting testing phase...")
    
    # Create environment
    env = SatelliteEnv(render_mode="human")
    env = DummyVecEnv([lambda: env])
    
    # Load the saved model
    model = PPO.load("satellite_rl_model")
    
    print("\nStarting continuous testing phase...")
    print("The simulation will run continuously until you press Ctrl+C")
    print("The satellite will navigate through the debris field...")
    
    try:
        while True:  # Run indefinitely until interrupted
            obs = env.reset()
            done = False
            total_reward = 0
            orbits_completed = 0
            
            while not done:
                action, _ = model.predict(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward[0]
                if "orbits_completed" in info:
                    orbits_completed = info["orbits_completed"]
                env.render()
                time.sleep(0.1)  # Small delay for visualization
            
            print(f"Episode complete. Total reward: {total_reward:.2f} - Orbits completed: {orbits_completed}")
            
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    
    env.close()

if __name__ == "__main__":
    test_model() 