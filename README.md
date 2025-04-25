# 🛰️ PPO-Based Satellite Navigation and Debris Avoidance

This project applies **Proximal Policy Optimization (PPO)**, a reinforcement learning algorithm, to train a satellite agent in a custom 2D orbital environment. The agent learns to maintain a stable orbit, avoid space debris, and manage fuel efficiently — all within a physics-based simulation.

---

## 🚀 Features

- **Custom Orbital Environment:**  
  A 20x20 grid simulating orbital lanes, debris, and movement constraints.
  
- **Physics-Inspired Dynamics:**  
  Includes velocity, acceleration, fuel consumption, and collision detection.

- **Reinforcement Learning with PPO:**  
  Uses `MlpPolicy` from Stable-Baselines3 to train a neural network agent with continuous action space (thrust control).

- **Reward System:**  
  Encourages fuel efficiency and stable orbits while penalizing collisions and boundary violations.

- **Visualization:**  
  Tracks performance using reward logs and offers visualization for debugging and progress monitoring.

---

## 🧠 Why PPO?

PPO is ideal for environments with continuous control tasks like thrust management. It offers:
- Sample-efficient learning
- Stable policy updates
- Robust performance in dynamic environments

---

## 🧪 Installation

```bash
git clone https://github.com/your-username/satellite-ppo.git
cd satellite-ppo
pip install -r requirements.txt
```
---
## How to Run

python train_satellite_agent.py

---

## 📈 Future Work
🌍 Expand to 3D orbital simulations

📡 Integrate real-time satellite telemetry

⚡ Improve energy/fuel modeling

🧪 Test on hardware-in-the-loop systems

---

## 📌 Project Summary
PPO for real-time satellite control and collision avoidance in simulated space. Future extensions aim to bring this into real-world applications with improved realism and onboard testing capabilities.


