import pygame 
import numpy as np
from gymnasium import spaces # type: ignore
import gymnasium as gym # type: ignore
from stable_baselines3 import PPO # type: ignore
from stable_baselines3.common.vec_env import DummyVecEnv # type: ignore
import matplotlib.pyplot as plt
import time
import random
import math
import os

# Constants
GRID_SIZE = 20
CELL_SIZE = 40
SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
SCREEN_HEIGHT = GRID_SIZE * CELL_SIZE
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
DARK_BLUE = (0, 0, 50)
STAR_COLOR = (200, 200, 255)
ORBIT_COLOR = (100, 100, 150)

MAX_VELOCITY = 1.0
MAX_ACCELERATION = 0.2
FUEL_CONSUMPTION_RATE = 0.05
INITIAL_FUEL = 200.0
COLLISION_DISTANCE = 0.3


class SatelliteEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(SatelliteEnv, self).__init__()
        self.grid_size = GRID_SIZE
        self.state = None
        self.velocity = np.array([0.0, 0.0])
        self.fuel = 50.0  # Start with 50 fuel
        
        # Start in the middle orbital lane
        self.start_pos = np.array([GRID_SIZE/2 + 5.0, GRID_SIZE/2])
        self.goal_pos = np.array([3*GRID_SIZE/4, 3*GRID_SIZE/4])
        
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.trajectory = []
        self.stars = [(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT)) 
                     for _ in range(200)]
        
        # Simple orbit tracking
        self.last_angle = 0.0
        self.orbit_direction = 1
        self.target_direction = 1
        self.orbits_completed = 0
        self.start_time = None
        self.max_episode_time = 10.0  # 10 seconds per episode

        # Initialize debris with random orbital parameters
        self.obstacle_pos = []
        center = np.array([GRID_SIZE/2, GRID_SIZE/2])
        
        # Define orbital lanes with some randomness
        debris_lanes = [3.0, 5.0, 7.0]
        debris_per_lane = [4, 5, 3]  # Different number of debris per lane
        
        for lane_idx, (base_radius, count) in enumerate(zip(debris_lanes, debris_per_lane)):
            # Add some randomness to the radius
            radius_variation = 0.3
            radius = base_radius + random.uniform(-radius_variation, radius_variation)
            
            # Random starting angles
            angles = sorted([random.uniform(0, 2 * math.pi) for _ in range(count)])
            
            for i, angle in enumerate(angles):
                # Add some randomness to the position
                pos_variation = 0.2
                pos_x = center[0] + radius * math.cos(angle) + random.uniform(-pos_variation, pos_variation)
                pos_y = center[1] + radius * math.sin(angle) + random.uniform(-pos_variation, pos_variation)
                
                # Random orbital speeds
                base_speed = 0.03
                speed_variation = 0.005
                orbit_speed = base_speed + random.uniform(-speed_variation, speed_variation)
                
                # Random sizes
                base_size = 0.3
                size_variation = 0.1
                size = base_size + random.uniform(-size_variation, size_variation)
                
                # Calculate initial velocity
                velocity_x = -orbit_speed * math.sin(angle)
                velocity_y = orbit_speed * math.cos(angle)
                
                self.obstacle_pos.append({
                    "pos": np.array([pos_x, pos_y], dtype=np.float32),
                    "velocity": np.array([velocity_x, velocity_y], dtype=np.float32),
                    "orbit_center": center,
                    "orbit_radius": radius,
                    "orbit_speed": orbit_speed,
                    "size": size,
                    "angle": angle
                })

        # Action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -MAX_VELOCITY, -MAX_VELOCITY, 0], dtype=np.float32),
            high=np.array([GRID_SIZE - 1, GRID_SIZE - 1, MAX_VELOCITY, MAX_VELOCITY, INITIAL_FUEL], dtype=np.float32),
            dtype=np.float32
        )
        self.reset()

    def reset(self, seed=None, **kwargs):
        self.state = self.start_pos.copy().astype(np.float32)
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.fuel = 50.0  # Start with 50 fuel
        self.trajectory = []
        self.last_angle = 0.0
        self.orbit_direction = 1
        self.target_direction = 1
        self.orbits_completed = 0
        self.start_time = None
        obs = np.concatenate([self.state, self.velocity, [self.fuel]], dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):
        if self.start_time is None:
            self.start_time = time.time()
        
        # Calculate orbital parameters
        center = np.array([GRID_SIZE/2, GRID_SIZE/2])
        distance_to_center = np.linalg.norm(self.state - center)
        
        # Define orbital lanes with strict boundaries
        orbital_lanes = [3.0, 5.0, 7.0]
        lane_boundaries = [(2.5, 3.5), (4.5, 5.5), (6.5, 7.5)]
        
        # Find current lane and target lane
        current_lane = None
        target_lane = None
        for i, (lower, upper) in enumerate(lane_boundaries):
            if lower <= distance_to_center <= upper:
                current_lane = orbital_lanes[i]
                break
        
        # If not in any lane, find nearest lane
        if current_lane is None:
            nearest_lane_idx = np.argmin([abs(distance_to_center - lane) for lane in orbital_lanes])
            current_lane = orbital_lanes[nearest_lane_idx]
        
        # Calculate current angle and orbital velocity
        current_angle = math.atan2(self.state[1] - center[1], self.state[0] - center[0])
        
        # Calculate proper orbital velocity for circular motion
        orbital_speed = 0.03  # Base orbital speed
        orbital_velocity = np.array([
            -orbital_speed * math.sin(current_angle),
            orbital_speed * math.cos(current_angle)
        ]) * 2 * self.target_direction
        
        # Apply thrust only in the direction of motion
        thrust = action * MAX_ACCELERATION
        velocity_component = np.dot(thrust, orbital_velocity) / np.linalg.norm(orbital_velocity)
        self.velocity = orbital_velocity * (1 + velocity_component)
        
        # Ensure velocity magnitude stays within bounds
        velocity_magnitude = np.linalg.norm(self.velocity)
        if velocity_magnitude > MAX_VELOCITY:
            self.velocity = self.velocity * (MAX_VELOCITY / velocity_magnitude)
        
        # Update position with proper circular motion
        self.state += self.velocity
        
        # Ensure satellite stays at the correct orbital radius
        current_radius = np.linalg.norm(self.state - center)
        if abs(current_radius - current_lane) > 0.1:  # If too far from target radius
            correction_factor = 0.1
            direction_to_center = (center - self.state) / current_radius
            self.state += direction_to_center * (current_lane - current_radius) * correction_factor
        
        # Update angle tracking
        angle_diff = (current_angle - self.last_angle) % (2 * math.pi)
        if angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        self.orbit_direction = 1 if angle_diff > 0 else -1
        self.last_angle = current_angle
        
        # Check for nearby debris and their relative positions
        debris_positions = []
        for obs in self.obstacle_pos:
            debris_pos = obs["pos"]
            debris_angle = math.atan2(debris_pos[1] - center[1], debris_pos[0] - center[0])
            angle_diff = (debris_angle - current_angle) % (2 * math.pi)
            
            if angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            
            distance = np.linalg.norm(self.state - debris_pos)
            debris_positions.append((distance, angle_diff, debris_pos, obs["orbit_radius"]))
        
        # Sort debris by distance
        debris_positions.sort(key=lambda x: x[0])
        
        # Check for debris in current lane
        debris_in_current_lane = False
        for distance, angle_diff, pos, radius in debris_positions:
            if abs(radius - current_lane) < 0.5 and distance < 4.0:
                debris_in_current_lane = True
                break
        
        # Handle lane changes for debris avoidance
        if debris_in_current_lane:
            # Determine safest lane to move to
            safe_lanes = []
            for lane in orbital_lanes:
                if lane != current_lane:
                    lane_safe = True
                    for distance, angle_diff, pos, radius in debris_positions:
                        if abs(radius - lane) < 0.5 and distance < 4.0:
                            lane_safe = False
                            break
                    if lane_safe:
                        safe_lanes.append(lane)
            
            if safe_lanes:
                target_lane = min(safe_lanes, key=lambda x: abs(x - current_lane))
            else:
                # If no safe lanes, choose the one with the least debris
                target_lane = min(orbital_lanes, key=lambda x: sum(1 for d, a, p, r in debris_positions 
                                                                  if abs(r - x) < 0.5 and d < 4.0))
        
        # Smooth lane transition
        if target_lane is not None:
            transition_speed = 0.1  # Slower transition for smoother motion
            current_radius = np.linalg.norm(self.state - center)
            new_radius = current_radius + (target_lane - current_radius) * transition_speed
            
            # Update position while maintaining circular motion
            self.state = center + (self.state - center) * (new_radius / current_radius)
            
            # Update velocity to match new radius
            orbital_velocity = np.array([
                -orbital_speed * math.sin(current_angle),
                orbital_speed * math.cos(current_angle)
            ]) * 2 * self.target_direction
            self.velocity = orbital_velocity
        
        # Update fuel - decrease with movement and thrust
        fuel_consumption = FUEL_CONSUMPTION_RATE * (np.linalg.norm(thrust) + 0.1)  # Base consumption + movement
        self.fuel = max(0, self.fuel - fuel_consumption)
        
        # Add current position to trajectory
        self.trajectory.append(self.state.copy())
        if len(self.trajectory) > 100:
            self.trajectory.pop(0)

        # Update debris positions
        for obs in self.obstacle_pos:
            obs["angle"] += obs["orbit_speed"]
            pos_variation = 0.1
            pos_x = obs["orbit_center"][0] + obs["orbit_radius"] * math.cos(obs["angle"]) + random.uniform(-pos_variation, pos_variation)
            pos_y = obs["orbit_center"][1] + obs["orbit_radius"] * math.sin(obs["angle"]) + random.uniform(-pos_variation, pos_variation)
            obs["pos"][0] = pos_x
            obs["pos"][1] = pos_y
            obs["velocity"] = np.array([-math.sin(obs["angle"]), math.cos(obs["angle"])]) * obs["orbit_speed"] * 2

        # Calculate distances
        min_debris_distance = min(np.linalg.norm(self.state - obs["pos"]) for obs in self.obstacle_pos)
        
        # Initialize reward components
        reward = 0.0
        done = False
        info = {}

        # Reward for maintaining proper orbital motion
        reward += 0.2  # Base reward for staying in orbit
        
        # Reward for proper orbital velocity
        velocity_error = abs(np.linalg.norm(self.velocity) - orbital_speed * 2)
        reward += 0.3 * (1 - velocity_error / (orbital_speed * 2))
        
        # Only stop on collision or complete out-of-bounds
        if min_debris_distance < COLLISION_DISTANCE:
            reward = -20.0
            done = True
            info["collision"] = True
        
        # Check if completely out of bounds (beyond recovery)
        if any(self.state < -2) or any(self.state > GRID_SIZE + 2):
            reward = -10.0
            done = True
            info["out_of_bounds"] = True

        # Keep within bounds but don't stop
        self.state = np.clip(self.state, 0, GRID_SIZE - 1)

        return np.concatenate([self.state, self.velocity, [self.fuel]], dtype=np.float32), reward, done, False, info

    def render(self):
        if self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                # Set window to not grab focus
                os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
                self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.NOFRAME)
                pygame.display.set_caption("Satellite Navigation")
                self.clock = pygame.time.Clock()

            # Draw space background
            self.screen.fill(DARK_BLUE)
            
            # Draw stars
            for star in self.stars:
                pygame.draw.circle(self.screen, STAR_COLOR, star, 1)

            # Draw orbit paths
            for obs in self.obstacle_pos:
                pygame.draw.circle(self.screen, ORBIT_COLOR, 
                                 (int(obs["orbit_center"][1] * CELL_SIZE), 
                                  int(obs["orbit_center"][0] * CELL_SIZE)),
                                 int(obs["orbit_radius"] * CELL_SIZE), 1)

            # Draw trajectory
            if len(self.trajectory) > 1:
                points = [(int(p[1] * CELL_SIZE), int(p[0] * CELL_SIZE)) for p in self.trajectory]
                pygame.draw.lines(self.screen, (100, 255, 100), False, points, 2)

            # Draw debris
            for obs in self.obstacle_pos:
                pos = (int(obs["pos"][1] * CELL_SIZE), int(obs["pos"][0] * CELL_SIZE))
                size = int(obs["size"] * CELL_SIZE)
                pygame.draw.circle(self.screen, RED, pos, size)
                pygame.draw.line(self.screen, YELLOW, pos,
                               (int(pos[0] + obs["velocity"][1] * CELL_SIZE * 2),
                                int(pos[1] + obs["velocity"][0] * CELL_SIZE * 2)), 2)

            # Draw the satellite
            sat_pos = (int(self.state[1] * CELL_SIZE), int(self.state[0] * CELL_SIZE))
            pygame.draw.circle(self.screen, GREEN, sat_pos, int(CELL_SIZE / 4))
            pygame.draw.circle(self.screen, WHITE, sat_pos, int(CELL_SIZE / 5), 2)
            pygame.draw.rect(self.screen, WHITE, 
                           (sat_pos[0] - int(CELL_SIZE / 3), sat_pos[1] - int(CELL_SIZE / 8),
                            int(CELL_SIZE / 1.5), int(CELL_SIZE / 4)), 2)
            pygame.draw.line(self.screen, YELLOW, sat_pos,
                           (int(sat_pos[0] + self.velocity[1] * CELL_SIZE * 2),
                            int(sat_pos[1] + self.velocity[0] * CELL_SIZE * 2)), 2)

            # Draw the space station
            station_pos = (int(self.goal_pos[1] * CELL_SIZE), int(self.goal_pos[0] * CELL_SIZE))
            pygame.draw.circle(self.screen, BLUE, station_pos, int(CELL_SIZE / 3))
            pygame.draw.circle(self.screen, WHITE, station_pos, int(CELL_SIZE / 4), 2)
            pygame.draw.rect(self.screen, WHITE, 
                           (station_pos[0] - int(CELL_SIZE / 2), station_pos[1] - int(CELL_SIZE / 6),
                            CELL_SIZE, int(CELL_SIZE / 3)), 2)
            for angle in [0, 90, 180, 270]:
                rad = math.radians(angle)
                port_pos = (int(station_pos[0] + math.cos(rad) * CELL_SIZE / 3),
                          int(station_pos[1] + math.sin(rad) * CELL_SIZE / 3))
                pygame.draw.circle(self.screen, WHITE, port_pos, int(CELL_SIZE / 8))

            # Draw fuel gauge
            fuel_width = int((self.fuel / INITIAL_FUEL) * SCREEN_WIDTH)
            pygame.draw.rect(self.screen, (50, 50, 50), (0, SCREEN_HEIGHT - 20, SCREEN_WIDTH, 20))
            pygame.draw.rect(self.screen, GREEN, (5, SCREEN_HEIGHT - 17, fuel_width - 10, 14))
            pygame.draw.rect(self.screen, WHITE, (0, SCREEN_HEIGHT - 20, SCREEN_WIDTH, 20), 1)

            # Update display without requiring focus
            pygame.display.flip()
            self.clock.tick(30)

            # Handle events without blocking
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None

def main():
    print("Initializing environment and training...")
    
    # Create and wrap environment
    env = SatelliteEnv(render_mode="human")
    env = DummyVecEnv([lambda: env])

    # Initialize PPO model with improved parameters
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        device="cpu"
    )
    
    # Training parameters
    total_timesteps = 30000  # Reduced to 30k
    timesteps_per_update = 1000
    rewards = []

    print("\nStarting training phase...")
    print(f"Total training steps: {total_timesteps}")
    print(f"Steps per update: {timesteps_per_update}")
    print("The agent will learn to navigate through the debris field...")
    print("The simulation will run in the background. You can switch to other windows.")
    
    # Training loop
    for i in range(total_timesteps // timesteps_per_update):
        model.learn(total_timesteps=timesteps_per_update, reset_num_timesteps=False)
        
        # Test the current policy
        obs = env.reset()
        episode_reward = 0
        done = False
        orbits_completed = 0
        
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            if "orbits_completed" in info:
                orbits_completed = info["orbits_completed"]
            env.render()
        
        rewards.append(episode_reward)
        progress = (i + 1) / (total_timesteps // timesteps_per_update) * 100
        print(f"Training progress: {progress:.1f}% - Update {i+1}/{total_timesteps//timesteps_per_update}")
        print(f"Episode reward: {episode_reward:.2f} - Orbits completed: {orbits_completed}")

    # Save the trained model
    model.save("satellite_rl_model")
    print("\nTraining completed! Model saved as 'satellite_rl_model'")

    # Plot training progress
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Training Progress')
    plt.xlabel('Update')
    plt.ylabel('Episode Reward')
    plt.savefig('training_progress.png')
    print("Training progress plot saved as 'training_progress.png'")

    # Continuous testing phase
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
    main() 