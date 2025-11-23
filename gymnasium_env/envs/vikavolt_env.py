from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from gymnasium_env.envs.simulation import Quadrotor
from gymnasium_env.envs.simulation import Utils

class VikavoltEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, mass=1.0, init_position = None, init_orientation = None, gate_positions = None,  dt=0.01):
        # we have a vector of 31 components representing the observation space
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0, -1.0] ,dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        )

        # We have 4 actions, corresponding to thrust, roll, pitch, yaw
        self.observation_space = spaces.Dict({
            "pos": spaces.Box(-np.inf, np.inf, (3,), np.float32),
            "vel": spaces.Box(-np.inf, np.inf, (3,), np.float32),
            "ori": spaces.Box(-np.inf, np.inf, (4,), np.float32),
            "R":   spaces.Box(-np.inf, np.inf, (9,), np.float32),
            "rel_gate_pos": spaces.Box(-np.inf, np.inf, (3,), np.float32),
            "rel_gate_R":   spaces.Box(-np.inf, np.inf, (9,), np.float32),
            "prev_action":  spaces.Box(-1, 1, (4,), np.float32)
        })

        # Quadrotor state
#        self.init_position = np.zeros(3) if init_position is None else init_position
        self.init_position = init_position
        self.position = self.init_position.copy()
        self.init_orientation = init_orientation
        self.orientation = self.init_orientation.copy()
        self.velocity = np.zeros(3)        
        self.R = np.eye(3)
        self.prev_action = np.zeros(4)

        self.gates = gate_positions

        self.current_gate_idx = 0
        #self.max_steps = max_steps
        self.step_count = 0
        self.collision_flag = False        

        self.lap_times = []
        self.lap_count = 0
        print(self.init_position)
        self.quadrotor = Quadrotor(mass=mass, initial_position=self.init_position, dt=dt)

    def _get_obs(self):
      #  gate = self.gates[self.current_gate_idx][:3]
        #rel_pos = gate - self.position
        rel_pos = np.zeros(3)
        # fake gate rotation for now
        rel_R = np.eye(3).flatten() # TODO placeholder for now

        return {
            "pos": self.position,
            "vel": self.velocity,
            "ori": self.orientation,
            "R": self.R.flatten(),
            "rel_gate_pos": rel_pos,
            "rel_gate_R": rel_R,
            "prev_action": self.prev_action,
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.position = self.init_position
        self.orientation = self.init_orientation
        self.velocity = np.zeros(3)
        self.R = np.eye(3)
        self.prev_action = np.zeros(4)
        self.step_count = 0
        self.current_gate_idx = 0
        self.collision_flag = False        

        observation = self._get_obs()

        return observation, {}


    def step(self, action):
        # call simulation step
        action = np.clip(action, self.action_space.low, self.action_space.high)
        state = self.quadrotor.step(action)
        self.position = state.position
        self.velocity = state.velocity
        self.orientation = state.orientation
        self.R = Utils.quat_to_rotation_matrix(state.orientation)
        #self.R = state.
        # obs = self.sim.step(action)
        # lap_times = self.lap_times
        # lap_count = self.lap_count

        self.step_count += 1
        
        self.prev_action = action.copy()

        terminated = False
        #if self.collision_flag:
        #    terminated = True        

        #gate_pos = self.gates[self.current_gate_idx][:3]
        #if np.linalg.norm(self.position - gate_pos) < 0.5:
        #    self.current_gate_idx += 1
        #    if self.current_gate_idx >= len(self.gates):
        #        terminated = True
            
        #truncated = self.step_count >= self.max_steps
        truncated = False
    
        #dist_to_gate = np.linalg.norm(self.position - gate_pos)
        dist_to_gate = 10
        reward = -dist_to_gate

        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        pass
