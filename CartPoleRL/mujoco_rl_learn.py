import os
import time
import numpy as np
import random

import mujoco
import mujoco.viewer as mujoco_viewer

import gymnasium as gym
from gymnasium import spaces


##from gymnasium.wrappers import FrameStackObservation

from typing import Tuple, Dict, Any, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


class CartPoleEnvSpec(gym.Env):
    """A class that serves as a model-agnostic specification and template."""
    metadata = {"render_modes": ["human"], "render_fps": 30}
    POLE_ANGLE_THRESHOLD_RADIANS: float = 0.3
    MAX_EPISODE_STEPS: int = 100000
    RAW_CART_POSITION_RANGE: Tuple[float, float] = (-5.5, 5.5)
    RAW_CART_VELOCITY_RANGE: Tuple[float, float] = (-2.0, 2.0)
    RAW_POLE_ANGLE_RANGE: Tuple[float, float] = (-0.3, 0.3)
    RAW_POLE_VELOCITY_RANGE: Tuple[float, float] = (-2.5, 2.5)

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        ###self.action_space = spaces.Discrete(2)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def _normalize_observation(self, raw_state: Tuple[float, ...]) -> np.ndarray:
        
        cart_pos, cart_vel, pole_ang, pole_vel = raw_state


        return np.array([cart_pos, cart_vel, pole_ang, pole_vel ], dtype=np.float32)

        def _normalize(val, min_v, max_v, clip=False):
            if clip: val = np.clip(val, min_v, max_v)
            return 2.0 * ((val - min_v) / (max_v - min_v)) - 1.0
        return np.array([
            _normalize(cart_pos, *self.RAW_CART_POSITION_RANGE),
            _normalize(cart_vel, *self.RAW_CART_VELOCITY_RANGE, clip=True),
            _normalize(pole_ang, *self.RAW_POLE_ANGLE_RANGE, clip=True),
            _normalize(pole_vel, *self.RAW_POLE_VELOCITY_RANGE, clip=True),
        ], dtype=np.float32)

# --- Concrete MuJoCo Implementation ---

class MuJoCoCartPoleEnv(CartPoleEnvSpec):
    MOTOR_SPEED = 5/20 # 1.0  20/20
    print(f"MOTOR_SPEED = {MOTOR_SPEED}")

    def __init__(self, xml_path: str, render_mode: Optional[str] = None):
        super().__init__()
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"MuJoCo XML file not found at: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)

        self.model.opt.timestep = 32/1000;  print(f'{self.model.opt.timestep}') #0.0001 

        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None
        # self.n_frames = 1
        self.step_counter = 0

        self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved
        self.episode_counter = 0


    def _get_raw_state(self) -> Tuple[float, float, float, float]:
        cart_position = self.data.qpos[0]
        cart_velocity = self.data.qvel[0]
        pole_angle = self.data.sensor('pole_angle_sensor').data[0]
        pole_angular_velocity = self.data.qvel[10]
        return (cart_position, cart_velocity, pole_angle, pole_angular_velocity)

    def _apply_action(self, action: int):
        ###self.data.ctrl[:] = -self.MOTOR_SPEED if action == 0 else self.MOTOR_SPEED
        action = np.clip(action, -1.0, 1.0)  # Ensure action is within expected bounds
        self.data.ctrl[:] = float(action[0]) * self.MOTOR_SPEED


    def _simulation_step(self):
        mujoco.mj_step(self.model, self.data)  # , nstep=self.n_frames   default 1

    ###def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:

        self._apply_action(action)
        self._simulation_step()
        self.step_counter += 1
        raw_state = self._get_raw_state()
        observation = self._normalize_observation(raw_state)
        cart_pos_raw, _, pole_angle_raw, _ = raw_state
        terminated = bool(abs(pole_angle_raw) > self.POLE_ANGLE_THRESHOLD_RADIANS) # or abs(cart_pos_raw) > self.RAW_CART_POSITION_RANGE[1]
        truncated = bool(self.step_counter >= self.MAX_EPISODE_STEPS)
        
        #reward = -10.0 if terminated else 1.0-  abs(pole_angle_raw)  # Reward shaping: penalize angle deviation
        reward = 1.0 -  abs(pole_angle_raw) -  (0.5*abs(cart_pos_raw))  / self.RAW_CART_POSITION_RANGE[1]  # Reward shaping: penalize angle deviation
        
        if self.render_mode == 'human': self.render()
        #print (f"Step {self.step_counter}: Action: {action}, Observation: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

        return observation, reward, terminated, truncated, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.episode_counter += 1
        self.episode_score_list.append(self.step_counter)
        
        print(f"-------\nEpisode: {self.episode_counter}   step_counter: {self.step_counter}  raw_state: {[round(x, 2) for x in self._get_raw_state()]}")
        
        
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_counter = 0

        #choice_tmp = self.MOTOR_SPEED*random.choice([-2.3,-1.2,-0.1,-0.01,0.01,0.1,1.2,2.3])
        #self.data.ctrl[:] = choice_tmp
        #for i in range(5):
        #    self._simulation_step()


        #init_qpos = np.zeros(self.model.nq)
        #    - inject a small random pole tilt (e.g. ±0.05 rad)
        #init_qpos[11] = self.np_random.uniform(low=-0.05, high=0.05)
        self.data.qpos[11] = self.np_random.uniform(low=-0.05, high=0.05)
        #    - small random velocities
        #init_qvel = self.np_random.normal(loc=0.0, scale=0.01, size=self.model.nv)
        #self.data.qvel[:] = init_qvel

        # 4) forward to compute sensors / collisions
        mujoco.mj_forward(self.model, self.data)

         # 5) warm up a few steps so velocities “settle”
        #for _ in range(10):
        #    self._simulation_step()


        # Get the actual state after the initial kick
        raw_initial_state = self._get_raw_state()
        # Normalize it to get the first observation
        initial_observation = self._normalize_observation(raw_initial_state)
        print(f"           Initial Observation: {initial_observation}")
        
        #return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32), {}
        return initial_observation, {}
    

    def render(self):
        if self.render_mode != 'human': return
        if self.viewer is None: self.viewer = mujoco_viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None



if __name__ == "__main__":
    start_time = time.time()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL_SAVE_PATH = os.path.join(script_dir, "mujoco_cartpole_ppo_model")  # MODEL_SAVE_PATH =  r"D:\Webots_examples\webots_mujoco\code_py\mujoco_cartpole_ppo_model"
    
    XML_FILE_PATH = os.path.join(script_dir, "four_wheels_webots_POLE.xml")  # XML_FILE_PATH = r"D:\Webots_examples\webots_mujoco\CartPole\model\four_wheels_webots_POLE.xml"
    TOTAL_TIMESTEPS = 2000000  # Increase for better performance

    
    env = MuJoCoCartPoleEnv(xml_path=XML_FILE_PATH)  # We create the environment without rendering for faster training
    ##env = FrameStackObservation(env, 4) # # stack the last 4 observations
    
    check_env(env)

    if os.path.exists(MODEL_SAVE_PATH + ".zip"):  ## --- 2. Create or Load the RL Model --- 
        print(f"\n--- Loading existing model from: {MODEL_SAVE_PATH}.zip ---")
        model = PPO.load(MODEL_SAVE_PATH, env=env)
    else:
        print(f"\n--- The file {MODEL_SAVE_PATH}.zip does not exist. Creating new model. ---")
        model = PPO('MlpPolicy', env, n_steps=20480, verbose=1)

    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    model.save(MODEL_SAVE_PATH) ;  print(f"\n--- Model saved to {MODEL_SAVE_PATH}  ---")

    #full_model_filename = MODEL_SAVE_PATH + ".zip"
    #absolute_model_path = os.path.abspath(full_model_filename)
    #print(f"\n--- Model saved to {absolute_model_path}  ---")
    
    env.close()
    print(f"\n--- Script execution time: {time.time() - start_time:.2f} seconds ---")
    

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'episode_score_list.tsv'), 'w') as file: # save files in the same folder where the Python script itself is located
        for score in env.episode_score_list:
            file.write(f"{score}\n")




'''
    # --- 5. Evaluate the Trained Agent ---
    print("\n--- Evaluating Trained Agent ---")
    # Create a new environment with rendering enabled to watch the agent
    eval_env = MuJoCoCartPoleEnv(xml_path=XML_FILE_PATH, render_mode="human")
    ##eval_env  = FrameStackObservation(eval_env , stack_size=4)

    # We don't need to load the model again if we just trained it, but this
    # shows how you would load it in a separate script for inference.
    # model = PPO.load(MODEL_SAVE_PATH)

    for episode in range(5):
        obs, info = eval_env.reset()
        done = False
        total_reward = 0
        step = 0
        while not done:
            # Use the model to predict the best action (deterministic=True)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            #print(f"{episode + 1} - Step {step + 1}: Action: {action}, Obs: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
            total_reward += reward
            step += 1
            done = terminated or truncated
            time.sleep(1 / 350) # Slow down rendering to be watchable

        print(f"\n---\nEvaluation Episode {episode + 1}: Finished after {step} steps. Total Reward: {total_reward}")

    eval_env.close()
    print("\n--- Demo Finished ---")
'''