import sys
import os
import csv
import json
from datetime import datetime
from pathlib import Path
from itertools import product

import pandas as pd

import mujoco
import mujoco.viewer as mujoco_viewer
import numpy as np
import time

class TestOrchestrator():

    def __init__(self, scenario_file):
        
        print(f"Loading test suite from: {scenario_file}")
        with open(scenario_file, 'r') as f:
            self.config = json.load(f)        
        
        self.timestep = int(self.config["global_settings"]["TIME_STEP"]  )  # self.getBasicTimeStep()
        print(f"Simulation timestep: {self.timestep} ms")

        
        xml_path = r"D:\Webots_examples\webots_mujoco\CartPole\model\four_wheels_webots_POLE.xml" 
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = self.timestep/1000
        self.data = mujoco.MjData(self.model)
        

        self.viewer = mujoco_viewer.launch_passive(self.model, self.data)
        #self.viewer = mujoco_viewer.launch(self.model, self.data)

        print(f"__init__ {self.data}  ")


    def run_test_suite(self):  # Loads and runs all scenarios from a json config file.

        
        
        results_root =  (Path(__file__).parent / "test_results").resolve() ; results_root.mkdir(exist_ok=True) 



        for scenario_i, scenario in enumerate(self.config["scenarios"]):
            timeseries_data, status = self.run_simulation_loop(scenario )
            df_log = pd.DataFrame(timeseries_data, columns=['time_s', 'pole_angle_rad', 'cart_pos_m'])
            df_log.to_csv(  results_root / f"simulation_log_SUP_{scenario_i}.tsv"   , sep="\t", index=False)
            print(f"SUPER scenario: {scenario[ 'name']} completed.\n\n")
 


    def run_simulation_loop(self, scenario ):

        max_duration_s = scenario['duration_s']
        log = []
        status = "Completed"

        initial_angle = scenario.get('initial_angle', 0.0)
        wheel_velocity = scenario.get('wheel_velocity', 0.0) 

        pole_qpos_index = self.model.joint('pole_joint').qposadr[0]
        self.data.qpos[pole_qpos_index] = initial_angle


        self.data.ctrl[:] = wheel_velocity/20.0

        #self.data.qpos[2] = initial_angle  # Set initial pole angle
        #mujoco.mj_forward(self.model, self.data)  # Apply changes


        start_time = time.monotonic()

        
        i = 0
        while self.viewer.is_running():
            step_start = time.monotonic()


            current_time = time.monotonic() - start_time
            if current_time > max_duration_s:
                break


            pole_angle = self.data.sensor('pole_angle_sensor').data[0]
            cart_pos = self.data.qpos[0]            
            
            
            if i!=0:
                log.append([current_time, pole_angle, cart_pos])


            mujoco.mj_step(self.model, self.data)
            
            with self.viewer.lock():   # Example modification of a viewer option: toggle contact points every two seconds.
                self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.data.time % 2)
            
            self.viewer.sync()
            i += 1

            time_elapsed = time.monotonic() - step_start
            
            time_to_wait = self.model.opt.timestep - time_elapsed
            if time_to_wait > 0:  # If the loop finished faster than the timestep, wait for the remaining time
                #print(self.model.opt.timestep)
                time.sleep(time_to_wait)            
            
            #print(f"{round(pole_angle,2)}")
        
        return log, status



if __name__ == "__main__":

    #scenario_file = Path(__file__).parent / "test_scenarios.json"
    #scenario_file = Path(__file__).parent.parent / "test_scenarios.json"

    scenario_file = Path(r"D:\Webots_examples\CartPole\controllers\test_scenarios.json")
    orchestrator = TestOrchestrator(scenario_file)
    orchestrator.run_test_suite()