# Dual Digital Twin RL Assets: MuJoCo CartPole-on-Car

## 🌟 Project Overview

This repository hosts the MuJoCo-specific components of the **Dual Digital Twin Framework** described in the paper *A Dual Digital Twin Framework for Reinforcement Learning: Bridging Webots and MuJoCo with Generative AI and Alignment Strategies*.

The core objective is to provide a validated MuJoCo environment for the CartPole-on-Car system that is aligned with a corresponding Webots model. This rigorous sim-to-sim alignment ensures that RL policies trained in the performant MuJoCo environment can be seamlessly transferred to the more realistic/validated Webots environment.

## ⚙️ Repository Structure

The files are organized into subdirectories reflecting the two main phases of the framework: **Alignment** and **Reinforcement Learning**.


## 🚀 Setup and Installation

This project requires a working installation of **MuJoCo** and related Python libraries.

1.  **Clone the repository:**
    ```bash
    git clone [your-repo-url]
    cd [your-repo-name]
    ```

2.  **Dependencies:**
    Install the necessary Python packages. This project utilizes `mujoco`, `gymnasium` (for the RL environment), and `stable-baselines3`.
    ```bash
    pip install mujoco gymnasium stable-baselines3 numpy pandas
    ```
    *(Note: Ensure your MuJoCo installation and license (if required) are correctly set up.)*

## 🔬 Usage: Alignment & RL

### 1. Digital Twin Alignment (CartPole_orchestrator)

The `test_orchestrator_MuJoCo.py` script is used to execute defined test scenarios on the MuJoCo twin. The resulting logs are then used to quantify the divergence against the Webots logs, as part of the iterative alignment loop (Figure 7).

* **Model File:** The `four_wheels_webots_POLE.xml` file is the physics model loaded by the orchestrator.
* **Execution:** Run the orchestrator with your scenario configuration:
    ```bash
    python CartPole_orchestrator/test_orchestrator_MuJoCo.py
    ```

### 2. Reinforcement Learning (CartPoleRL)

The scripts in `CartPoleRL/` define a custom MuJoCo CartPole environment compatible with **Gymnasium** and **Stable-Baselines3**.

* **Training/Inference:** These scripts handle the primary policy training.
    * Use `mujoco_rl_load_predict.py` for continuous control (e.g., applying a force magnitude).
    * Use `mujoco_rl_load_predict_2D.py` for discrete control (e.g., applying max left or max right force).

* **Example Execution (Inference):**
    ```bash
    python CartPoleRL/mujoco_rl_load_predict.py
    ```
    This script loads a pre-trained PPO model and evaluates it in the MuJoCo environment with human rendering enabled.

## 📄 Reference

If you use this work or the underlying framework, please cite the associated paper:

*A Dual Digital Twin Framework for Reinforcement Learning: Bridging Webots and MuJoCo with Generative AI and Alignment Strategies*
