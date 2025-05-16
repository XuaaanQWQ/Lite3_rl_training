# RL-Lite3

[简体中文](./README_ZH.md)




- [RL-Lite3](#rl-lite3)
- [Introduction](#introduction)
  - [System Architecture](#system-architecture)
  - [Actor Network and Critic Network](#actor-network-and-critic-network)
    - [Network Structure](#network-structure)
    - [Observations (`obs_buf = 117`)](#observations-obs_buf--117)
    - [Privileged Observations (`privileged_obs_buf = 54`)](#privileged-observations-privileged_obs_buf--54)
    - [Network Input](#network-input)
      - [Environment Encoder](#environment-encoder)
    - [Network Output](#network-output)
  - [Proximal Policy Optimization](#proximal-policy-optimization)
    - [Policy Gradient and Generalized Advantage Estimation](#policy-gradient-and-generalized-advantage-estimation)
    - [PPO —— Clip the Gradient](#ppo--clip-the-gradient)
    - [On Policy Runner](#on-policy-runner)
- [Software architecture](#software-architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Train policy in the simulation](#train-policy-in-the-simulation)
  - [Run controller in the simulation](#run-controller-in-the-simulation)
  - [Run controller in the real-world](#run-controller-in-the-real-world)
  - [Reference](#reference)



# Introduction
A Learning-based locomotion controller for quadruped robots. It includes all components needed for training and hardware deployment on DeepRobotics Lite3.

## System Architecture



![alt text](./doc/system_architecture.png)

This project is based on the **Actor-Critic framework**:

- The **actor network** serves as the core **policy network**, taking the system's observation as input and outputting the **mean of desired joint positions**. These outputs are used as feedforward commands for the **low-level controller**, which executes the actual motor control.

- The **critic network** takes in **privileged observations** of the system and outputs the **advantage estimate**, which is then used by the **PPO algorithm module** to compute the policy gradient and assist in training the actor network.

The project uses the **Isaac Gym** simulator to provide a high-performance environment for interaction, and utilizes a `RolloutStorage` object to buffer and organize trajectory data collected during rollouts.

## Actor Network and Critic Network


![alt text](doc/Network_architecture.png)
The system involves a total of **four neural networks (MLPs)** throughout training and deployment.  
These include the core **actor network** and **critic network**, as well as the **environment factor encoder**, which infers environment-specific information from privileged observations, and the **adaptation module**, which replaces the privileged observations during real-world deployment by leveraging historical observations.



### Network Structure


Actor MLP

| Layer Index | Layer Type | Input Dim | Output Dim | Activation |
|-------------|------------|-----------|------------|------------|
| 0           | Linear     | 135       | 512        | –          |
| 1           | ELU        | –         | –          | α = 1.0    |
| 2           | Linear     | 512       | 256        | –          |
| 3           | ELU        | –         | –          | α = 1.0    |
| 4           | Linear     | 256       | 128        | –          |
| 5           | ELU        | –         | –          | α = 1.0    |
| 6           | Linear     | 128       | 12         | –          |




Critic MLP

| Layer Index | Layer Type | Input Dim | Output Dim | Activation |
|-------------|------------|-----------|------------|------------|
| 0           | Linear     | 135       | 512        | –          |
| 1           | ELU        | –         | –          | α = 1.0    |
| 2           | Linear     | 512       | 256        | –          |
| 3           | ELU        | –         | –          | α = 1.0    |
| 4           | Linear     | 256       | 128        | –          |
| 5           | ELU        | –         | –          | α = 1.0    |
| 6           | Linear     | 128       | 1          | –          |

**Note:** The above represents the default network architecture used in this project.  
Depending on the network inputs and task requirements, the architecture can be customized to achieve optimal performance.




### Observations (`obs_buf = 117`)



| Feature                         | Description                                  | Dim |
|----------------------------------|----------------------------------------------|-----|
| `commands[:, :3]`               | Desired `[x_vel, y_vel, yaw_vel]` commands   | 3   |
| `rpy`                           | Base orientation (roll, pitch, yaw)          | 3   |
| `base_ang_vel`                  | Base angular velocity                         | 3   |
| `dof_pos`                       | Current joint positions (12 DOFs)            | 12  |
| `dof_vel`                       | Current joint velocities                      | 12  |
| `dof_pos_history`              | Joint positions from last 3 timesteps         | 36  |
| `dof_vel_history`              | Joint velocities from last 2 timesteps        | 24  |
| `action_history` (dof targets) | Past 2 action outputs                         | 24  |
| **Total**                       |                                              | **117** |


### Privileged Observations (`privileged_obs_buf = 54`)



| Feature                      | Description                                                   | Dim |
|------------------------------|---------------------------------------------------------------|-----|
| `contact_states`            | Binary flags for foot contacts (thresholded sensor_force)     | 4   |
| `friction_coefficients`     | Ground friction under each foot                               | 4   |
| `push_forces` / `push_torques` | External disturbances applied to base (force + torque)     | 6   |
| `mass_payloads - 6`         | Payload perturbation from nominal value                       | 1   |
| `com_displacements`         | Displacement of center of mass from nominal                   | 3   |
| `motor_strengths - 1`       | Multipliers for actuator strength (per joint)                 | 12  |
| `Kp_factors - 1`            | Proportional gain multipliers                                 | 12  |
| `Kd_factors - 1`            | Derivative gain multipliers                                   | 12  |
| **Total**                   |                                                               | **54** |


### Network Input

In this project, the input of both actor and critic networks  is the same, which consists of **135 dimensions** in total.  
The network input is composed of **117 dimensions from the observation** and **18 dimensions of environment latent features** generated by passing the privileged observation through an environment encoder (MLP).

#### Environment Encoder

The **Environment Factor Encoder** is used to transform the privileged observations into a low-dimensional **environment embedding vector** (latent), which serves as an additional input to the actor network.  
This enables the policy to generate adaptive actions based on environment-specific features.

    [ **privileged_obs (54)** ] → Used only during training 

                  ↓
  
    [ **env_factor_encoder (MLP)** ] → Encodes environment

                  ↓ 
         
    [ **latent_env_embedding (18D)** ]  

                  ↓ 

    [ **obs (117D) + latent (18D)** ] → Concatenated as input

                  ↓ 

            [ **actor MLP** ]   

                  ↓ 

    [ **action distribution** ] → Mean and std of joint targets






Environment Factor Encoder
| Layer Index | Layer Type | Input Dim | Output Dim | Activation |
|-------------|------------|-----------|------------|------------|
| 0           | Linear     | 54        | 256        | –          |
| 1           | ELU        | –         | –          | α = 1.0    |
| 2           | Linear     | 256       | 128        | –          |
| 3           | ELU        | –         | –          | α = 1.0    |
| 4           | Linear     | 128       | 18         | –          |






However, since **privileged observations are not available during real-world deployment**, an additional module called the **online encoder** (also known as the **adaptation module**) is introduced to fulfill the role of the environment encoder during deployment.

The adaptation module takes **observable historical data** as input. In this project, it is implemented as a trajectory of observations over the past **40 time steps**, resulting in an input size of **117 × 40 = 4680 dimensions**.

Adaptation Module
| Layer Index | Layer Type | Input Dim | Output Dim | Activation |
|-------------|------------|-----------|------------|------------|
| 0           | Linear     | 4680        | 256        | –          |
| 1           | ELU        | –         | –          | α = 1.0    |
| 2           | Linear     | 256       | 32        | –          |
| 3           | ELU        | –         | –          | α = 1.0    |
| 4           | Linear     | 32       | 18         | –          |


During training, the adaptation module learns to **approximate the output of the environment encoder**, so that it can replace it when privileged information is not accessible.






### Network Output

The output `action` of the actor network represents the **mean of desired joint positions**, denoted as `self.transition.action_mean`. Together with `self.transition.action_sigma`, it defines a **Gaussian distribution** over the desired joint angles. An actual action is sampled from this distribution and then used to compute joint torques according to the type of low-level controller (e.g., PD control or P control).

This structure follows the standard **stochastic policy** design. By introducing sampling-based randomness into the policy, the agent is encouraged to explore a wider range of state-action pairs, helping to avoid getting stuck in local optima.


The output of the critic network is `self.transition.values`.These values are later used during the PPO update process to compute the policy gradient. This process will be elaborated in the upcoming section on PPO and Generalized Advantage Estimation (GAE).







## Proximal Policy Optimization

### Policy Gradient and Generalized Advantage Estimation

In this project, the policy is essentially a neural network, the training process is about optimizing the network parameters to maximize the total reward:


![alt text](doc/PG1.png)

![alt text](doc/PG2.png)

Where $\hat{A}_t$ is the Advantage function, measuring how much an action is better than the average action. In this project, the Advantage $\hat{A}_t$ is defined as:

![alt text](doc/Advantage.png)

Where $δ_t$ is the Temporal Difference error, define as:

![alt text](doc/TD.png)


In supervised learning, we typically need a loss function to compute the error gradient for each
parameter, and the parameters are updated in the direction that minimizes the loss. In this project, we construct a **surrogate loss function** by taking the negative of the **total reward**.

This surrogate loss is then used as the loss function during training, and gradients are propagated backward to update the network parameters.

In essence, **minimizing the surrogate loss is equivalent to maximizing the total reward**, thereby guiding the agent to learn a more optimal policy.





### PPO —— Clip the Gradient
The core idea of **Proximal Policy Optimization (PPO)** is to address the issue of **catastrophic update** that arises in vanilla policy gradient methods. 

In standard policy gradient algorithms, large policy updates can cause the new policy to deviate significantly from the old one, often leading to performance degradation.

To address this, the **Natural Policy Gradient** method treats the policy as a probability distribution and updates it in the “most natural” direction — one that considers the **Kullback-Leibler (KL) divergence** between the new and old policies:

![alt text](doc/PPO1.png)

However, this requires estimating or inverting the **Fisher Information Matrix**, which is computationally expensive.

Later, **Trust Region Policy Optimization (TRPO)** simplified the natural gradient approach by formulating the update as a constrained optimization problem with a KL divergence constraint:

![alt text](doc/PPO2.png)


While more stable, TRPO still involves complex implementation and high computational cost.

**PPO** is a further simplification of TRPO. It achieves the same goal — **restricting the policy update step size** — using a much simpler mechanism known as the **Clipped Surrogate Objective**, which is what we adopt in this project.


![alt text](doc/PPO3.png)



### On Policy Runner

During the entire training process, the **on-policy runner** iteratively updates the parameters of three neural networks — excluding the `env_factor_encoder` — through backpropagation using Adam optimizer.

The parameters of the **env_factor_encoder** remain fixed throughout training. It serves solely as a **teacher network**, providing supervision signals during the training phase.

The loss functions for the other three networks are defined as follows:

Actor Network Surrogate Loss:

![alt text](doc/surrogateloss.png)

Critic Network Value Loss:

![alt text](doc/ValueLoss.png)

Adaptation Module Loss:

![alt text](doc/AdaptationLoss.png)


For detailed implementation of loss definitions and network updates, see:

`Lite3_rl_training/rsl_rl/rsl_rl/algorithms/ppo.py` → `Update()` function




# Software architecture
This repository consists of below directories:
- rsl_rl: a package wrapping RL methods.
- legged_gym: gym-style environments of quadruped robots.


# Installation
1.  Create a python (3.6/3.7/3.8, 3.8 recommended) environment on Ubuntu OS.

2.  Install pytorch with cuda.
```
# pytorch
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

3.  Download Isaac Gym (version >=preview 3) from the official website and put it into the root directory of the project.

4. Install python dependencies with pip.
```
pip3 install transformations matplotlib gym tensorboard numpy=1.23.5
```

5. Install legged_gym and rsl_rl by pip
```
cd legged_gym
pip install -e .

cd rsl_rl
pip install -e .
```

6. Install pandas for visualization
```
pip install pandas

```

# Usage

## Train policy in the simulation
```
cd ${PROJECT_DIR}
python3 legged_gym/legged_gym/scripts/train.py --rl_device cuda:0 --sim_device cuda:0 --headless
```

## Run controller in the simulation
```
cd ${PROJECT_DIR}
python3 legged_gym/legged_gym/scripts/play.py --rl_device cuda:0 --sim_device cuda:0 --load_run ${model_dir} --checkpoint ${model_name}
```
Check that your computer has a GPU, otherwise, replace the word `cuda:0` with `cpu`.
You should assign the path of the network model via `--load_run` and `--checkpoint`. 

## Run controller in the real-world

Copy your policy file to the project [rl_deploy](https://github.com/DeepRoboticsLab/Lite3_rl_deploy.git),then you can run your reinforcement learning controller in the real world





## Reference
- [legged_gym](https://github.com/leggedrobotics/legged_gym.git)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
- [quadruped-robot](https://gitee.com/HUAWEI-ASCEND/quadruped-robot.git)
  
<a name="ref1">[1]</a> Viktor Makoviychuk, Lukasz Wawrzyniak, Yunrong Guo, Michelle Lu, Kier Storey, Miles Macklin, David Hoeller, Nikita Rudin, Arthur Allshire, Ankur Handa, and Gavriel State. *"Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning."* arXiv preprint [arXiv:2108.10470](https://arxiv.org/abs/2108.10470), 2021.
  
[Communication](https://www.deeprobotics.cn/en/index/company.html#maps)
