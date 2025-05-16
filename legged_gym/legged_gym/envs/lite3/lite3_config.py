from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Lite3RoughCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]  # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_HipX_joint': 0.0,   # [rad]
            'HL_HipX_joint': 0.0,   # [rad]
            'FR_HipX_joint': -0.0,  # [rad]
            'HR_HipX_joint': -0.0,   # [rad]

            'FL_HipY_joint': -1.5,     # [rad] -0.7 -1.5
            'HL_HipY_joint': -1.5,   # [rad] 
            'FR_HipY_joint': -1.5,     # [rad]
            'HR_HipY_joint': -1.5,   # [rad]

            'FL_Knee_joint': 2.0,   # [rad]   1.5 2.0
            'HL_Knee_joint': 2.0,    # [rad]
            'FR_Knee_joint': 2.0,  # [rad]
            'HR_Knee_joint': 2.0,    # [rad]
        }

    class env(LeggedRobotCfg.env):
        num_envs = 4096  # number of parallel environments
        num_observations = 117  # {133, 320}
        num_privileged_obs = 54  # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_observation_history = 40
        episode_length_s = 20  # episode length in seconds
        curriculum_factor = 0.8

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.0}  # 27 20 17 # [N*m/rad]
        damping = {'joint': 0.7}  # 1.0 0.7 [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        use_torch_vel_estimator = False
        use_actuator_network = False
        use_pmtg = False

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/lite3/urdf/Lite3.urdf'
        name = "Lite3"
        foot_name = "FOOT"
        shoulder_name = "HIP"  # urdf 里面没shoulder
        # penalize_contacts_on = ["THIGH", "shoulder", "SHANK"]
        penalize_contacts_on = ["THIGH", "SHANK"]
        # terminate_after_contacts_on = ["TORSO", "shoulder"]
        terminate_after_contacts_on = ["TORSO"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        restitution_mean = 0.5
        restitution_offset_range = [-0.1, 0.1]
        compliance = 0.5

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        # 0.36
        base_height_target = 0.30  
        still_all = True
        only_positive_rewards = False
        pitch_roll_factor = [1, 1]

        class scales(LeggedRobotCfg.rewards.scales):
            # -4.0
            lin_vel_z = -2.0 
            # -0.05 -0.025
            ang_vel_xy = -0.025 
            # -0.5
            orientation = -0.025 
            # -1.0
            base_height = -0.3 
            #  -2.5e-5
            torques = -2.5e-5 
            # -0.0005
            dof_vel = -0.0 
            
            # torque_limits = -20.0
            
            # dof_vel_limits = -20.0
            # -1.25e-7
            dof_acc = -2.5e-7 
            # -0.0
            action_rate = -0.0 
            # -0.01
            target_smoothness = -0.01 
            # -1.0
            collision = -1.5 
            # -1.0
            termination = -1.0 
            
            # power = -2.5e-5
            # -10.0
            dof_pos_limits = -10.0 
            # 3.0
            tracking_lin_vel = 3.0 
            # 0.9
            tracking_ang_vel = 0.9 
            # 2.0 
            feet_air_time = 3.0  
            
            # stumble = -0.0 # -0.5
            # -0.2
            stand_still = -0.1   
            
            # feet_velocity = -0.05 # -0.2
            # 0.1
            episode_length = 0.1 
            
            # trot_symmetry = 0.005 # 0.05
            
            # feet_height = 5
            
            # step_frequency_penalty = -0.01
            





    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            height_measurements = 0.0

        dof_history_interval = 1
        clip_angles = [[-0.523, 0.523], [-0.314, 3.6], [-2.792, -0.524]]

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        heights_uniform_noise = False
        heights_gaussian_mean_mutable = True
        heights_downgrade_frequency = False  # heights sample rate: 10 Hz

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            height_measurements = 0.0

    class commands(LeggedRobotCfg.commands):
        curriculum = False # False
        fixed_commands = None  # None or [lin_vel_x, lin_vel_y, ang_vel_yaw]
        resampling_time = 6  # time before command are changed[s]

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'  # none, plane, heightfield or trimesh
        dummy_normal = True
        random_reset = True
        curriculum = True
        max_init_terrain_level = 2
        horizontal_scale = 0.05  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 5  # [m]
        # terrain_length = 8.
        # terrain_width = 8.
        # num_rows = 10  # number of terrain rows (levels)
        # num_cols = 10  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping stones, wave]
        terrain_proportions = [0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0]  # proportions of each terrain type
        # terrain_types: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping stones, wave]
        # terrain_proportions = [0.15, 0.15, 0.15, 0.0, 0.2, 0.2, 0.15]
        # terrain_proportions = [0.2, 0.2, 0, 0.0, 0.2, 0.2, 0.2]
        # rough terrain only:
        measure_heights = False

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        randomize_com_offset = True
        com_offset_range = [[-0.05, 0.01], [-0.03, 0.03], [-0.03, 0.03]]
        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]
        randomize_Kp_factor = True
        Kp_factor_range = [0.8, 1.2]
        randomize_Kd_factor = True
        Kd_factor_range = [0.8, 1.2]

    # class pmtg(LeggedRobotCfg.pmtg):
    #     gait_type = 'trot'
    #     duty_factor = 0.6
    #     base_frequency = 1.8
    #     max_clearance = 0.12
    #     body_height = 0.31
    #     max_horizontal_offset = 0.05
    #     train_mode = True


class Lite3RoughCfgPPO(LeggedRobotCfgPPO):

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'rough_lite3'
        max_iterations = 10  # number of policy updates
        resume = False
        resume_path = 'legged_gym/logs/rough_lite3'  # updated from load_run and chkpt
        load_run = '' # -1 = last run
        checkpoint = -1  # -1 = last saved model
