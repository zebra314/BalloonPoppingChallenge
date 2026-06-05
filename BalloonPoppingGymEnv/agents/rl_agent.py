import numpy as np
from BalloonPoppingGymEnv.agents.base_agent import BaseAgent


class RLAgent(BaseAgent):
    """
    Reinforcement Learning Agent

    System Architecture & Division of Labor:
    1. High-Level (Target Selector): Heuristic greedy logic to select the nearest active balloon.
    2. Mid-Level (Guidance Planner): RL policy predicting continuous optimal angular rates and throttle.
    3. Low-Level (Controller): Traditional PID loop tracking desired rates and executing actuator outputs.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes agent configuration parameters, constraints, and internal tracking states.

        @param args tuple: Positional arguments passed to the base agent.
        @param kwargs dict: Keyword arguments containing configuration parameters.

        return None
        """
        super().__init__(*args)

        # 1. Extract Low-Level Control Parameters
        sensors_cfg = self.given_parameters["rocket"]["sensors"]
        control_cfg = self.given_parameters["rocket"]["control"]

        self.dt = 1.0 / sensors_cfg["sampling_rate"]
        self.max_gimbal = control_cfg["gimbal_range"]
        self.max_roll = control_cfg["max_roll_torque"]

        # 2. Extract Mid-Level RL Constraints
        self.throttle_min = control_cfg["throttle_range"][0]
        self.throttle_max = control_cfg["throttle_range"][1]
        self.gimbal_rate_limit = control_cfg["gimbal_rate_limit"]
        self.balloon_radius = self.given_parameters["balloon"]["radius"]

        # 3. Extract Navigation & Physical Constants
        tank_cfg = self.given_parameters["rocket"]["tank"]
        self.gyro_bias = sensors_cfg["gyro_constant_bias"]
        self.accel_bias = sensors_cfg["accelerometer_constant_bias"]
        self.initial_fuel_mass = tank_cfg["initial_liquid_mass"]
        self.fuel_flow_rate = tank_cfg["liquid_mass_flow_rate_out"]
        self.launch_elevation = self.given_parameters["environment"]["elevation"]

        # 4. Initialize Tracking and Control States
        self.launch_time = kwargs.get("launch_time", 1.0)
        self.current_target = None
        self.current_target_idx = None
        self.rate_errors = np.zeros((3, 1))

        # Temporary mock for navigation tracking state
        self.states = np.zeros((10,))
        self.states[2] = self.launch_elevation

    def get_action(self, observation):
        """
        Coordinates target selection, trajectory planning, and low-level control loops.

        @param observation dict: Current environment observations containing telemetry and balloon data.

        return dict: Actuator command dictionary for the gym environment.
        """
        t = observation["simulation_time"]
        launch = t >= self.launch_time

        # Step 1: Target Selection Layer
        if self.current_target is None or self._is_target_popped(observation):
            self.current_target = self._select_next_target(observation)

        # Step 2: Guidance Layer (RL Plan - Fixed Output Placeholder)
        desired_rates, desired_throttle = self._plan_trajectory_rl(
            observation, self.current_target
        )

        # Step 3: Control Layer (PID tracking execution)
        torque_cmd = self._run_pid_control(observation, desired_rates)

        # Step 4: Actuator Guard and Mapping
        tvc_cmd = np.clip(torque_cmd[:2], -self.max_gimbal, self.max_gimbal)
        roll_cmd = np.clip(torque_cmd[2], -self.max_roll, self.max_roll)
        throttle_cmd = np.clip(desired_throttle, self.throttle_min, self.throttle_max)

        return {
            "launch": launch,
            "launch_inclination_heading": np.array([90.0, 0.0]),
            "tvc": tvc_cmd,
            "roll": roll_cmd,
            "throttle": throttle_cmd,
        }

    def _select_next_target(self, observation):
        """
        Selects the nearest active balloon using a greedy approach based on current distance.

        @param observation dict: Current environment observations containing balloon matrices.

        return ndarray: 3D coordinates of the chosen target balloon, or None.
        """
        status = observation["balloon_status"].flatten()
        states = observation["balloon_states"]
        rocket_pos = self.states[0:3]

        min_dist = float("inf")
        best_target_idx = None

        for i in range(len(status)):
            if status[i] == 1:
                balloon_pos = states[i, 0:3]
                dist = np.linalg.norm(balloon_pos - rocket_pos)
                if dist < min_dist:
                    min_dist = dist
                    best_target_idx = i

        if best_target_idx is not None:
            self.current_target_idx = best_target_idx
            return states[best_target_idx, 0:3]

        return None

    def _is_target_popped(self, observation):
        """
        Evaluates whether the currently tracked balloon target has been eliminated.

        @param observation dict: Current environment observations containing balloon status arrays.

        return bool: True if the target is popped or unassigned, False otherwise.
        """
        if self.current_target is None or self.current_target_idx is None:
            return True

        status = observation["balloon_status"].flatten()
        if status[self.current_target_idx] == 0:
            return True

        return False

    def _plan_trajectory_rl(self, observation, target):
        """
        Executes the RL guidance policy to output desired angular tracking rates and throttle commands.

        @param observation dict: Current environment observations.
        @param target ndarray: 3D coordinates of the selected target balloon.

        return tuple: Pair of (desired_rates, desired_throttle).
        """
        fixed_rates = np.array([0.0, 0.0, 0.0])
        fixed_throttle = 1.0
        return fixed_rates, fixed_throttle

    def _run_pid_control(self, observation, desired_rates):
        """
        Computes low-level actuator torque commands to minimize tracking errors.

        @param observation dict: Current environment observations.
        @param desired_rates ndarray: Target angular tracking rates from the guidance layer.

        return ndarray: 3D vector containing X-axis, Y-axis, and Z-axis torque commands.
        """
        # TODO: Implement error integration loop using self.dt
        return np.array([0.0, 0.0, 0.0])
