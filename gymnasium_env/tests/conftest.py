import numpy as np
import pytest

@pytest.fixture
def params():
    return {
        "mass": 1.0,
        "I": np.diag([0.005, 0.005, 0.009]),
        "arm_length": 0.1,
        "k_thrust": 1.0,
        "k_torque": 0.02,
        "gravity": 9.81,
    }

@pytest.fixture
def zero_state():
    from gymnasium_env.envs.simulation import State
    return State(
        position=np.zeros(3),
        velocity=np.zeros(3),
        orientation=np.array([0,0,0,1]),
        angular_velocity=np.zeros(3)
    )
