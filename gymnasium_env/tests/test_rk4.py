import numpy as np
from gymnasium_env.envs.simulation import Quadrotor

def test_quaternion_normalized(zero_state, params):
    dt = 0.01
    quad = Quadrotor()
    u = np.array([params["mass"]*params["gravity"], 0.1, -0.2, 0.3])

    quad.rk4_step(u)
    next_state = quad.state

    assert np.isclose(np.linalg.norm(next_state.orientation), 1.0)
