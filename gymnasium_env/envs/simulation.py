import numpy as np
from dataclasses import dataclass

class Utils:
    @staticmethod
    def quat_to_rotation_matrix(q):
        # Rotation matrix
        qx, qy, qz, qw = q
        R = np.array([
            [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw,     1 - 2*qx*qx - 2*qz*qz,     2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw,         2*qy*qz + 2*qx*qw,   1 - 2*qx*qx - 2*qy*qy]
        ])
        return R

    @staticmethod
    def normalize_quat(q):
        return q / np.linalg.norm(q)

    @staticmethod
    def quat_mul(q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        return np.array([x, y, z, w])

    @staticmethod
    def quat_from_omega(q, omega):
        """Quaternion derivative q_dot = 0.5 * q (x) [0, w]."""
        ox, oy, oz = omega
        omega_q = np.array([ox, oy, oz, 0.0])
        return 0.5 * Utils.quat_mul(q, omega_q)

@dataclass
class State:
    position: np.ndarray # (3,)
    velocity: np.ndarray # (3,)
    orientation: np.ndarray # (x,y,z,w)
    angular_velocity: np.ndarray # (3,)
#    motor_speeds: np.ndarray # (4,)

class Quadrotor:
    def __init__(self, mass=1.0, initial_position=None, dt=0.01):
        # physical constants
        self.mass = mass
        #self.arm_length = 0.2 # meters?
        #self.kf = 1e-5 # thrust coeff
        #self.km = 2e-6 # moment coeff
        self.motor_tau = 0.02 # motor time constant (seconds)

        # Inertia matrix
        self.J = np.diag([0.01, 0.01, 0.02])
        self.J_inv = np.linalg.inv(self.J)

        self.g = np.array([0,0,-9.81])
        self.dt = dt

        # initial state
        self.state = State(
            position=np.zeros(3) if initial_position is None else initial_position,
            velocity=np.zeros(3),
            orientation=np.array([0,0,0,1]),
            angular_velocity=np.zeros(3),
         #   motor_speeds=np.zeros(4),
        )
    

    def derivatives(self, state, control):
        thrust, tau_roll, tau_pitch, tau_yaw = control
        pos, vel, q, w = (
            state.position,
            state.velocity,
            state.orientation,
            state.angular_velocity,
        )

        # Rotation matrix
        R = Utils.quat_to_rotation_matrix(q)

        # Translational dynamics
        thrust_world = R @ np.array([0,0,thrust])
        accel = self.g + thrust_world / self.mass

        # Rotational dynamics
        tau = np.array([tau_roll, tau_pitch, tau_yaw])
        ang_accel = self.J_inv @ (tau - np.cross(w, self.J @ w))

        # Quaternion derivative
        q_dot = Utils.quat_from_omega(q, w)

        return State(
            position=vel,
            velocity=accel,
            orientation=q_dot,
            angular_velocity=ang_accel
        )

    def rk4_step(self, control):
        dt = self.dt 
        s = self.state
        
        def add(s, k, scale):
            return State(
                position=s.position + scale * k.position,
                velocity=s.velocity + scale * k.velocity,
                orientation=Utils.normalize_quat(s.orientation + scale * k.orientation),
                angular_velocity=s.angular_velocity + scale * k.angular_velocity
            )
        
        k1 = self.derivatives(s, control)
        k2 = self.derivatives(add(s, k1, dt/2), control)
        k3 = self.derivatives(add(s, k2, dt/2), control)
        k4 = self.derivatives(add(s, k3, dt), control)
    
        self.state = add(
            s,
            State(
                position=k1.position + 2*k2.position + 2*k3.position + k4.position,
                velocity=k1.velocity + 2*k2.velocity + 2*k3.velocity + k4.velocity,
                orientation=k1.orientation + 2*k2.orientation + 2*k3.orientation + k4.orientation,
                angular_velocity=k1.angular_velocity + 2*k2.angular_velocity + 2*k3.angular_velocity + k4.angular_velocity,
            ),
            dt/6,
        )
    
    def step(self, control):
        self.rk4_step(control)
        return self.state
