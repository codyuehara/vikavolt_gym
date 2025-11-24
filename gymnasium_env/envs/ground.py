class GroundContact:
    def __init__(self,
                 restitution=0.0,
                 friction=0.2,
                 angular_damping=0.5):
        """
        restitution: bounce factor (0 = no bounce)
        friction: horizontal deceleration when touching ground
        angular_damping: reduces angular velocity on contact
        """
        self.restitution = restitution
        self.friction = friction
        self.angular_damping = angular_damping

    def apply(self, state):
        pos = state.position
        vel = state.velocity
        ang_vel = state.angular_velocity

        # 1. Prevent going below the ground
        if pos[2] < 0.0:
            pos[2] = 0.0

        # 2. If below ground OR exactly at ground with downward speed → collision
        if pos[2] == 0.0 and vel[2] < 0:
            # Reverse vertical velocity * restitution
            vel[2] = -self.restitution * vel[2]

        # 3. Friction only applies when on the ground
        if pos[2] == 0.0:
            vel[0] *= (1 - self.friction)
            vel[1] *= (1 - self.friction)

            # 4. Angular damping on ground
            ang_vel[:] = ang_vel * (1 - self.angular_damping)

        # Write back updated values
        state.position = pos
        state.velocity = vel
        state.angular_velocity = ang_vel
