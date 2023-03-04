import jsbsim
class PID:
    def __init__(self, kp: float = 0, ki: float = 0, kd: float = 0, dt: float = 0, limit: float = 0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.limit = limit
        self.integral = 0
        self.prev_error = 0
        self.ref: float = 0.0

    def set_reference(self, ref: float) -> None:
        self.ref = ref

    def _saturate(self, u):
        if u >= self.limit:
            u_sat = self.limit
        elif u <= -self.limit:
            u_sat = -self.limit
        else:
            u_sat = u
        return u_sat

    def update(self, state: float, state_dot: float) -> float:
        error: float = self.ref - state
        self.integral += error * self.dt
        self.prev_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * state_dot
        return self._saturate(output)
