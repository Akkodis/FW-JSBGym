import time
from jsbgym.trim.trim_point import TrimPoint
from math import pi as PI

class PID:
    def __init__(self, kp: float = 0, ki: float = 0, kd: float = 0, dt: float = None,
                 trim: TrimPoint = None, limit: float = 0, is_throttle:bool = False):
        self.kp: float = kp
        self.ki: float = ki
        self.kd: float = kd
        self.dt: float = dt
        self.trim: TrimPoint = trim
        self.is_throttle: bool = is_throttle
        self.limit: float = limit
        self.integral: float = 0.0
        self.prev_error: float = 0.0
        self.ref: float = 0.0
        self.last_time = time.monotonic()

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.ref = 0.0
        self.last_time = time.monotonic()

    def set_reference(self, ref: float) -> None:
        # reset integral and prev_error if reference changes
        if self.ref != ref:
            self.integral = 0.0
            self.prev_error = 0.0
        self.ref = ref

    def _saturate(self, u):
        u_sat: float = u
        # throttle command is between 0 and 1
        if self.is_throttle:
            if u > self.limit:
                u_sat = self.limit
            elif u < 0:
                u_sat = 0
        # flight control surfaces (aileron, elevator, rudder) are between -limit and +limit
        else:
            if u >= self.limit:
                u_sat = self.limit
            elif u <= -self.limit:
                u_sat = -self.limit
        return u_sat

    def update(self, state: float, state_dot: float = 0, saturate: bool = False, 
               normalize: bool = False, is_course: bool = False) -> tuple[float, float]:
        now = time.monotonic()
        if self.dt is None:
            self.dt = now - self.last_time if (now - self.last_time) else 1e-16
        elif self.dt <= 0:
            raise ValueError('dt has negative value {}, must be positive'.format(self.dt))

        error: float = self.ref - state

        # if we're computing the course error, ensure we're picking the error for making the shortest turn
        if is_course:
            if error > PI :
                error -= 2 * PI
            elif error < -PI:
                error += 2 * PI

        self.integral += error * self.dt
        self.prev_error = error
        u: float = self.kp * error + self.ki * self.integral - self.kd * state_dot

        if self.is_throttle:
            u = self.trim.throttle + u
        if saturate:
            u = self._saturate(u)
        if normalize:
            u = self._normalize(u)
        self.last_time = now
        return u, error, self.integral

    def _normalize(self, u: float) -> float:
        t_min: float # target min
        t_max: float # target max
        if self.is_throttle:
            t_min = 0
            t_max = 1
        else:
            t_min = -1
            t_max = 1
        return (u - (-self.limit)) / (self.limit - (-self.limit)) * (t_max - t_min) + t_min
