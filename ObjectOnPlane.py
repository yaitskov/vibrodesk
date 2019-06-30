from math import copysign


def speed(a, t):
    return a * t


def location(v0, a, t):
    return v0 * t + a * t * t / 2.0


class ObjectOnPlane:
    def __init__(self, m, r, g=9.8, v=0.0, x=0.0):
        self._v = v
        self._x = x
        self._m = m
        self._r = r
        self._g = g

    def step(self, dt, F):
        (x, v, m, r) = (self._x, self._v, self._m, self._r)
        Ftr = copysign(m * r * self._g, -v)
        a = (F + Ftr) / m
        if a * v < 0:  # result force and speed are opposed
            tStop = -v / a
            print("tstop %f; Ftr = %f; a = %f" % (tStop, Ftr, a))
            if tStop < dt:  # stop before dt runs out
                xStop = x + location(v, a, tStop)
                if abs(Ftr) >= abs(F):
                    return (xStop, 0)
                Ftr = copysign(Ftr, -F)
                a = (F + Ftr) / m
                t = dt - tStop
                print("t = %f; a = %f; Ftr = %f " % (t, a, Ftr))
                return (xStop + location(0, a, t), speed(a, t))
        return (x + location(v, a, dt), v + speed(a, dt))

    def setState(self, state):
        (x, v) = state
        self._x = x
        self._v = v
