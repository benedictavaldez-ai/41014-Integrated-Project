import numpy as np

TICKS_PER_REV = 4096


class Encoders:
    def __init__(self, data, ticks_per_rev=TICKS_PER_REV):
        self.data = data
        self.rad_per_tick = 2 * np.pi / ticks_per_rev
        self.prev = self.read()

    # read wheel angles
    def read(self):
        raw = np.array([self.data.sensor("enc_left").data[0], self.data.sensor("enc_right").data[0]])
        return np.round(raw / self.rad_per_tick) * self.rad_per_tick

    # wheel angle change since last read
    def delta(self):
        now = self.read()
        d = now - self.prev
        self.prev = now
        return d
