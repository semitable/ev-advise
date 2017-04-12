""" A simple battery model
"""

from math import exp, sqrt, pow
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
import itertools


class Battery:
    def __init__(self, Q, it):
        self.V_batt = None
        """battery voltage"""

        self.Q = Q
        """maximum battery capacity"""

        if it is None:
            self.it = Q - 0.1 * Q
        else:
            self.it = it
        """actual battery charge"""

        self.E0 = None
        """battery constant voltage"""
        self.K = None
        """polarization constant"""

        self.i = None
        """battery current"""
        self.i_star = None
        """filtered current"""

        self.A = None
        """exponential zone amplitude"""
        self.B = None
        """exponential zone time constant inverse"""
        self.R = None
        """internal resistance of battery"""

    def set_charge_percentage(self, percent_charged):
        self.it = self.Q * (1 - percent_charged)

    def set_open_circuit(self):
        self.i = 0
        self.i_star = 0

    def set_stable_current(self, i):
        self.i = i
        self.i_star = i

    def calc_charging_voltage(self):
        self.V_batt = self.E0 - self.R * self.i - self.K * (
            self.Q / (self.it - 0.1 * self.Q)) * self.i_star - self.K * (
            self.Q / (self.Q - self.it)) * self.it + self.A * exp(-self.B * self.it)
        return self.V_batt

    def charge(self, P, dt):
        self.calc_charging_voltage()
        c_star = (-self.V_batt + sqrt(pow(self.V_batt, 2) + 4000 * self.R * P)) / (2 * self.Q * self.R)
        dc = c_star * dt

        self.it -= dc

    def charge_ampere(self, A, dt):
        self.set_stable_current(-A)
        self.calc_charging_voltage()
        self.it -= A * dt

    def get_soc(self):
        soc = (self.Q - self.it) / self.Q
        if soc >= 1:
            return 1
        else:
            return soc


class NissanLeaf(Battery):
    """2013 Nissan Leaf S - VIN 9270"""

    def __init__(self, it=None):
        super(NissanLeaf, self).__init__(66.2, it)  # about 24 kwh
        self.E0 = 390.3692
        self.R = 0.054054
        self.K = .040496
        self.A = 30.2313
        self.B = 0.91685


class Charger:
    def __init__(self, percent_charged=0.1):
        self.efficiency = 0.9
        self.max_charge = 6.6  # in kw
        self.action_set = [0, 0.25, 0.5, 0.75, 1]

        self.battery = NissanLeaf()
        self.battery.set_open_circuit()
        self.battery.calc_charging_voltage()

        self.battery.set_charge_percentage(percent_charged)

        self.dt = 0.5

    def charge(self, action, interval):
        N = int(interval / self.dt)

        for _ in itertools.repeat(None, N):  # todo check how many times this is repeated
            self.battery.charge(self.efficiency * action * self.max_charge, self.dt)

        consumption = action * self.max_charge * interval / 60

        self.battery.set_charge_percentage(round(self.battery.get_soc(), 3))

        return self.battery.get_soc(), consumption
