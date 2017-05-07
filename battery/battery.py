""" A simple battery model
"""

from math import exp, sqrt, pow
import numpy as np
from scipy.interpolate import interp1d
import plotly.offline as py
import plotly.graph_objs as go
import itertools
import yaml

with open("config.yml", 'r') as ymlfile:
	cfg = yaml.load(ymlfile)

if cfg['battery']['actions'] == 'two':
	battery_action_set = [0, 1]
	battery_efficiency = [1, 0.91]

elif cfg['battery']['actions'] == 'five':
	battery_action_set = [0, 0.25, 0.5, 0.75, 1]
	battery_efficiency = [0, 0.85, 0.89, 0.9, 0.91]

elif cfg['battery']['actions'] == 'all':

	# interpolating efficiency
	x = [0.2, 0.25, 0.5, 0.75, 1]
	y = [0.81, 0.85, 0.89, 0.9, 0.91]

	f = interp1d(x, y)

	battery_action_set = np.linspace(0.2, 1, 81)
	battery_efficiency = f(battery_action_set)

	battery_action_set = list(battery_action_set)
	battery_efficiency = list(battery_efficiency)

else:
	raise ValueError




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
            return round(soc, 2)


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

        self.max_charge = 6.6  # in kw

		self.action_set = battery_action_set
		self.efficiency = battery_efficiency


        self.battery = NissanLeaf()
        self.battery.set_open_circuit()
        self.battery.calc_charging_voltage()

        self.battery.set_charge_percentage(percent_charged)

        self.dt = 1

    def charge(self, action, interval):

        if action == 0 or interval == 0:
            return self.battery.get_soc(), 0

        current_efficiency = self.efficiency[self.action_set.index(action)]

        N = int(interval / self.dt)
        total_charges = N

        for i in range(N):  # todo check how many times this is repeated
            self.battery.charge(current_efficiency * action * self.max_charge, self.dt)

            if self.battery.get_soc() >= 1:
                total_charges = i
                break

        consumption = (total_charges / N) * action * self.max_charge * interval / 60

        self.battery.set_charge_percentage(self.battery.get_soc())

        return self.battery.get_soc(), consumption
