import logging
from datetime import timedelta
from functools import partial

import GPy
import numpy as np
import pandas as pd
import yaml

# Load dataset
# dataset = np.loadtxt('data.dat')


# COL_TIMESTAMP = ''
# COL_WATTAGE = 2
col_production = 'WTG Production'
col_prediction = 'WTG Prediction'

# load predictions

logger = logging.getLogger('advise-unit')

logger.info("reading wind predictions...")
iers = pd.read_csv("windpower.csv.gz", index_col=[0, 1], parse_dates=True)
iers.index = iers.index.set_levels([iers.index.levels[0], pd.to_timedelta(iers.index.levels[1])])
iers = iers.tz_localize('UTC', level=0).tz_convert('Europe/Zurich', level=0)

with open("config/common.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

iers['Windspeed'] = iers['Windspeed'] * cfg['adjustment']['wtg-scale']

'''
@returns predictions from renes
'''


class IER(object):
    """The Intelligent Energy Component of a house.
    IEC will use several methods to predict the energy consumption of a house
    for a given prediction window using historical data.
    """

    def __init__(self, data, current_time):
        """Initializing the IEC.

        Args:
            :param data: Historical dataset
            :param current_time: current_time (will discard any truth values after now)
        """
        self.data = data
        self.now = current_time

        # cheesy solution to find historical_offset as thanos used it...
        window_stop_row = data[data.index == current_time].iloc[-1]
        historical_offset = len(data.index) - data.index.get_loc(window_stop_row.name)

        self.prediction_window = 16 * 60
        self.algorithms = {
            "Renes": partial(self.renes, cur_time=self.now, prediction_window=self.prediction_window),
            "Renes Hybrid": partial(self.renes_gpy,
                                    prediction_window=self.prediction_window),
        }

    def predict(self, alg_keys):
        index = pd.DatetimeIndex(start=self.now, freq='T', periods=self.prediction_window)
        result = pd.DataFrame(index=index)

        for key in alg_keys:
            r = self.algorithms[key]()
            if (r.shape[1] if r.ndim > 1 else 1) > 1:
                result[key] = r[:, 0]
                result[key + ' STD'] = r[:, 1]
            else:
                result[key] = r

        return result

    def renes(self, cur_time, prediction_window):

        predictions = np.zeros(prediction_window)
        predictions[:] = (iers.loc[cur_time.replace(minute=0)].resample('1T').interpolate() / 60)[
                         timedelta(minutes=cur_time.minute):timedelta(
                             minutes=prediction_window + cur_time.minute - 1)]['Windspeed'].values

        predictions[0] = self.data.loc[cur_time][col_production]  # 0 is the ground truth...

        return predictions

    def renes_gpy(self, prediction_window, stochastic_interval=60, training_cycles=2):

        try:
            training_length = training_cycles * prediction_window

            cur_time = self.now

            prev_truth = []
            prev_predictions = []

            predictions = np.zeros((prediction_window, 2))
            predictions[:, 0] = self.renes(self.now, prediction_window)

            for i in range(training_cycles + 1, 1, -1):
                test_time = cur_time - timedelta(minutes=prediction_window) * i

                t = self.data[col_production][test_time:test_time + timedelta(minutes=prediction_window - 1)] / 60
                p = self.renes(test_time, prediction_window)

                prev_truth = np.concatenate([prev_truth, t.values])
                prev_predictions = np.concatenate([prev_predictions, p])

            index = np.arange(stochastic_interval, training_length, stochastic_interval)
            temp = prev_truth - prev_predictions

            x1 = np.atleast_2d(index / float(training_length)).T
            y1 = np.atleast_2d(np.mean(temp.reshape(-1, stochastic_interval), axis=1)[:-1]).T
            std = np.std(y1[:, 0])
            y1[:, 0] = (y1[:, 0]) / std

            # Train GP
            kernel = GPy.kern.Matern32(1)  # , variance=0.1, lengthscale=float(intervalST/float(TrainingLength)))
            m = GPy.models.GPRegression(x1, y1, kernel=kernel)
            m.optimize()
            #
            # m.plot()
            # pylab.show(block=True)

            # Initialize and standardize GP input set
            x = np.atleast_2d(
                np.arange(training_length, training_length + prediction_window, 1) / float(training_length)).T

            # GP Predict
            y_mean, y_var = m.predict(x)

            # Destandardize output
            y_mean *= std
            y_var *= std ** 2

            # todo check bound to physical limits
            # Populate array (1st element is the ground truth) and bound to physical limits
            nominal_power_wtg = 3000 * 60  # 3 Kw --> 3*60 joule (in a minute) #np.inf
            predictions[1:, 0] = np.clip(predictions[1:, 0] + y_mean[1:, 0], 0, np.inf)
            predictions[1:, 1] = y_var[1:, 0]

            predictions = predictions[:, 0]
            return predictions
        except:
            return self.renes(self.now, prediction_window)
