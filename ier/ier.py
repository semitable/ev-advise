from functools import partial

import GPy
import numpy as np
import pandas as pd

# Load dataset
# dataset = np.loadtxt('data.dat')


# COL_TIMESTAMP = ''
# COL_WATTAGE = 2
col_production = 'WTG Production'
col_prediction = 'WTG Prediction'

'''
@returns predictions from renes
'''


class IER(object):
    """The Intelligent Energy Component of a house.
    IEC will use several methods to predict the energy consumption of a house
    for a given prediction window using historical data.
    """

    def __init__(self, data, historical_offset):
        """Initializing the IEC.

        Args:
            :param data: Historical dataset
            :param historical_offset: offset which will be the current time (will be used as negative)
        """
        self.data = data
        self.now = data.index[-historical_offset]
        self.prediction_window = 12 * 60
        self.algorithms = {
            "Renes": partial(self.renes, historical_offset=historical_offset),
            "Renes Hybrid": partial(self.renes_hybrid, historical_offset=historical_offset),
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

    def renes(self, prediction_window=12 * 60, historical_offset=0):

        assert historical_offset > prediction_window, "Not enough predictions available from renes"

        predictions = np.zeros(prediction_window)
        predictions[:] = self.data[-historical_offset:-historical_offset + prediction_window][col_prediction].values
        predictions[0] = self.data.iloc[-historical_offset][col_production]  # 0 is the ground truth...
        return np.squeeze(predictions)

    def renes_hybrid(self, training_cycle=2, prediction_window=12 * 60, historical_offset=0, stochastic_interval=2):

        assert historical_offset > prediction_window, "Not enough predictions available from renes"

        training_length = (training_cycle * prediction_window)

        # Fetch historical predictions from dataset according to TrainingCycle
        prev_predictions = self.renes(prediction_window=training_length,
                                      historical_offset=historical_offset + training_length)

        predictions = np.zeros((prediction_window, 2))
        predictions[:, 0] = self.renes(prediction_window=prediction_window, historical_offset=historical_offset)

        # Fetch data given the TrainingCycle and PredictionWindow
        ground_truth = self.data[-historical_offset - training_length:-historical_offset][
            col_production].values  # Shift by one to align timestamps (prediction vs truth)

        # print Predictions[:,0]
        # print dataset[-HistoricalOffset-TrainingLength:-HistoricalOffset+1, 0]

        # Initialize and standardize GP training set

        index = np.arange(stochastic_interval, training_length, stochastic_interval)

        temp = ground_truth - prev_predictions

        x1 = np.atleast_2d(index / float(training_length)).T

        y1 = np.atleast_2d(np.mean(temp.reshape(-1, stochastic_interval), axis=1)[:-1]).T

        std = np.std(y1[:, 0])
        y1[:, 0] = (y1[:, 0]) / std

        # print x1, y1

        # Train GP
        kernel = GPy.kern.Matern32(1)  # , variance=0.1, lengthscale=float(intervalST/float(TrainingLength)))
        m = GPy.models.GPRegression(x1, y1, kernel=kernel)

        m.optimize()

        # m.plot()
        # pylab.show(block=True)

        # Initialize and standardize GP input set
        x = np.atleast_2d(np.arange(training_length, training_length + prediction_window, 1) / float(training_length)).T

        # GP Predict
        y_mean, y_var = m.predict(x)

        # Destandardize output
        y_mean *= std
        y_var *= std ** 2

        # Populate array (1st element is the ground truth) and bound to physical limits
        nominal_power_wtg = 3000 * 60  # 3 Kw --> 3*60 joule (in a minute) #np.inf
        predictions[1:, 0] = np.clip(predictions[1:, 0] + y_mean[1:, 0], 0, nominal_power_wtg)
        predictions[1:, 1] = y_var[1:, 0]

        return predictions
