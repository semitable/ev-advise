from functools import partial

import GPy
import numpy as np
import pandas as pd

####Load dataset
#dataset = np.loadtxt('data.dat')


#COL_TIMESTAMP = ''
#COL_WATTAGE = 2
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
        '''Initializing the IEC.

        Args:
            param1: Historical Dataset. Last value must be current time
        '''
        self.data = data
        self.now = data.index[-historical_offset]
        self.PredictionWindow = 12 * 60
        self.algorithms = {
            "Renes": partial(self.renes, HistoricalOffset = historical_offset),
            "Renes Hybrid": partial(self.renesHybrid, HistoricalOffset = historical_offset),
        }

    def predict(self, AlgKeys):
        index = pd.DatetimeIndex(start=self.now, freq='T', periods=self.PredictionWindow)
        result = pd.DataFrame(index=index)

        for key in AlgKeys:
            r = self.algorithms[key]()
            if (r.shape[1] if r.ndim > 1 else 1) > 1:
                result[key] = r[:, 0]
                result[key+' STD'] = r[:, 1]
            else:
                result[key] = r

        return result


    def renes(self, PredictionWindow=12*60, HistoricalOffset=0):

        assert HistoricalOffset > PredictionWindow, "Not enough predictions available from renes"

        predictions = np.zeros(PredictionWindow)
        predictions[:] = self.data[-HistoricalOffset:-HistoricalOffset+PredictionWindow][col_prediction].values
        predictions[0] = self.data.iloc[-HistoricalOffset][col_production] #0 is the ground truth...
        return np.squeeze(predictions)



    def renesHybrid(self, TrainingCycle=2, PredictionWindow=12*60, HistoricalOffset=0, stochastic_interval=2):

        assert HistoricalOffset > PredictionWindow, "Not enough predictions available from renes"

        training_length=(TrainingCycle*PredictionWindow)

        #Fetch historical predictions from dataset according to TrainingCycle
        prev_predictions= self.renes(PredictionWindow = training_length, HistoricalOffset = HistoricalOffset+training_length)

        predictions = np.zeros((PredictionWindow, 2))
        predictions[:, 0] = self.renes(PredictionWindow = PredictionWindow, HistoricalOffset = HistoricalOffset)


        #Fetch data given the TrainingCycle and PredictionWindow
        ground_truth = self.data[-HistoricalOffset-training_length:-HistoricalOffset][col_production].values #Shift by one to align timestamps (prediction vs truth)

        #print Predictions[:,0]
        #print dataset[-HistoricalOffset-TrainingLength:-HistoricalOffset+1, 0]

        #Initialize and standardize GP training set

        Index=np.arange(stochastic_interval, training_length, stochastic_interval)

        temp = ground_truth - prev_predictions

        X1=np.atleast_2d(Index/float(training_length)).T

        Y1= np.atleast_2d(np.mean(temp.reshape(-1, stochastic_interval), axis=1)[:-1]).T

        std=np.std(Y1[:,0])
        Y1[:,0]=(Y1[:,0])/std

        #print X1, Y1

        #Train GP
        kernel =  GPy.kern.Matern32(1)#, variance=0.1, lengthscale=float(intervalST/float(TrainingLength)))
        m = GPy.models.GPRegression(X1,Y1,kernel=kernel)

        m.optimize()

        #m.plot()
        #pylab.show(block=True)

        #Initialize and standardize GP input set
        x=np.atleast_2d(np.arange(training_length,training_length+PredictionWindow,1)/float(training_length)).T

        #GP Predict
        y_mean, y_var = m.predict(x)

        #Destandardize output
        y_mean=y_mean*std
        y_var=y_var*std**2

        #Populate array (1st element is the groundtruth) and bound to physical limits
        NominalPowerWTG=3000*60 #3 Kw --> 3*60 joule (in a minute) #np.inf
        predictions[1:, 0]=np.clip(predictions[1:, 0]+y_mean[1:,0],0, NominalPowerWTG)
        predictions[1:,1]=y_var[1:,0]

        return predictions
