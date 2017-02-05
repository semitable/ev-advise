import numpy as np
import GPy

from matplotlib import pylab
import matplotlib.pyplot as plt

####Load dataset
dataset = np.loadtxt('data.dat')


COL_TIMESTAMP = 0
COL_WATTAGE = 2
COL_PRODUCTION = 4
COL_PROD_PREDICTION = 5


'''
@returns predictions from renes
'''

class IER(object):
    """The Intelligent Energy Component of a house.
    IEC will use several methods to predict the energy consumption of a house
    for a given prediction window using historical data.
    """

    def __init__(self, data):
        '''Initializing the IEC.

        Args:
            param1: Historical Dataset. Last value must be current time
        '''
        self.data = data
        self.PredictionWindow = 12 * 60
        self.algorithms = {
            "renes": self.renes,
            "renes hybrid": self.renesHybrid
        }

    def predict(self, AlgKeys):
        result = dict()

        for key in AlgKeys:
            result[key] = self.algorithms[key]()

        return result

    def renes(PredictionWindow=12*60, HistoricalOffset=0):
        if(HistoricalOffset <= PredictionWindow):
            print("Not enough predictions available from renes")
            return np.full((PredictionWindow, 2), np.nan)

        Predictions = np.empty((PredictionWindow, 2))
        Predictions[1:,: ] = dataset[-HistoricalOffset:-HistoricalOffset+PredictionWindow-1, [COL_TIMESTAMP, COL_PROD_PREDICTION]] #fetch predictions from renes
        Predictions[1:, 0] = Predictions[1:, 0] + 60 #Shift timestamps to +1 min (the prediction time)

        Predictions[0, : ] = dataset[-HistoricalOffset,[COL_TIMESTAMP, COL_PRODUCTION]] #row 0 is the "Ground Truth"
        return Predictions


    def renesHybrid(TrainingCycle=2, PredictionWindow=12*60, HistoricalOffset=0, intervalST=2):

        if(HistoricalOffset <= PredictionWindow):
            print("Not enough predictions available from renes")
            return np.full((PredictionWindow, 3), np.nan)


        TrainingLength=(TrainingCycle*PredictionWindow)

        #Fetch historical predictions from dataset according to TrainingCycle
        Predictions= renes(PredictionWindow = TrainingLength, HistoricalOffset = HistoricalOffset+TrainingLength)

        #Fetch predictions for current prediction window
        PredictionsLast=np.zeros((PredictionWindow,3))
        PredictionsLast[:,:2] = renes(PredictionWindow = PredictionWindow, HistoricalOffset = HistoricalOffset)

        #Add last element of PredictionsLast to Predictions
        Predictions=np.concatenate((Predictions,[PredictionsLast[0,:2]]), axis=0)#[1::]

        #Fetch data given the TrainingCycle and PredictionWindow
        Hist_Production = dataset[-HistoricalOffset-TrainingLength:-HistoricalOffset+1, COL_PRODUCTION] #Shift by one to align timestamps (prediction vs truth)


        #print Predictions[:,0]
        #print dataset[-HistoricalOffset-TrainingLength:-HistoricalOffset+1, 0]

        #Initialize and standardize GP training set

        Index=np.arange(intervalST,TrainingLength,intervalST)
        X1=np.atleast_2d(Index/float(TrainingLength)).T
        Y1=np.atleast_2d(Hist_Production[Index]-Predictions[Index,1]).T

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
        x=np.atleast_2d(np.arange(TrainingLength,TrainingLength+PredictionWindow,1)/float(TrainingLength)).T

        #GP Predict
        y_mean, y_var = m.predict(x)

        #Destandardize output
        y_mean=y_mean*std
        y_var=y_var*std**2

        #Populate array (1st element is the groundtruth) and bound to physical limits
        NominalPowerWTG=3000*60 #3 Kw --> 3*60 joule (in a minute) #np.inf
        PredictionsLast[1:,1]=np.clip(PredictionsLast[1:,1]+y_mean[1:,0],0, NominalPowerWTG)
        PredictionsLast[1:,2]=y_var[1:,0]


        return PredictionsLast




def renesHybrid(TrainingCycle=2, PredictionWindow=12*60, HistoricalOffset=0, intervalST=2):

    if(HistoricalOffset <= PredictionWindow):
        print("Not enough predictions available from renes")
        return np.full((PredictionWindow, 3), np.nan)


    TrainingLength=(TrainingCycle*PredictionWindow)

    #Fetch historical predictions from dataset according to TrainingCycle
    Predictions= renes(PredictionWindow = TrainingLength, HistoricalOffset = HistoricalOffset+TrainingLength)

    #Fetch predictions for current prediction window
    PredictionsLast=np.zeros((PredictionWindow,3))
    PredictionsLast[:,:2] = renes(PredictionWindow = PredictionWindow, HistoricalOffset = HistoricalOffset)

    #Add last element of PredictionsLast to Predictions
    Predictions=np.concatenate((Predictions,[PredictionsLast[0,:2]]), axis=0)#[1::]

    #Fetch data given the TrainingCycle and PredictionWindow
    Hist_Production = dataset[-HistoricalOffset-TrainingLength:-HistoricalOffset+1, COL_PRODUCTION] #Shift by one to align timestamps (prediction vs truth)


    #print Predictions[:,0]
    #print dataset[-HistoricalOffset-TrainingLength:-HistoricalOffset+1, 0]

    #Initialize and standardize GP training set

    Index=np.arange(intervalST,TrainingLength,intervalST)
    X1=np.atleast_2d(Index/float(TrainingLength)).T
    Y1=np.atleast_2d(Hist_Production[Index]-Predictions[Index,1]).T

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
    x=np.atleast_2d(np.arange(TrainingLength,TrainingLength+PredictionWindow,1)/float(TrainingLength)).T

    #GP Predict
    y_mean, y_var = m.predict(x)

    #Destandardize output
    y_mean=y_mean*std
    y_var=y_var*std**2

    #Populate array (1st element is the groundtruth) and bound to physical limits
    NominalPowerWTG=3000*60 #3 Kw --> 3*60 joule (in a minute) #np.inf
    PredictionsLast[1:,1]=np.clip(PredictionsLast[1:,1]+y_mean[1:,0],0, NominalPowerWTG)
    PredictionsLast[1:,2]=y_var[1:,0]


    return PredictionsLast

def renes(PredictionWindow=12*60, HistoricalOffset=0):
    if(HistoricalOffset <= PredictionWindow):
        print("Not enough predictions available from renes")
        return np.full((PredictionWindow, 2), np.nan)

    Predictions = np.empty((PredictionWindow, 2))
    Predictions[1:,: ] = dataset[-HistoricalOffset:-HistoricalOffset+PredictionWindow-1, [COL_TIMESTAMP, COL_PROD_PREDICTION]] #fetch predictions from renes
    Predictions[1:, 0] = Predictions[1:, 0] + 60 #Shift timestamps to +1 min (the prediction time)

    Predictions[0, : ] = dataset[-HistoricalOffset,[COL_TIMESTAMP, COL_PRODUCTION]] #row 0 is the "Ground Truth"
    return Predictions