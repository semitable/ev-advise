'''  Intelligent energy component contains the IEC class, that includes several algorithms
     for predicting consumption of a house, given historical data. It also contains an IECTester
     class that can be used to test and provide results on multiple IEC runs '''

import pickle
from datetime import datetime as dt
from datetime import timedelta
from functools import partial
from multiprocessing import Pool, cpu_count

import GPy
import numpy as np
import pandas as pd
import scipy.ndimage.filters
import scipy.signal
from scipy import spatial
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

cons_col = 'House Consumption'


def is_weekend(utc):
    '''Evaluate if a day is a weekend Day
    Args:
        param1: A Unix Timestamp

    Returns:
        True if weekend, false otherwise
    '''
    if utc.isoweekday() < 6:
        return 0
    else:
        return 1


def similar_weekday(utc1, utc2):
    ''' Given two utc dates, returns true if they are both weekdays (Mon-Friday) or if they
    are both weekend days
    '''
    return is_weekend(utc1) == is_weekend(utc2)

def binary_hamming_distance(a, b):
    '''Calculate the hamming distance of two binary vectors of equal length
    Args:
        param1: First vector
        param2: Second vector
    Returns:
        The Binary Hamming Distance
    '''
    return np.count_nonzero(a != b)


def cosine_similarity(A, B):
    '''Calculate the cosine similarity between
    two non-zero vectors of equal length (https://en.wikipedia.org/wiki/Cosine_similarity)
    '''
    return 1.0 - spatial.distance.cosine(A, B)


def baseline_similarity(A, B):
    similarity = -mean_squared_error(med_filt(A, 201), med_filt(B, 201))**0.5
    return similarity


def highpass_filter(A):

    cutoff = 2

    baseline = gauss_filt(A)
    highpass = A - baseline
    highpass[highpass < baseline * cutoff] = 0

    return highpass


def advanced_similarity(A, B):

    sigma = 10

    base_similarity = baseline_similarity(A, B)

    HighPassA = highpass_filter(A)
    HighPassB = highpass_filter(B)

    HighPassA = scipy.ndimage.filters.gaussian_filter1d(HighPassA, sigma)
    HighPassB = scipy.ndimage.filters.gaussian_filter1d(HighPassB, sigma)

    highpassSimilarity = -mean_squared_error(HighPassA, HighPassB)

    return base_similarity + highpassSimilarity


def group_to_interval(data, interval, func=lambda x: np.sum(x, axis=1)):
    '''Calculates a vector used by ACP to find similarity between days.
    This function will aggregate the by-minute power consumption to intervals
    '''
    return func(data.reshape(-1, interval))


def is_similar_time(time1, time2):
    '''Checks if the two timestamps have the same hour and minute
       and if they are the same type of week day
    '''
    # print(time1)

    return (time1.hour == time2.hour
            and (time1.minute == time2.minute)
            #and similar_weekday(utc1, utc2)
            )


def running_mean(seq, N):
    """
     Purpose: Find the mean for the points in a sliding window (fixed size)
              as it is moved from left to right by one point at a time.
      Inputs:
          seq -- list containing items for which a mean (in a sliding window) is
                 to be calculated (N items)
            N -- number of items in sliding window
      Otputs:
        means -- list of means with size len(seq)

    """
    cumsum = np.cumsum(np.insert(seq, 0, 0))
    means = (cumsum[N:] - cumsum[:-N]) / N
    means = np.insert(means, 0, np.repeat(means[0], len(seq) - len(means)))
    return means


def mins_in_day(timestamp):
    return timestamp.hour * 60 + timestamp.minute


def find_similar_days(TrainingData, ObservationLength, K, interval, method=cosine_similarity):

    now = TrainingData.index[-1]
    timezone = TrainingData.index.tz

    # Find moments in our dataset that have the same hour/minute and is_weekend() == weekend.
    # Those are the indexes of those moments in TrainingData


    min_time = TrainingData.index[0] + timedelta(minutes=ObservationLength)

    selector = (
        (TrainingData.index.minute==now.minute) &
        (TrainingData.index.hour == now.hour) &
        (TrainingData.index > min_time)
    )
    SimilarMoments = TrainingData[selector][:-1].tz_convert('UTC')
    TrainingData = TrainingData.tz_convert('UTC')


    LastDayVector = (TrainingData
                     .tail(ObservationLength)
                     .resample(timedelta(minutes=interval))
                     .sum()
                     )



    obs_td = timedelta(minutes=ObservationLength)

    SimilarMoments['Similarity'] = [
        method(
            LastDayVector.as_matrix(columns=[cons_col]),
            TrainingData[i - obs_td:i].resample(timedelta(minutes=interval)).sum().as_matrix(columns=[cons_col])
        ) for i in SimilarMoments.index
    ]


    indexes = (SimilarMoments
               .sort_values('Similarity', ascending=False)
               .head(K)
               .index
               .tz_convert(timezone))

    return indexes


def lerp(x, y, alpha):
    assert x.shape == y.shape and x.shape == alpha.shape  # shapes must be equal

    x = x * (1 - alpha)
    y = y * alpha

    return x + y


def med_filt(x, k=201):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    if x.ndim > 1:
        x = np.squeeze(x)
    med = np.median(x)
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros((len(x), k), dtype=x.dtype)
    y[:, k2] = x
    for i in range(k2):
        j = k2 - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[0]
        y[:-j, -(i + 1)] = x[j:]
        y[-j:, -(i + 1)] = med
    return np.median(y, axis=1)


def gauss_filt(x, k=201):
    """Apply a length-k gaussian filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    if x.ndim > 1:
        x = np.squeeze(x)
    med = np.median(x)
    assert k % 2 == 1, "mean filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros((len(x), k), dtype=x.dtype)
    y[:, k2] = x
    for i in range(k2):
        j = k2 - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[0]
        y[:-j, -(i + 1)] = x[j:]
        y[-j:, -(i + 1)] = med
    return np.mean(y, axis=1)


def calc_baseline(TrainingData, similar_moments,
                  PredictionWindow, half_window=100, method=gauss_filt):

    if type(PredictionWindow) is not timedelta:
        PredictionWindow = timedelta(minutes=PredictionWindow)


    K = len(similar_moments)

    r = np.zeros((721, 1))
    for i in similar_moments:
        r += (1 / K) * TrainingData[i:i + PredictionWindow].rolling(window=half_window*2, center=True,
                                                                    min_periods=1).mean().as_matrix()
    baseline = np.squeeze(r)

    recent_baseline = TrainingData[-2*half_window:-1].mean()[cons_col]
    interp_range = 2 * half_window
    baseline[:interp_range] = lerp(np.repeat(recent_baseline, interp_range),
                                   baseline[:interp_range],
                                   np.arange(interp_range) / interp_range)

    return baseline


def highpass_filter(A):

    cutoff = 2

    baseline = gauss_filt(A)
    highpass = A - baseline
    highpass[highpass < baseline * cutoff] = 0

    return highpass


def calc_highpass(TrainingData, similar_moments,
                  PredictionWindow, half_window, method=gauss_filt):

    K = len(similar_moments)

    similar_data = np.zeros((K, PredictionWindow + 2 * half_window))

    for i in range(K):
        similar_data[i] = TrainingData[similar_moments[i] - half_window:
                                       similar_moments[
                                           i] + PredictionWindow + half_window,
                                       2]

    highpass = np.apply_along_axis(highpass_filter, 1, similar_data)

    highpass = highpass[:, half_window: -half_window]

    W = 3
    confidence_threshold = 0.5

    paded_highpass = np.pad(highpass, ((0,), (W,)), mode='edge')

    highpass_prediction = np.zeros(PredictionWindow)

    for i in range(W, PredictionWindow + W):
        window = paded_highpass[:, i - W:i + W]
        confidence = np.count_nonzero(window) / window.size

        if confidence > confidence_threshold:
            highpass_prediction[
                i - W] = np.mean(window[np.nonzero(window)]) * confidence

    return highpass_prediction


def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero()

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size]  # Edit

    # Reshape the result into two columns
    idx.shape = (-1, 2)
    return idx


def create_region(start, end, value, length=720):
    r = np.zeros(length)

    r[start:end] = value
    return r


def usage_region(signal, baseline=None):

    cutoff = 2

    if baseline is None:
        baseline = med_filt(signal, 201)

    highpass = signal - baseline

    highpass[highpass < baseline * cutoff] = 0
    regions = contiguous_regions(highpass > 0)

    return regions


def match_regions(regions1, regions2):
    k = 10
    MatchingRegions = list()

    for r1 in regions1:
        for r2 in regions2:
            if abs(r1[0] - r2[0]) < k and abs(r1[1] - r2[1]) < k:
                MatchingRegions.append(
                    (int(np.mean((r1[0], r2[0]))), int(np.mean((r1[1], r2[1])))))
                break
    return MatchingRegions


def sum_baseline_with_usage(baseline, Regions, fillvalues):
    for start, end in Regions:
        r = create_region(start, end, np.mean(fillvalues[start:end]))
        baseline = baseline + r
    return baseline




class IEC(object):
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
        self.now = data.index[-1]
        self.PredictionWindow = 12 * 60
        self.algorithms = {
            "ACP": self.ACP,
            "ACP baseline": partial(self.ACP, method=baseline_similarity),
            "ACP highpass": partial(self.ACP, method=advanced_similarity),
            "Hybrid": self.stochastic_acp_hybrid,
            "Old ACP": self.acp_old,
            "Simple Mean": self.simple_mean,
            "Baseline Finder": self.baseline_finder,
            "Baseline Finder Hybrid": self.baseline_finder_hybrid,
            "Usage Zone Finder": self.usage_zone_finder
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

    def simple_mean(self, TrainingWindow=24 * 60):
        TrainingData = self.data.tail(TrainingWindow)
        mean = TrainingData[cons_col].mean()
        return np.repeat(mean, self.PredictionWindow)

    def baseline_finder(self, TrainingWindow=1440 * 60, K=5):
        TrainingData = self.data.tail(TrainingWindow)[[cons_col]]

        # ObservationLength is ALL of the current day (till now) + 4 hours
        ObservationLength = mins_in_day(self.now) + (4 * 60)

        similar_moments = find_similar_days(
            TrainingData, ObservationLength, K, 15, method=baseline_similarity)

        half_window = 60

        baseline = calc_baseline(
            TrainingData, similar_moments, self.PredictionWindow, half_window, method=gauss_filt)

        # interpolate our prediction from current consumption to predicted
        # consumption
        interp_range = 15
        # First index is the current time
        current_consumption = TrainingData.tail(1)[cons_col]

        baseline[:interp_range] = lerp(np.repeat(current_consumption, interp_range),
                                       baseline[:interp_range],
                                       np.arange(interp_range) / interp_range)

        return baseline[:-1] #slice last line because we are actually predicting PredictionWindow-1

    def baseline_finder_hybrid(self, TrainingWindow=60 * 24 * 30 * 2, K=5, TrainingCycle=1, stochastic_interval=15):

        assert TrainingCycle == 1, "Not implemented for TrainingCycle > 1"

        TrainingData = self.data.tail(TrainingWindow)[[cons_col]]

        prev_predictions = IEC(TrainingData[:-self.PredictionWindow]).baseline_finder(K=K)
        ground_truth = np.squeeze(TrainingData[-self.PredictionWindow-1:-1].values)

        # Initialize and standardize GP training set
        training_length = (TrainingCycle * self.PredictionWindow)
        # Index=np.arange(0,TrainingLength+intervalST,intervalST)

        index = np.arange(stochastic_interval, training_length, stochastic_interval)


        X1 = np.atleast_2d(index/training_length).T

        temp = ground_truth - prev_predictions
        # temp=gauss_filt(DataA[1:TrainingLength+1, 2], 201)-Predictions[1:TrainingLength+1, 2] #DEN BGAZEI NOHMA TO VAR

        Y1 = np.atleast_2d(np.mean(temp.reshape(-1, stochastic_interval), axis=1)[:-1]).T
        std = np.std(Y1[:, 0])
        Y1[:, 0] = (Y1[:, 0]) / std

        # Train GP
        # K1 =  GPy.kern.Matern32(1, variance=0.1, lengthscale=2*float(intervalST/float(TrainingLength)))+ GPy.kern.White(1)
        # kernel=K1
        kernel = GPy.kern.Exponential(1)#GPy.kern.Exponential(1, variance=0.1, lengthscale=float(stochastic_interval/float(training_length)))

        #kernel.plot()
        #pylab.show(block=True)
        m = GPy.models.GPRegression(X1, Y1, kernel=kernel)
        m.optimize()
        #m.plot()
        #pylab.show(block=True)

        # Initialize and standardize GP input set
        x = np.atleast_2d(np.arange(training_length, training_length + self.PredictionWindow, 1) / float(training_length)).T

        # GP Predict
        y_mean, y_var = m.predict(x)

        # Destandardize output
        y_mean = y_mean * std
        y_var = y_var * std ** 2

        # Populate array (1st element is the groundtruth) and bound to physical limits

        baseline_predictions = np.zeros((self.PredictionWindow, 2))
        baseline_predictions[:, 0] = IEC(TrainingData).baseline_finder(K=K)


        """
        TODO: is the following prediction + mean? I think it is prediction - mean...
        changed it for now, but must double-check
        """
        baseline_predictions[1:, 0] = np.clip(baseline_predictions[1:, 0] - y_mean[1:, 0], 0, np.inf)
        baseline_predictions[1:, 1] = y_var[1:, 0]

        return baseline_predictions


    def usage_zone_finder(self, TrainingWindow=24 * 60 * 120, K=5):

        TrainingData = self.data[-(TrainingWindow):, :]
        CurrentTime = TrainingData[-1, 1]
        # ObservationLength is ALL of the current day (till now) + 4 hours
        ObservationLength = mins_in_day(CurrentTime) + (4 * 60)

        KSimilarDays = find_similar_days(
            TrainingData, ObservationLength, K, 15, method=baseline_similarity)

        half_window = 100  # half window of the lowpass and high pass filter we will use
        baseline = calc_baseline(TrainingData, KSimilarDays,
                                 self.PredictionWindow, half_window, method=gauss_filt)
        highpass = calc_highpass(TrainingData, KSimilarDays,
                                 self.PredictionWindow, half_window, method=gauss_filt)
        final = baseline + highpass

        # interpolate our prediction from current consumption to predicted
        # consumption
        interp_range = 15
        # First index is the current time
        current_consumption = TrainingData[-1, 2]

        final[:interp_range] = lerp(np.repeat(current_consumption, interp_range),
                                    final[:interp_range],
                                    np.arange(interp_range)/interp_range)

        # create the array to be returned. Column 0 has the timestamps, column
        # 1 has the predictions
        Predictions = np.zeros((self.PredictionWindow, 2))
        Predictions[:, 0] = np.arange(
            CurrentTime, CurrentTime + self.PredictionWindow * 60, 60)
        Predictions[:, 1] = final

        return Predictions

    def acp_old(self, TrainingWindow=24 * 60 * 60, K=5, interval=15):

        Data = self.data[-(TrainingWindow)::]
        # Find Last Quarter Index
        LastQuarter = 0
        for i in range(0, interval):
            if dt.utcfromtimestamp(Data[-1 - i, 1]).minute % interval == 0:
                LastQuarter = -1 - i
                break

        # Extract Last Quarter characteristics
        weekend = is_weekend(Data[LastQuarter, 1])
        hour = dt.utcfromtimestamp(Data[LastQuarter, 1]).hour
        minute = dt.utcfromtimestamp(Data[LastQuarter, 1]).minute

        # Extract last partialy observed Day (+4 hours) and aggregate in
        # interval-min intervals
        POLength = (hour * 60 + minute) + (4 * 60)
        LastDay = np.zeros(POLength / interval)
        for i in range(0, POLength // interval):
            LastDay[i] = sum(Data[LastQuarter - POLength + i *
                                  interval:LastQuarter - POLength + i * interval + interval, 2])

        # Choose the K most similar Days
        DayLength = POLength + abs(LastQuarter) + self.PredictionWindow
        KclosestDays = np.zeros((K, DayLength / interval))
        KclosestDaysDistance = np.full(K, np.inf)
        KclosestDaysDates = np.zeros((K, 2))
        TempDay = np.zeros(DayLength / interval)
        for i in range(0, np.shape(Data[:LastQuarter - POLength])[0]):
            # Identify a candidate day
            if (is_weekend(Data[i, 1]) == weekend
                    and dt.utcfromtimestamp(Data[i, 1]).hour == hour
                    and dt.utcfromtimestamp(Data[i, 1]).minute == minute):
                # Aggregate the candidate day in interval-min intervals and
                # calculate distance
                for j in range(0, DayLength // interval):
                    TempDay[j] = sum(
                        Data[i - POLength + j * interval:i - POLength + j * interval + interval, 2])
                Distance = cosine_similarity(
                    TempDay[:np.shape(LastDay)[0]], LastDay)
                # Update the K most similar days
                if Distance < KclosestDaysDistance[-1]:
                    KclosestDaysDistance[-1] = Distance
                    KclosestDays[-1, :] = TempDay[:]
                    KclosestDaysDates[-1, :] = Data[i, :2]  # Check!!!!!!!
                    # Quickshort the K similar Days list
                    ShortedIndexes = np.argsort(KclosestDaysDistance)
                    KclosestDaysDistance = KclosestDaysDistance[ShortedIndexes]
                    KclosestDays = KclosestDays[ShortedIndexes, :]
                    KclosestDaysDates = KclosestDaysDates[ShortedIndexes, :]

        # Calculate predictions based on the K similar Days
        PredictionsQuarter = KclosestDays[
            :, POLength / interval:].sum(axis=0) / K

        # Extrapolate to 1-min interval
        Predictions = np.full(
            (self.PredictionWindow + abs(LastQuarter), 2), np.NAN)
        for i in range(0, np.shape(PredictionsQuarter)[0]):
            Predictions[(interval // 2) + i * interval,
                        1] = PredictionsQuarter[i] / interval
        # Place the last observed ocupancy at the predictions interval
        Predictions[-LastQuarter - 1, 0] = Data[-1, 0]
        Predictions[-LastQuarter - 1, 1] = Data[-1, 2]
        # Trim observed 1-min intervals from Predictions vector
        Predictions = Predictions[-LastQuarter - 1:, :]
        # Extrapolate
        not_nan = np.logical_not(np.isnan(Predictions[:, 1]))
        indices = np.arange(len(Predictions[:, 1]))
        Predictions[:, 1] = np.interp(
            indices, indices[not_nan], Predictions[:, 1][not_nan])

        # Populate UTC dates
        for i in range(1, self.PredictionWindow):
            Predictions[i, 0] = Predictions[i - 1, 0] + 60

        # return predictions
        # end of Last Quarter and biginning of Pred.Window are the same (1st
        # element is the ground truth)
        return Predictions[:-1]

    def ACP(self, TrainingWindow=24 * 60 * 60, K=5, interval=15, method=cosine_similarity):
        '''Simple ACP function. Finds the K Similar days and returns a prediction based on those.
        Args:
            param1: Training window, the days before the current one that
                    will be searched for similarity
            param2: K, The number of similar days
            param3: The interval which should be used to group the minutes
        Returns:
            A PredictionWindow sized array containing predictions. First item is the ground truth.
        '''

        # Data from inside the Training Window
        TrainingData = self.data[-(TrainingWindow):, :]

        CurrentTime = TrainingData[-1, 1]

        # ObservationLength is ALL of the current day (till now) + 4 hours
        ObservationLength = mins_in_day(CurrentTime) + (4 * 60)

        # TrainingData indexes of K similar days
        KSimilarDays = find_similar_days(
            TrainingData, ObservationLength, K, interval, method)

        Predictions = np.zeros((self.PredictionWindow, 2))

        for DayIndex in KSimilarDays:
            Predictions[:, 1] = (Predictions[:, 1]
                                 + TrainingData[DayIndex: DayIndex + self.PredictionWindow, 2] / K)

        Predictions[0, 1] = TrainingData[-1, 2]

        Predictions[:, 1] = running_mean(Predictions[:, 1], interval)

        for i in range(self.PredictionWindow):
            Predictions[i, 0] = TrainingData[-1, 0] + 60 * i

        return Predictions

    def stochastic_acp_hybrid(self, TrainingWindow=1440 * 60, K=5, interval=15, intervalST=30):

        TrainingData = self.data[-(TrainingWindow)::]

        CurrentTime = TrainingData[-1, 1]
        # ObservationLength is ALL of the current day (till now) + 4 hours
        ObservationLength = mins_in_day(CurrentTime) + (4 * 60)
        # indexes (TrainingData) of K similar days
        KSimilarDays = find_similar_days(
            TrainingData, ObservationLength, K, interval)

        # Find Error in ACP Predictions of K most similar days
        MeanError = np.zeros(self.PredictionWindow)
        for i in KSimilarDays:
            MeanError = (MeanError
                         + (TrainingData[i - 1:i + self.PredictionWindow - 1][:, 2]
                            - IEC(TrainingData[:i]).ACP()[:, 1]) / K)

        X = np.atleast_2d(
            np.arange(0, 1, intervalST / self.PredictionWindow)).T
        Y = np.atleast_2d(group_to_interval(
            MeanError, intervalST, lambda x: np.mean(x, axis=1)))

        std = np.std(Y)
        Y = (Y / std).T

        # Train GP
        kernel = (GPy.kern.RBF(1, variance=1, lengthscale=intervalST / self.PredictionWindow)
                  + GPy.kern.White(1))
        m = GPy.models.GPRegression(X, Y, kernel=kernel)
        m.optimize()

        X = np.atleast_2d(np.arange(0, 1, 1 / self.PredictionWindow)).T
        y_mean, y_var = m.predict(X)

        # Destandardize output
        y_mean = y_mean * std
        y_var = y_var * std**2

        Predictions = np.zeros((self.PredictionWindow, 3))
        Predictions[:, :2] = self.ACP()

        # Populate array (1st element is the groundtruth) and bound to physical
        # limits
        Predictions[1:, 1] = np.clip(
            Predictions[1:, 1] - y_mean[1:, 0], 0, np.inf)
        Predictions[1:, 2] = y_var[1:, 0]

        return Predictions


def worker(ie, AlgKeys):
    return ie.predict(AlgKeys)


class IECTester():
    '''Performs several tests to the Intelligent Energy Component.
    '''
    version = 0.1

    def __init__(self, data, PredictionWindow, TestingRange, SaveFile='save.p'):
        self.data = data
        self.PredictionWindow = PredictionWindow
        self.range = TestingRange
        self.SaveFile = SaveFile

        self.hash = 0

        self.TestedAlgorithms = set()
        self.results = dict()
        if SaveFile is not None:
            self.load()

    def load(self):
        try:
            with open(self.SaveFile, "rb") as f:
                savedata = pickle.load(f)
                if(savedata['version'] == self.version
                   and savedata['range'] == self.range
                   and savedata['hash'] == self.hash
                   and savedata['PredictionWindow'] == self.PredictionWindow):
                    self.TestedAlgorithms = savedata['TestedAlgorithms']
                    self.results = savedata['results']

        except (IOError, EOFError):
            pass

    def save(self):
        savedata = dict()
        savedata['version'] = self.version
        savedata['range'] = self.range
        savedata['hash'] = self.hash
        savedata['PredictionWindow'] = self.PredictionWindow
        savedata['TestedAlgorithms'] = self.TestedAlgorithms
        savedata['results'] = self.results

        with open(self.SaveFile, "wb") as f:
            pickle.dump(savedata, f)

    def run(self, *args, multithread=True, ForceProcesses=None):
        '''Runs the tester and saves the result
        '''

        AlgorithmsToTest = set(args) - self.TestedAlgorithms
        if not AlgorithmsToTest:
            return

        for key in AlgorithmsToTest:
            self.results[key] = np.zeros(
                [len(self.range), self.PredictionWindow])
            self.results[key+" STD"] = np.zeros(
                [len(self.range), self.PredictionWindow])

        self.results['GroundTruth'] = np.zeros(
            [len(self.range), self.PredictionWindow])

        IECs = [IEC(self.data[:(-offset)]) for offset in self.range]

        if multithread:
            if ForceProcesses is None:
                p = Pool(processes=cpu_count() - 2)
            else:
                p = Pool(ForceProcesses)
            FuncMap = p.imap(
                partial(worker, AlgKeys=AlgorithmsToTest),
                IECs)
        else:
            FuncMap = map(
                partial(worker, AlgKeys=AlgorithmsToTest),
                IECs)
        try:
            with tqdm(total=len(IECs), smoothing=0.0) as pbar:
                for offset, result in zip(self.range, FuncMap):

                    index = (offset - self.range[0]) // self.range.step
                    for key in AlgorithmsToTest:
                        std_key = key + " STD"

                        self.results[key][index, :] = result[key].as_matrix()
                        if std_key in result:
                            self.results[std_key][index, :] = result[std_key].as_matrix()

                    self.results['GroundTruth'][index, :] = self.data[-offset - 1
                                                                      :-offset
                                                                      + self.PredictionWindow - 1
                                                                     ][cons_col].as_matrix()
                    pbar.update(1)

            self.TestedAlgorithms.update(AlgorithmsToTest)

        except KeyboardInterrupt:
            pass
        finally:
            if multithread:
                p.terminate()
                p.join()

    def rmse(self):
        '''For each second in the future find the root mean square prediction error
        '''
        rmse = dict()

        for key in self.TestedAlgorithms:
            rmse[key] = [mean_squared_error(
                self.results['GroundTruth'][:, col],
                self.results[key][:, col])**0.5 for col in range(self.PredictionWindow)]

        return rmse

    def simple_prediction(self, offset):

        prediction = dict()
        for key in self.TestedAlgorithms:
            prediction[key] = self.results[key][offset, :]
        prediction['GroundTruth'] = self.results['GroundTruth'][offset, :]

        return prediction

    def average_rmse(self):
        '''Average the RMSE of each algorithms over our runs
        '''

        armse = dict()

        for key in self.TestedAlgorithms:
            rmse = [mean_squared_error(self.results['GroundTruth'][i, :],
                                       self.results[key][i, :]
                                      ) for i in range(self.results[key].shape[0])
                   ]
            armse[key] = np.mean(rmse)
        return armse

    def average_total_error(self):
        ate = dict()

        for key in self.TestedAlgorithms:
            total_error = [abs(np.sum(self.results['GroundTruth'][i, :])
                               - np.sum(self.results[key][i, :])
                              ) for i in range(self.results[key].shape[0])]
            ate[key] = np.mean(total_error)
        return ate

    def similarity_tester(self, offset, method=cosine_similarity):
        pass


def main():
    data = np.loadtxt("dataset.gz2")

    PredictionWindow = 720
    TestingRange = range(PredictionWindow, PredictionWindow + 200, 1)

    tester = IECTester(data, PredictionWindow, TestingRange, SaveFile=None)
    tester.run('ACP', 'Usage Zone Finder',
               'Baseline Finder', multithread=False)


if __name__ == '__main__':
    main()
