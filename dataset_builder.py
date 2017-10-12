import timeit
import pandas as pd
import datetime


def build_dataset(cfg):

    dataset_filename = cfg['dataset']['filename']
    dataset_tz = cfg['dataset']['timezone']

    print("Reading dataset...", end='')
    start_time = timeit.default_timer()
    dataset = pd.read_csv(dataset_filename, parse_dates=[0], index_col=0).tz_localize('UTC').tz_convert(dataset_tz)
    elapsed = timeit.default_timer() - start_time
    print("done ({:0.2f}sec)".format(elapsed))

    # trimming the dataset because before that we have bad values
    dataset = dataset[datetime.datetime(2012, 9, 7, 0, 0, 0):]

    # scaling down the WTG
    dataset['WTG Production'] = cfg['adjustment']['wtg-scale'] * dataset['WTG Production']
    dataset['WTG Prediction'] = cfg['adjustment']['wtg-scale'] * dataset['WTG Prediction']

    # #scale up the house
    dataset['House Consumption'] = cfg['adjustment']['house-scale'] * dataset['House Consumption']

    # fixing december
    dataset['WTG Production'][datetime.datetime(2012, 12, 1): datetime.datetime(2012, 12, 12)] = dataset[
                                                                                                     'WTG Production'][
                                                                                                 datetime.datetime(2013,
                                                                                                                   1,
                                                                                                                   20): datetime.datetime(
                                                                                                     2013, 1, 31)]
    dataset['WTG Prediction'][datetime.datetime(2012, 12, 1): datetime.datetime(2012, 12, 12)] = dataset[
                                                                                                     'WTG Prediction'][
                                                                                                 datetime.datetime(2013,
                                                                                                                   1,
                                                                                                                   20): datetime.datetime(
                                                                                                     2013, 1, 31)]

    print(dataset.describe())

    return dataset
