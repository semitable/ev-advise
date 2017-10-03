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
    # Create Usage Cost Column
    usage_cost_key = cfg['prices']['usage_cost_key']
    sell_price_key = cfg['prices']['sell_price_key']

    if cfg['location'] == 'UK':
        europe = True
    elif cfg['location'] == 'US':
        europe = False
    else:
        raise ValueError

    print("Setting up prices...", end='')
    start_time = timeit.default_timer()
    if europe:
        # https://economy10.files.wordpress.com/2016/12/economy10-com-survey-results-dec-20163.pdf
        dataset[usage_cost_key] = 0.16  # peak energy

        # off peak hours: https://www.ovoenergy.com/guides/energy-guides/economy-10.html
        dataset.loc[(dataset.index.time > datetime.time(hour=13, minute=00)) & (
            dataset.index.time < datetime.time(hour=16, minute=00)), usage_cost_key] = 0.107

        dataset.loc[(dataset.index.time > datetime.time(hour=20, minute=00)) & (
            dataset.index.time < datetime.time(hour=22, minute=00)), usage_cost_key] = 0.107

        dataset.loc[(dataset.index.time > datetime.time(hour=00, minute=00)) & (
            dataset.index.time < datetime.time(hour=5, minute=00)), usage_cost_key] = 0.107

        # https://www.gov.uk/feed-in-tariffs/overview
        dataset[sell_price_key] = 0.0485
        min_price = 0.0485

    else:
        # summer only: https://www.pge.com/en_US/business/rate-plans/rate-plans/peak-day-pricing/peak-day-pricing.page
        dataset[usage_cost_key] = 0.202
        dataset.loc[(dataset.index.time > datetime.time(hour=8, minute=30)) & (
            dataset.index.time < datetime.time(hour=21, minute=30)), usage_cost_key] = 0.230
        dataset.loc[(dataset.index.time > datetime.time(hour=12, minute=00)) & (
            dataset.index.time < datetime.time(hour=18, minute=00)), usage_cost_key] = 0.253

        dataset[usage_cost_key] = dataset[usage_cost_key]

        min_price = 0.202

    elapsed = timeit.default_timer() - start_time
    print("done ({:0.2f}sec)".format(elapsed))

    print(dataset.describe())

    return dataset
