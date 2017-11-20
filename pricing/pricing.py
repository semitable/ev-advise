"""
Pricing Model 
Filippos Christianos
"""
import datetime

import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
import yaml
# we have implemented two pricing models: US and UK

PRICING_MODELS = ('UK_PRICING', 'US_PRICING')

with open("config/common.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

usage_cost_key = cfg['prices']['usage_cost_key']
sell_price_key = cfg['prices']['sell_price_key']
del cfg


class PricingModel:
    _name = 'Not Implemented'
    _cheap_period = (None, None)
    def setup_prices(self):
        raise NotImplemented()

    def __init__(self, time_index: pd.DatetimeIndex):
        self._billing_period = None

        self._prices = pd.DataFrame(index=time_index, columns=[usage_cost_key, sell_price_key])

        self.setup_prices()

    def start_billing_period(self):
        self._billing_period = {
            'max_demand': 0
        }

    def get_usage_cost(self, usage):
        raise NotImplemented()

    def get_demand_cost(self, demand):
        raise NotImplemented()

    def has_demand_prices(self):
        raise NotImplemented()

    def ideal_charging_cost(self, charge):
        return self._min_price * charge

    def draw_prices(self):
        # plotly fix for timezone correction
        naive_df = self._prices.tz_localize(None)  # (this way we will display in local time, like we want to)
        data = [
            go.Scatter(
                x=naive_df.index,  # assign x as the dataframe column 'x'
                y=naive_df[usage_cost_key],
                name='Usage Cost'
            ),
            go.Scatter(
                x=naive_df.index,  # assign x as the dataframe column 'x'
                y=naive_df[sell_price_key],
                name='Sell Price'
            ),
        ]
        layout = go.Layout(
            title='Plot Title',
            xaxis=dict(
                title='Time (Local Time)',
            ),
            yaxis=dict(
                title='Price',
            )
        )
        fig = go.Figure(data=data, layout=layout)
        py.plot(fig)


class EuropePricingModel(PricingModel):
    _name = 'Europe'
    _cheap_period = (datetime.time(hour=00, minute=00), datetime.time(hour=5, minute=00))
    def has_demand_prices(self):
        return False

    def get_demand_cost(self, demand):
        return 0

    def get_usage_cost(self, usage):
        pbuy = self._prices[usage.index[0]: usage.index[-1]][usage_cost_key]
        psell = self._prices[usage.index[0]: usage.index[-1]][sell_price_key]
        price = pbuy.copy()
        price[usage < 0] = psell
        final = (usage * price).sum()

        return final

    def setup_prices(self):
        # https://economy10.files.wordpress.com/2016/12/economy10-com-survey-results-dec-20163.pdf
        self._prices[usage_cost_key] = 0.16  # peak energy

        # off peak hours: https://www.ovoenergy.com/guides/energy-guides/economy-10.html
        self._prices.loc[(self._prices.index.time > datetime.time(hour=13, minute=00)) & (
            self._prices.index.time < datetime.time(hour=16, minute=00)), usage_cost_key] = 0.107

        self._prices.loc[(self._prices.index.time > datetime.time(hour=20, minute=00)) & (
            self._prices.index.time < datetime.time(hour=22, minute=00)), usage_cost_key] = 0.107

        self._prices.loc[(self._prices.index.time > datetime.time(hour=00, minute=00)) & (
            self._prices.index.time < datetime.time(hour=5, minute=00)), usage_cost_key] = 0.107

        # https://www.gov.uk/feed-in-tariffs/overview
        self._prices[sell_price_key] = 0.0485
        self._min_price = 0.0485


class USPricingModel(PricingModel):
    _name = 'US'
    _cheap_period = (datetime.time(hour=21, minute=30), datetime.time(hour=8, minute=30))
    def has_demand_prices(self):
        return True

    def get_demand_cost(self, demand):
        return demand * 8.03

    def get_usage_cost(self, usage):
        pbuy = self._prices[usage.index[0]: usage.index[-1]][usage_cost_key]
        final = (usage * pbuy).sum()  # pbuy == psell
        return final

    def setup_prices(self):
        # summer only: https://www.pge.com/en_US/business/rate-plans/rate-plans/peak-day-pricing/peak-day-pricing.page
        self._prices[usage_cost_key] = 0.202
        self._prices.loc[(self._prices.index.time > datetime.time(hour=8, minute=30)) & (
            self._prices.index.time < datetime.time(hour=21, minute=30)), usage_cost_key] = 0.230
        self._prices.loc[(self._prices.index.time > datetime.time(hour=12, minute=00)) & (
            self._prices.index.time < datetime.time(hour=18, minute=00)), usage_cost_key] = 0.253

        self._prices[usage_cost_key] = self._prices[usage_cost_key]

        self._min_price = 0.202
