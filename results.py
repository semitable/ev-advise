from collections import namedtuple


class SimulatorResults():
    args_list = {
        'pricing': ('Europe', 'US'),
        'agent': ('SmartCharge', 'Simple-Delayed', 'Simple-Informed', 'Informed-Delayed', 'Dummy'),
        'mpc': (True, False),
        'ier_type': ('solar', 'wind'),
        'month': ('Jan2017', 'Feb2017', 'Mar2017', 'Apr2017', 'May2017', 'Jun2017', 'Jul2017'),
        'online-periods': ('night-only', 'with-lunch'),
    }
    args_map = {
        'pricing': {
            'Europe': '--uk',
            'US': '--us',
        },
        'agent': {
            'SmartCharge': '--smartcharge',
            'Simple-Delayed': '--delayed',
            'Simple-Informed': '--informed',
            'Informed-Delayed': '--informed-delayed',
            'Dummy': '--dummy',
        },
        'mpc': {
            True: None,
            False: '--no-mpc',
        },
        'ier_type': {
            'solar': '--solar-only',
            'wind': '--wind-only',
        },
        'month': {
            'Jan2017': '--month-index 0',
            'Feb2017': '--month-index 1',
            'Mar2017': '--month-index 2',
            'Apr2017': '--month-index 3',
            'May2017': '--month-index 4',
            'Jun2017': '--month-index 5',
            'Jul2017': '--month-index 6',
        },
        'online_periods': {
            'night-only': None,
            'with-lunch': '--lunch-break',
        },
    }

    ResultMeta = namedtuple('ResultMeta', ['execution_date', 'seed',
                                           'agent', 'actions', 'month', 'mpc', 'online_periods', 'pricing', 'ier_type',
                                           'house_scale', 'solar_scale', 'wind_scale'])

    @staticmethod
    def result_tuple_from_dict(d):
        return SimulatorResults.ResultMeta(**{k.replace("-", "_"): v for k, v in d.items()})

    @staticmethod
    def make_args(t: ResultMeta):
        args = [SimulatorResults.args_map[k][v] for k, v in t._asdict().items() if
                k in SimulatorResults.args_map and SimulatorResults.args_map[k][v] is not None]
        # and split them (mostly for the --month-index X thing)
        return [w for segments in args for w in segments.split()]
