from collections import namedtuple, OrderedDict
from itertools import chain, product

class SimulatorResults():
    args_list = OrderedDict([
        ('pricing', ('Europe', 'US')),
        ('agent', ('SmartCharge', 'Simple-Delayed', 'Simple-Informed', 'Informed-Delayed', 'Dummy')),
        ('mpc', (True, False)),
        ('ier_type', ('solar', 'wind')),
        ('month', ('Jan2017', 'Feb2017', 'Mar2017', 'Apr2017', 'May2017', 'Jun2017', 'Jul2017')),
        ('online_periods', ('night-only', 'with-lunch')),
        ('actions', ('two', 'five')),

    ])
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
        'actions': {
            'two': '--battery-actions two',
            'five': '--battery-actions five',
        },

    }


    @staticmethod
    def result_tuple_from_dict(d):
        return ResultMeta(**{k.replace("-", "_"): v for k, v in d.items()})

    @classmethod
    def make_args(cls, t):
        args = [cls.args_map[k][v] for k, v in t._asdict().items() if
                k in cls.args_map and cls.args_map[k][v] is not None]
        # and split them (mostly for the --month-index X thing)
        return [w for segments in args for w in segments.split()]

    @classmethod
    def argsmeta_from_resultmeta(cls, m):
        return ArgsMeta(**{k: v for k, v in m._asdict().items() if k in cls.args_list})

    @classmethod
    def get_result_permutations(cls):
        """
        :return: a list of all possible permutations of ArgsMeta
        """
        p = product(*cls.args_list.values())
        return [ArgsMeta(**{k: v for k, v in zip(cls.args_list.keys(), perm)}) for perm in p]


ArgsMeta = namedtuple('ArgsMeta', SimulatorResults.args_list.keys())
ResultMeta = namedtuple('ResultMeta', chain(['execution_date', 'seed'],
                                            SimulatorResults.args_list.keys(),
                                            ['house_scale', 'solar_scale', 'wind_scale']))


def rfilter(k, r):
    """
    
    :param k: the attribute name of the ResultMeta tuple (e.g. 'ier_type')
    :param r: the value we want to check (e.g. 'wind')
    :return: 
    """

    def f(x):
        return getattr(x, k) == r

    return f
