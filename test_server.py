import itertools
import logging
import sys
from multiprocessing import Pool, cpu_count

import pandas as pd

import advise
from advise import write_hdf
from constants import _RESULTS_FILE

logger = logging.getLogger('ev-advise-tester')

def extract_args_from_meta(meta):
    return {
        'pricing': meta['pricing'],
        'agent': meta['agent'],
        'mpc': meta['mpc'],
        'month': meta['month']
    }


pricing_all = ('Europe', 'US')
pricing_arg = ('--uk', '--us')
agents_all = ('SmartCharge', 'Simple-Delayed', 'Simple-Informed', 'Informed-Delayed', 'Dummy')
agents_arg = ('--smartcharge', '--delayed', '--informed', '--informed-delayed', '--dummy')
mpc_all = (True, False)
mpc_arg = (None, '--no-mpc')
months_all = ('Jan2017', 'Feb2017', 'Mar2017', 'Apr2017', 'May2017', 'Jun2017', 'Jul2017')


def get_hash_tuple(args):
    return (args['pricing'], args['agent'], args['mpc'], args['month'])


def get_args_dict_from_tuple(t):
    args = {
        'pricing': t[0],
        'agent': t[1],
        'mpc': t[2],
        'month': t[3]
    }
    return args


def build_args(args_dict):
    pricing = pricing_arg[pricing_all.index(args_dict['pricing'])]
    agent = agents_arg[agents_all.index(args_dict['agent'])]
    mpc = mpc_arg[mpc_all.index(args_dict['mpc'])]
    month_index = months_all.index(args_dict['month'])

    return filter(lambda x: x is not None,
                  [pricing, agent, mpc, '--month-index', str(month_index), '--suppress-tqdm', '--quiet'])


def get_existing_results():
    hdf = pd.HDFStore(_RESULTS_FILE, complib='zlib', complevel=9)
    existing = []
    for key in hdf.keys():
        meta = hdf.get_storer(key).attrs.meta
        existing.append({'meta': meta, 'key': key})
    hdf.close()
    return existing


def worker(args):
    return advise.main(*args)


def main():
    existing = [get_hash_tuple(v['meta']) for v in get_existing_results()]
    permutations = list(itertools.product(pricing_all, agents_all, mpc_all, months_all))

    not_executed = set(permutations) - set(existing)

    not_executed = list(filter(lambda x: x[1] != 'smartcharge', not_executed))

    args_list = [list(build_args(get_args_dict_from_tuple(t))) for t in not_executed]

    print("Number of permutations not executed: ", len(not_executed))
    input("Press enter to continue..")

    p = Pool(processes=cpu_count() - 1)

    func_map = p.imap_unordered(worker, args_list)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)

    for result in func_map:
        logger.info('Writing {} result to hdf'.format(result['key']))
        write_hdf(_RESULTS_FILE, result['key'], result['results'], meta=result['meta'], complib='zlib')


if __name__ == '__main__':
    main()
