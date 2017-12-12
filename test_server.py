import logging
import signal
import sys
from multiprocessing import Pool, cpu_count

import pandas as pd

import advise
from advise import write_hdf
from constants import _RESULTS_FILE
from results import SimulatorResults

logger = logging.getLogger('ev-advise-tester')


def get_existing_results():
    hdf = pd.HDFStore(_RESULTS_FILE, complib='zlib', complevel=9)
    existing = []
    for key in hdf.keys():
        meta = hdf.get_storer(key).attrs.meta
        existing.append({'meta': SimulatorResults.argsmeta_from_resultmeta(meta), 'key': key})
    hdf.close()
    return existing


class DelayedKeyboardInterrupt(object):
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        logging.debug('SIGINT received. Delaying KeyboardInterrupt.')

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)


def worker(args):
    return advise.main(*args)


def main():
    existing = [k['meta'] for k in get_existing_results()]
    permutations = SimulatorResults.get_result_permutations()

    not_executed = set(permutations) - set(existing)

    # not_executed = filter(lambda x: x.agent != 'SmartCharge', not_executed)

    args_list = map(SimulatorResults.make_args, not_executed)
    args_list = [e + ['--suppress-tqdm', '--quiet'] for e in args_list]

    print("Number of permutations not executed: ", len(args_list))
    input("Press enter to continue..")

    p = Pool(processes=cpu_count() - 1)

    func_map = p.imap_unordered(worker, args_list)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)

    for result in func_map:
        with DelayedKeyboardInterrupt():
            logger.info('Writing {} result to hdf'.format(result['key']))
            write_hdf(_RESULTS_FILE, result['key'], result['results'],
                      meta=SimulatorResults.result_tuple_from_dict(result['meta']), complib='zlib')


if __name__ == '__main__':
    main()
