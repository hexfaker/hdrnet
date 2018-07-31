from collections import OrderedDict
import os
import argparse
import pandas as pd
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts filelist to csv containing filenames and params for LRNetDataPipeline")

    parser.add_argument("filelist", help="os.path to filelist")
    parser.add_argument("-o", dest="result", help="os.path to resultig csv", default='')

    args = parser.parse_args()

    if args.result is None:
        result_fname = os.path.splitext(args.filelist)[0]
        result_fname = '{}.csv'.format(result_fname)
    else:
        result_fname = args.result

    data_root = os.path.dirname(args.filelist)

    with open(args.filelist, 'r') as fl:
        names = [name.strip() for name in fl.readlines()]

    res = OrderedDict()
    for name in names:
        prefix = os.path.splitext(name)[0]

        with open(os.path.join(data_root, 'json', '{}.json'.format(prefix))) as jf:
            params = json.load(jf)
        params['input'] = os.path.join(data_root, 'input', name)
        params['output'] = os.path.join(data_root, 'output', name)

        res[name] = params

    table = pd.DataFrame.from_dict(res, orient='index') # type: pd.DataFrame

    cols = table.columns.tolist()
    cols = cols[-2:] + cols[:-2]

    table[cols].to_csv(result_fname, index=False)
