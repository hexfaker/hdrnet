from collections import OrderedDict
import os
import argparse
import pandas as pd
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converts filelist to csv containing filenames and params")

    parser.add_argument("filelist", help="os.path to filelist")
    parser.add_argument("-o", dest="result", help="os.path to resultig csv", default='')


    args = parser.parse_args()

    if args.result is None:
        result_fname = os.path.splitext(args.filelist)[0]
        result_fname = f'{result_fname}.csv'
    else:
        result_fname = args.result

    data_root = os.path.dirname(args.filelist)

    with open(args.filelist, 'r') as fl:
        names = [name.strip() for name in fl.readlines()]

    res = OrderedDict()
    for name in names:
        prefix = os.path.splitext(name)[0]

        with open(os.path.join(data_root, 'json', f'{prefix}.json')) as jf:
            params = json.load(jf)

        res[name] = params

    table = pd.DataFrame.from_dict(res, orient='index')

    table.to_csv(result_fname, index_label='name')
