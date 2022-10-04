import argparse
import json
from collections import defaultdict
from itertools import groupby

# parse command line arguments
parser = argparse.ArgumentParser(description='BIDSify the MS brain database.')
parser.add_argument('-i', '--input', help='Paths of list of multiple json files we want to merge.', nargs='+', default=[], required=True)
parser.add_argument('-o', '--output', help='Path to output filename.', required=True)

args = parser.parse_args()

# check if all elements in list are identical
def check_identical(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)

# read all jsons into dict list
dict_list = []
for i in range(len(args.list)):
    with open(args.list[i]) as f:
        d = json.load(f)
    dict_list.append(d)

dd = defaultdict(list)
for d in dict_list:
    for key, value in d.items():
        dd[key].append(value)
dd = dict(dd)
dd_copy = dd.copy()

# reduce entries if identical to one element
for key, values in dd.items():
    for value in values:
        if check_identical(values):
            dd_copy[key]=values[0]

# save to new dict
with open(args.output, 'w') as fp:
    json.dump(dd_copy, fp)
