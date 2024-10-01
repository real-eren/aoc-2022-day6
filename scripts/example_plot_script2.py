#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
import sys
from os import path

"""
input:
> concat(rng(x,310), copy(967, srand(310k, x)), lit(qwertyuiopasdfghjkl)); 299.770MB; first window at 99.99999366180778%
name,threads,times...
"david_a_perez",1,172790563,166763938,164031312,162629309,161493903,160168280,159218572

uses file name in figure title
"""

@dataclass
class DataPoint:
    # The command used to produce the input
    input: str
    algo: str
    threads: int
    times: list[int]

def parse_block(lines: list[str]) -> list[DataPoint]:
    for line in lines:
        if not line.strip():
            continue
        if line.startswith('>'):
            # start new section
            break
        # basic attempt to catch invalid files
        print("error", line)

    out = []
    for _line in lines:
        line = _line.strip()
        if not line:
            continue
        if line.startswith('>'):
            # start new section
            input_name = line.removeprefix('>').strip()
        elif line.startswith('"'):
            #name,threads,best,median,mean,stddev
            fields: list = line.split(',')
            out.append(DataPoint(
                input=input_name,
                algo=fields[0].strip('"'),
                threads=int(fields[1]),
                times=list(map(int, fields[2:])),
            ))
        elif line != "name,threads,times...":
            import sys
            print("Error: csv format mismatch, plotting script was designed for a different schema", line)
            sys.exit(1)
    return out

if len(sys.argv) == 1 or sys.argv[1].lower() in ["--help", "-h", "help"]:
    print("plotting script. Expects `{cmd} in_file_path out_file_path`")
    sys.exit(1)
in_path = sys.argv[1]
out_path = sys.argv[2] if len(sys.argv) >= 3 else None

dataset_name = path.basename(in_path).removesuffix(".times")
with open(in_path) as f:
    datapoints = parse_block(f.readlines())
fig = plt.figure(layout='constrained')
ax = fig.add_subplot(1, 1, 1)
ax.grid()
for datapoint in datapoints:
    label = datapoint.input.rsplit(';', 1)[0]
    ax.plot(np.arange(1, len(datapoint.times) + 1), datapoint.times, label=label)
ax.legend()
ax.set_title(dataset_name)
ax.set_xlabel("Trial number")
ax.set_ylabel('Time (ns)')
fig.set_size_inches(15,7)

if out_path is None:
    fig.show()
    input('enter to close')
else:
    fig.savefig(out_path)
