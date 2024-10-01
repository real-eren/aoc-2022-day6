#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass

@dataclass
class DataPoint:
    # The processor these results were obtained from
    cpu: str
    # The value that was passed for "-Ctarget-cpu="
    target_cpu: str
    # The command used to produce the input
    input: str
    algo: str
    threads: int
    best: float | int
    median: float | int
    mean: float | int
    stddev: float | int

def visit_dir(root_path) -> list[DataPoint]:
    """
    Schema of returned value:
    { "processor": { "target_cpu": { "input": stats } } }
    Keys in stats: {'threads', 'best', 'median', 'mean', 'stddev'}
    """
    from os import walk, path
    datapoints: list[DataPoint] = []

    for (_dirpath, _dirnames, filenames) in walk(root_path):
        for filename in filenames:
            if not filename.endswith(".txt"):
                continue
            cpu, target = filename.removesuffix(".txt").split('_')
            with open(path.join(root_path, filename)) as f:
                parse_block(f.readlines(), out=datapoints, cpu=cpu, target=target)
        break # only do root dir
    return datapoints

EXPECTED_HEADER = "name,threads,best,median,mean,stddev"
HEADER_FIELDS = EXPECTED_HEADER.split(',')

def parse_block(lines: list[str], out: list[DataPoint], cpu: str, target: str):
    for line in lines:
        if not line.strip():
            continue
        if line.startswith('>'):
            # start new section
            break
        print("error", line)


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
                cpu=cpu,
                target_cpu=target,
                input=input_name,
                algo=fields[0].strip('"'),
                threads=int(fields[1]),
                best=float(fields[2]),
                median=float(fields[3]),
                mean=float(fields[4]),
                stddev=float(fields[5]),
            ))
        elif line != EXPECTED_HEADER:
            import sys
            print("Error: csv format mismatch, plotting script was designed for a different schema", line)
            sys.exit(1)

def plot_lines(data: dict,
               suptitle_prefix: str,
               xlabel: str,
               ylabel: str = 'Throughput (GB/s)',
               subplot_title_prefix: str = 'CPU: ',
               xscale: str = 'linear'
               ) -> dict[str, plt.Figure]:
    """
    A Figure for each top-level key
    A subplot for each 2nd-level key
    A line for each 3rd-level key (which should have a tuple(xs,ys) as its value)
    """
    figures_dict: dict = {}
    for (t, d) in data.items():
        fig = plot_line(d, xlabel, ylabel, subplot_title_prefix, xscale)
        fig.suptitle(f'{suptitle_prefix}{t}')
        figures_dict[t] = fig
    return figures_dict

def plot_line(data: dict,
              xlabel: str,
              ylabel: str = 'Throughput (GB/s)',
              subplot_title_prefix: str = 'CPU: ',
              xscale: str = 'linear'
              ) -> plt.Figure:
    """
    A subplot for each top-level key
    A line for each 2nd-level key (which should have a tuple(xs,ys) as its value)
    """
    num_lines = len(data)
    fig = plt.figure(layout='constrained')
    for idx, (top, line_datas) in enumerate(data.items(), 1):
        ax = fig.add_subplot(1, num_lines, idx)
        ax.grid()
        for (label, (xs, ys)) in line_datas.items():
            line, = ax.plot(xs, ys, label=label)
        ax.legend()
        ax.set_title(f'{subplot_title_prefix}{top}')
        ax.set_xscale(xscale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    return fig

def plot_histograms(data:dict,
                    suptitle_prefix: str,
                    xlabel: str,
                    subplot_title_prefix: str = 'CPU: ',
                    ylabel: str = 'Throughput (GB/s)',
                    stagger_xticks: bool = False,
                    ) -> dict[str, plt.Figure]:
    """
    A figure for each top-level key
    A subplot for each 2nd-level key
    An xtick for each 3rd-level key
    A column for each 4th-level key

    Returns dict[top-level key, Figure]
    """
    figures_dict: dict = {}
    for (top, dict_per_figure) in data.items():
        fig = plot_histogram(dict_per_figure, xlabel, subplot_title_prefix, ylabel, stagger_xticks)
        fig.suptitle(f'{suptitle_prefix}{top}')
        figures_dict[top] = fig
    return figures_dict

def plot_histogram(data: dict,
                   xlabel: str,
                   subplot_title_prefix: str = 'CPU: ',
                   ylabel: str = 'Throughput (GB/s)',
                   stagger_xticks: bool = False,
                   ) -> plt.Figure:

    """
    A subplot for each top-level key
    An xtick for each 2nd-level key
    A column for each 3rd-level key

    Returns Figure
    """
    fig = plt.figure(layout='constrained')
    width = 0.20
    previous_ax: None | plt.Axes = None
    for idx, (top, by_second) in enumerate(data.items(), 1):
        ax = fig.add_subplot(1, len(data), idx)
        if previous_ax is not None:
            ax.sharey(previous_ax)
        previous_ax = ax

        # sort so that we get similar orders between different plots
        groups = sorted(by_second.keys())
        columns = sorted({third for by_third in by_second.values() for third in by_third})
        xticks_offset = width * (len(columns) - 1) / 2
        x = np.arange(len(groups))

        # have to collect stats along each bar (color)
        # so, for a given fourth, get its values for all third
        algo_stats = {
            third: tuple(by_second.get(second, {}).get(third, 0) for second in groups) for third in columns
        }
        for multiplier, (third, stats_by_fourth) in enumerate(algo_stats.items()):
            offset = width * multiplier
            rects = ax.bar(x + offset, stats_by_fourth, width, label=third)
            # ax.bar_label(rects, padding=3)
        ax.set_title(f'{subplot_title_prefix}{top}')
        ax.set_ylabel(ylabel)
        ax.set_xticks(x + xticks_offset, groups)
        if stagger_xticks:
            for tick in ax.xaxis.get_major_ticks()[1::2]:
                tick.set_pad(25)
        ax.set_xlabel(xlabel)
        ax.legend(columns, loc='upper left')

    return fig

