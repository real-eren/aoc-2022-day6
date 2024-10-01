#!/usr/bin/env python3

import plot_utils as plots
import matplotlib.pyplot as plt
from plot_utils import DataPoint

datapoints: list[DataPoint] = plots.visit_dir("./data/")

def on_fig(fig, filename):
    fig.set_size_inches(15,7)
    # fig.show()
    fig.savefig(f'./data/plots/{filename}.svg')
    plt.close(fig)

UNIT_DICT = {
    'k': 1000,
    'ki': 1024,
    'm': 1000 ** 2,
    'mi': 1024 ** 2,
    'g': 1000 ** 3,
    'gi': 1024 ** 3,
}
def unit_int(s: str) -> int:
    unit = s[-2:] if s.lower().endswith('i') else s[-1]
    unit_multiplier = UNIT_DICT.get(unit.lower(), 1)
    base = s if unit_multiplier == 1 else s.removesuffix(unit)
    return unit_multiplier * int(base)


def prep_hist(filter, fig_key, subplot_key, group_key, col_key, v = lambda d: d.mean, cmp = lambda old, new: old<new) -> dict:
    dic: dict = {}
    for d in datapoints:
        if not filter(d):
            continue
        key4 = col_key(d)
        v4 = v(d)
        parent_dict = dic.setdefault(fig_key(d), {}).setdefault(subplot_key(d), {}).setdefault(group_key(d), {})
        prior_entry = parent_dict.get(key4)
        if prior_entry is None or cmp(prior_entry, v4):
            parent_dict[key4] = v4
    return dic


def prep_line(filter, fig_key, subplot_key, line_key, xtick_key, v = lambda d: d.mean, cmp = lambda old, new: old<new) -> dict:
    dic = prep_hist(filter, fig_key, subplot_key, line_key, xtick_key, v, cmp)
    return {k1: {k2: {k3: list(zip(*sorted(v3.items()))) for k3,v3 in v2.items()} for k2,v2 in v1.items()} for k1,v1 in dic.items()}


def slim_input(d: DataPoint) -> str:
    return d.input.rsplit(';', maxsplit=1)[0].strip().replace(';', '\n')

def split_input(d: DataPoint) -> str:
    return slim_input(d).strip().replace(';', '\n')


# Effects of compiler option 'target-cpu'
dic = prep_hist(filter=lambda d: d.threads==1 and '981394' in d.input,
                fig_key=lambda d: d.algo,
                subplot_key=lambda d: d.cpu,
                group_key=slim_input,
                col_key=lambda d: d.target_cpu,
                )
figs = plots.plot_histograms(dic, suptitle_prefix='Effect of "target-cpu" on ', xlabel='Input')
for algo, fig in figs.items():
    on_fig(fig, f'Effect_of_target-cpu_on_{algo}')

# Comparing Prime's input to random data on danny vs benny
F2_ALLOWED_INPUTS=("concat(copy(300k, file(./res/data_body)), file(./res/data_end))",
                   "concat(rng(x, 3456), srand(600m, x), lit(qwertyuiopasdfgh))",
                   )
dic = prep_hist(filter=lambda d: d.threads==1 and d.input.split(';')[0].strip().lower() in F2_ALLOWED_INPUTS and d.algo in ['david_a_perez', 'benny'],
                fig_key=lambda d: '',
                subplot_key=lambda d: d.cpu,
                group_key=split_input,
                col_key=lambda d: d.algo,
                )
fig = plots.plot_histogram(dic[''], xlabel='Input', stagger_xticks=True)
fig.suptitle("Disparity between Prime's data and random data")
on_fig(fig, 'prime_data_vs_random')


# David with varying rand seq lengths
# "concat(rng(x,$i), copy($((300000/$i)), srand(${i}k, x)), lit(zxcvbnmasdfghjkl))"
dic = prep_line(filter=lambda d: d.threads==1 and d.algo == 'david_a_perez' and "zxcvbnmasdfghjkl" in d.input,
                fig_key=lambda d: '',
                subplot_key=lambda d: d.cpu,
                line_key=lambda d: d.algo,
                xtick_key=lambda d: unit_int(d.input.split('srand(')[1].split(',', maxsplit=1)[0]),
                )
fig = plots.plot_line(dic[''], xlabel='Rand Seq Length (Bytes)')
fig.suptitle('`david_a_perez` mean throughput with varying rand seq length\nInput: "concat(rng(x,$i), copy($((300000/$i)), srand(${i}k, x)), lit(zxcvbnmasdfghjkl))"')
on_fig(fig, 'david_varying_rand_length')


# IPC effects w/ hyperthreading among Benny variants
dic = prep_line(filter=lambda d: d.algo.startswith('benny') and d.input.split(';')[0].strip() == "copy(400M, lit(c))",
                fig_key=lambda d: '',
                subplot_key=lambda d: d.cpu,
                line_key=lambda d: d.algo,
                xtick_key=lambda d: d.threads,
                )
fig = plots.plot_line(dic[''], xlabel='# threads')
fig.suptitle('Thread scaling for Benny variants, input: `copy(400M, lit(c))`')
on_fig(fig, 'benny_thread_scaling')


def david_and_conflict_filter(d: DataPoint) -> bool:
    return d.threads == 1 \
        and d.input.startswith('copy(') \
        and 'lit(a' in d.input \
        and d.algo in ['david_a_perez', 'benny', 'conflict_mc9b']

# Pathological for David and conflict
dic = prep_hist(filter=lambda d: '30.000MB' not in d.input and david_and_conflict_filter(d),
                fig_key=lambda d: '',
                subplot_key=lambda d: d.cpu,
                group_key=lambda d: d.algo,
                col_key=split_input,
                )
fig = plots.plot_histogram(dic[''], xlabel='Algorithm')
fig.suptitle('Pathological Case for David and Conflict')
on_fig(fig, 'david_worst_case')


# Best case for David and conflict
dic = prep_hist(filter=lambda d: '30.000MB' in d.input and david_and_conflict_filter(d),
                fig_key=lambda d: '',
                subplot_key=lambda d: d.cpu,
                group_key=lambda d: d.algo,
                col_key=split_input,
                )
fig = plots.plot_histogram(dic[''], xlabel='Algorithm')
fig.suptitle('Best Case for David and Conflict')
on_fig(fig, 'david_best_case')


# Loads
dic = prep_line(filter=lambda d: d.algo.startswith('load') and 'lit(z)' in d.input,
                fig_key=lambda d: '',
                subplot_key=lambda d: d.cpu,
                line_key=lambda d: d.algo,
                xtick_key=lambda d: d.threads,
                )
fig = plots.plot_line(dic[''], xlabel='# Threads')
fig.suptitle('Loads, input: "copy(500M, lit(z)))"')
on_fig(fig, 'loads_compare')


# IPC gains with conflicts
dic = prep_line(filter=lambda d: d.threads==1 and d.algo.startswith('conflict_mc'),
                fig_key=lambda d: '',
                subplot_key=lambda d: d.cpu,
                line_key=split_input,
                xtick_key=lambda d: int(d.algo.removeprefix('conflict_mc').removesuffix('b')),
                )
fig = plots.plot_line(dic[''], xlabel='Unroll Factor')
fig.suptitle('IPC Gains from Unrolling Conflict')
on_fig(fig, 'conflicts_compare')


# Gather, hiding latency of gather AVX512 pre,nopre
dic = prep_hist(filter=lambda d: d.threads==1 and d.input.startswith('copy(100M, lit(a))') and d.algo in ['gather_avx512_pre', 'gather_avx512_nopre'],
                fig_key=lambda d: '',
                subplot_key=lambda d: d.cpu,
                group_key=split_input,
                col_key=lambda d: d.algo,
                )
fig = plots.plot_histogram(dic[''], xlabel='Input')
fig.suptitle('Hiding Latency of Gathers')
on_fig(fig, 'gather_avx512_latency_compare')


# Gathers w/ and w/out chunking in the presence of many windows
dic = prep_line(filter=lambda d: d.threads==1 and d.algo.startswith('gather') and d.input.startswith("concat(rng(x, 3456), drand("),
                fig_key=lambda d: '',
                subplot_key=lambda d: d.cpu,
                line_key=lambda d: d.algo,
                xtick_key=lambda d: unit_int(d.input.removeprefix("concat(rng(x, 3456), drand(").split(',')[0].strip()),
                )
fig = plots.plot_line(dic[''],
                        subplot_title_prefix='CPU: ',
                        xlabel='Distance between windows (bytes)', xscale='log')
fig.suptitle('Chunking with inputs of varying density')
on_fig(fig, 'gather_chunking_compare_on_dense_inputs')


# AVX2 gather w/ and w/out manual register spilling
dic = prep_hist(filter=lambda d: d.threads==1 and d.algo.startswith('gather_avx2') and d.input.startswith("copy(100M, lit(a))"),
                fig_key=lambda d: '',
                subplot_key=lambda d: d.cpu,
                group_key=lambda d: d.target_cpu,
                col_key=lambda d: d.algo,
                )
fig = plots.plot_histogram(dic[''], xlabel='Target-CPU')
fig.suptitle('AVX2 Gathers w/ & w/out Manual Register Spilling')
on_fig(fig, 'gather_avx2_compare_reg')


# Final comparisons
final_algos = ['david_a_perez', 'benny', 'benny_x2', 'gather_avx2_few_chnk', 'conflict_mc10b', 'gather_avx512_chunks']
dic =  prep_hist(filter=lambda d: d.threads==1 and d.algo in final_algos and d.input.startswith("concat(rng(x, 981394)"),
                 fig_key=lambda d: '',
                 subplot_key=lambda d: d.cpu,
                 group_key=lambda d: d.input,
                 col_key=lambda d: d.algo,
                 )
fig = plots.plot_histogram(dic[''], xlabel='Input')
fig.suptitle('Final Comparison')
on_fig(fig, 'final_comparison')

# input() # don't close immediately (for fig.show())
