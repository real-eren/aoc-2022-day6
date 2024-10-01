#![feature(avx512_target_feature)]
#![feature(stdarch_x86_avx512)]

mod eren_impls;
mod input_gen;

use crate::eren_impls::*;
use input_gen::{eval, new_node_arena, parse};

use std::{
    collections::HashSet, env::args, hint::black_box, io::Write as _, ops::Div, time::Duration,
};

/// Bandwidth test
#[target_feature(enable = "avx512f")]
#[no_mangle]
unsafe fn avx512_load64(input: &[u8]) -> Option<usize> {
    use std::arch::x86_64::_mm512_loadu_ps;
    const OVERLAP: usize = 13;
    const CHUNK_SIZE: usize = 64;
    if input.len() < 14 {
        return None;
    }

    // load 64 bytes, overlap 16
    // use rotates to find 14 bytes same
    let mut idx = 0;
    while let Some(slice) = input.get(idx..(idx + CHUNK_SIZE)) {
        let v = _mm512_loadu_ps(slice.as_ptr().cast());
        black_box(v);
        idx += CHUNK_SIZE - OVERLAP;
    }
    None
}

/// Bandwidth test
#[target_feature(enable = "avx512f")]
#[no_mangle]
unsafe fn avx512_loadgather_4bx16(input: &[u8]) -> Option<usize> {
    use std::arch::x86_64::_mm512_loadu_epi32;
    let mut offsets = [0u32; 16];
    for i in 0..16 {
        offsets[i] = (i * input.len() / 16).try_into().expect("");
    }
    let offsets = _mm512_loadu_epi32(offsets.as_ptr().cast());
    // black_box stores val to stack, simpler to just write the darn asm
    std::arch::asm!(
        "2:
	vpxor {z1:x}, {z1:x}, {z1:x}
	kxnorw {k1}, {k0}, {k0}
	vpgatherdd {z1} {{{k1}}}, zmmword ptr [{ptr} + {offsets}]
	vpxor {z2:x}, {z2:x}, {z2:x}
	kxnorw {k2}, {k0}, {k0}
	vpgatherdd {z2} {{{k2}}}, zmmword ptr [{ptr} + {offsets} + 4]
	add {ptr}, 8
	dec {rem_iters}
	jne 2b
",
        offsets = in(zmm_reg) offsets,
        z1 = out(zmm_reg) _,
        z2 = out(zmm_reg) _,
        k0 = out(kreg) _,
        k1 = out(kreg) _,
        k2 = out(kreg) _,
        ptr = inout(reg) input.as_ptr() => _,
        rem_iters = inout(reg) input.len() / 16 / 8 => _
    );
    None
}

/// Bandwidth test
#[target_feature(enable = "avx,avx512f")]
#[no_mangle]
unsafe fn avx512_loadgather_8bx8(input: &[u8]) -> Option<usize> {
    use std::arch::x86_64::_mm256_loadu_si256;
    let mut offsets = [0u32; 8];
    for i in 0..offsets.len() {
        offsets[i] = (i * input.len() / offsets.len()).try_into().expect("");
    }
    let offsets = _mm256_loadu_si256(offsets.as_ptr().cast());
    std::arch::asm!(
        "2:
	vpxor {z1:x}, {z1:x}, {z1:x}
	kxnorw {k1}, {k0}, {k0}
	vpgatherdq {z1} {{{k1}}}, zmmword ptr [{ptr} + {offsets}]
	vpxor {z2:x}, {z2:x}, {z2:x}
	kxnorw {k2}, {k0}, {k0}
	vpgatherdq {z2} {{{k2}}}, zmmword ptr [{ptr} + {offsets} + 8]
	vpxor {z3:x}, {z3:x}, {z3:x}
	kxnorw {k3}, {k0}, {k0}
	vpgatherdq {z3} {{{k3}}}, zmmword ptr [{ptr} + {offsets} + 16]
	add {ptr}, 24
	dec {rem_iters}
	jne 2b
",
        offsets = in(ymm_reg) offsets,
        z1 = out(zmm_reg) _,
        z2 = out(zmm_reg) _,
        z3 = out(zmm_reg) _,
        k0 = out(kreg) _,
        k1 = out(kreg) _,
        k2 = out(kreg) _,
        k3 = out(kreg) _,
        ptr = inout(reg) input.as_ptr() => _,
        rem_iters = inout(reg) input.len() / 8 / 24 => _
    );
    None
}

/// Bandwidth test
#[no_mangle]
unsafe fn sse2_load16(input: &[u8]) -> Option<usize> {
    use std::arch::x86_64::_mm_loadu_si128;
    const CHUNK_SIZE: usize = 16;
    let mut idx = 0;
    while let Some(slice) = input.get(idx..(idx + CHUNK_SIZE)) {
        let v = _mm_loadu_si128(slice.as_ptr().cast());
        black_box(v);
        idx += CHUNK_SIZE;
    }
    None
}

/// Bandwidth test
#[no_mangle]
unsafe fn load8(input: &[u8]) -> Option<usize> {
    let (l, m, r) = input.align_to::<u64>();
    for b in l.iter().copied() {
        black_box(b);
    }
    for b in m.iter().copied() {
        black_box(b);
    }
    for b in r.iter().copied() {
        black_box(b);
    }
    None
}

/// Bandwidth test
#[no_mangle]
unsafe fn load1(input: &[u8]) -> Option<usize> {
    for b in input.iter().copied() {
        black_box(b);
    }
    None
}

#[inline(never)]
pub fn benny(input: &[u8]) -> Option<usize> {
    if input.len() < 14 {
        return None;
    }
    let mut filter = 0u32;
    input
        .iter()
        .take(14 - 1)
        .for_each(|c| filter ^= 1 << (c % 32));

    input.windows(14).position(|w| {
        let first = w[0];
        let last = w[w.len() - 1];
        filter ^= 1 << (last % 32);
        let res = filter.count_ones() == 14;
        filter ^= 1 << (first % 32);
        res
    })
}

/// SAFETY: Uses popcnt intrinsic
#[inline(never)]
#[target_feature(enable = "popcnt")]
pub unsafe fn benny_popcount(input: &[u8]) -> Option<usize> {
    if input.len() < 14 {
        return None;
    }
    let mut filter = 0u32;
    input
        .iter()
        .take(14 - 1)
        .for_each(|c| filter ^= 1 << (c % 32));

    input.windows(14).position(|w| {
        let first = w[0];
        let last = w[w.len() - 1];
        filter ^= 1 << (last % 32);
        let res = filter.count_ones() == 14;
        filter ^= 1 << (first % 32);
        res
    })
}

#[inline(never)]
fn david_a_perez(input: &[u8]) -> Option<usize> {
    let mut idx = 0;
    while let Some(slice) = input.get(idx..idx + 14) {
        let mut state = 0u32;

        if let Some(pos) = slice.iter().rposition(|byte| {
            let bit_idx = byte % 32;
            let ret = state & (1 << bit_idx) != 0;
            state |= 1 << bit_idx;
            ret
        }) {
            idx += pos + 1;
        } else if state.count_ones() == 14 {
            return Some(idx);
        }
    }
    None
}

const ID_TO_FN: &[(&str, unsafe fn(&[u8]) -> Option<usize>)] = &[
    ("benny", benny),
    ("benny_popcnt", benny_popcount),
    ("benny_alt", benny_alt),
    ("benny_x2", bbeennnnyy),
    ("gather_avx512_pre", gather_avx512_prefetch),
    ("gather_avx512_chunks", gather_avx512_chunked),
    ("gather_avx512_nopre", gather_avx512_noprefetch),
    ("gather_avx2", gather_avx2),
    ("gather_avx2_chnk", gather_avx2_chunked),
    ("gather_avx2_few_regs", gather_avx2_few_regs),
    ("gather_avx2_few_chnk", gather_avx2_few_chunked),
    ("david_a_perez", david_a_perez),
    ("conflict", conflict),
    ("conflict_mc1b", conflict_mc1b),
    ("conflict_mc2b", conflict_mc2b),
    ("conflict_mc3b", conflict_mc3b),
    ("conflict_mc4b", conflict_mc4b),
    ("conflict_mc5b", conflict_mc5b),
    ("conflict_mc6b", conflict_mc6b),
    ("conflict_mc7b", conflict_mc7b),
    ("conflict_mc8b", conflict_mc8b),
    ("conflict_mc9b", conflict_mc9b),
    ("conflict_mc10b", conflict_mc10b),
    ("conflict_mc11b", conflict_mc11b),
    ("conflict_mc12b", conflict_mc12b),
    ("load_64B", avx512_load64),
    ("loadgather_4Bx16", avx512_loadgather_4bx16),
    ("loadgather_8Bx8", avx512_loadgather_8bx8),
    ("load_16B", sse2_load16),
    ("load_8B", load8),
    ("load_1B", load1),
];

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy)]
enum FeatureDetector {
    avx2,
    avx512f,
    avx512bw,
    avx512cd,
    avx512vpopcntdq,
    bmi1,
    popcnt,
    sse2,
}
impl FeatureDetector {
    fn is_met(&self) -> bool {
        match self {
            FeatureDetector::avx2 => is_x86_feature_detected!("avx2"),
            FeatureDetector::avx512f => is_x86_feature_detected!("avx512f"),
            FeatureDetector::avx512bw => is_x86_feature_detected!("avx512bw"),
            FeatureDetector::avx512cd => is_x86_feature_detected!("avx512cd"),
            FeatureDetector::avx512vpopcntdq => is_x86_feature_detected!("avx512vpopcntdq"),
            FeatureDetector::bmi1 => is_x86_feature_detected!("bmi1"),
            FeatureDetector::popcnt => is_x86_feature_detected!("popcnt"),
            FeatureDetector::sse2 => is_x86_feature_detected!("sse2"),
        }
    }
}

/// In order to give a useful message for functions with features absent on the current platform,
/// We associate function ids to feature lists
const ID_TO_REQUIRED_FEATURES: &[(&str, &[FeatureDetector])] = &{
    use FeatureDetector as FD;
    let mut list = [("", &[] as &[FD]); ID_TO_FN.len()];
    let mut i = 0;
    while i < list.len() {
        let id = ID_TO_FN[i].0;
        list[i].0 = id;
        // what not having const methods does to a mf
        list[i].1 = match id.as_bytes() {
            &[b'c', b'o', b'n', b'f', b'l', b'i', b'c', b't', ..] => &[FD::avx512f, FD::avx512cd],
            &[b'g', b'a', b't', b'h', b'e', b'r', b'_', b'a', b'v', b'x', b'5', b'1', b'2', ..] => {
                &[FD::avx512f, FD::avx512bw, FD::avx512vpopcntdq, FD::bmi1]
            }
            &[b'g', b'a', b't', b'h', b'e', b'r', b'_', b'a', b'v', b'x', b'2', ..] => &[FD::avx2],
            b"load_64B" | b"loadgather_8Bx8" | b"loadgather_4Bx16" => &[FD::avx512f],
            b"load_16B" => &[FD::sse2],
            b"benny_popcnt" | b"benny_x2" => &[FD::popcnt],
            _ => &[],
        };
        i += 1;
    }

    list
};

/// Group ID to ID prefix
const GROUPS: &[(&str, &str)] = &[
    ("bennys", "benny"),
    ("gathers", "gather"),
    ("loads", "load"),
    ("gather_avx2s", "gather_avx2"),
    ("gather_avx512s", "gather_avx512"),
    ("conflicts", "conflict"),
    ("all", ""),
];

fn find_unmet_req(name: &str) -> Option<FeatureDetector> {
    ID_TO_REQUIRED_FEATURES
        .iter()
        .copied()
        .find_map(|(id, reqs)| if name == id { Some(reqs) } else { None })
        .iter()
        .copied()
        .flatten()
        .copied()
        .find(|req| !req.is_met())
}

/// How to format the results
#[derive(Clone, Copy)]
enum OutputFormat {
    Pretty,
    Csv,
}

/// What stat to report
#[derive(Clone, Copy, PartialEq, Eq)]
enum OutputMode {
    Throughput,
    Time,
    /// Just list the measurements, as nanoseconds
    Raw,
}

struct CliOptions {
    /// List of mt values to test
    mts: Vec<usize>,
    /// List of algos to test, and their names
    fns: Vec<(&'static str, unsafe fn(&[u8]) -> Option<usize>)>,
    /// How many iterations to run each algo for
    iter_count: usize,
    /// How to format the results
    output_format: OutputFormat,
    /// What stat to report
    output_mode: OutputMode,
    /// if true, just run in a loop.
    /// Else, also track times and validate output
    /// useful for profiling
    just_run: bool,
    /// Whether to perform warm-up iterations
    warm_up: bool,
}

impl CliOptions {
    fn new() -> Self {
        Self {
            mts: vec![1],
            fns: Vec::new(),
            iter_count: 30,
            output_format: OutputFormat::Pretty,
            output_mode: OutputMode::Throughput,
            just_run: false,
            warm_up: true,
        }
    }
}

const HELP_MESSAGE: &str = "
A benchmark suite for AoC 2022 Day 6 part 2.

Arguments are interpreted as they are encountered, so flags don't affect previous inputs.

Examples:
- Run a benchmark suite with several functions and inputs
`{exe} --fns=bennys,gathers --iters=50 \"copy(10M, lit(a))\" \"concat(rng(x,123), srand(60M, x))\" \"file(./data/input.txt)\"` 
- Run a single function on an input in a loop, no stats. Useful for profiling.
`{exe} --fns=david_a_perez --iters=500 --just-run \"file(./data/input.txt)\"`
- Run all functions on one input, and output stats as a CSV. Useful for scripting.
`{exe} --fns=all --csv \"file(./data/input.txt)\"`

Use --help=dsl for information on the input generation DSL.
Use --help=fns for a list of the functions and groups of functions.

--iters=N  | Warm-up for `N/2` iterations and run each function for `N` iterations.
           | Default 30

--warm-up=B| Enable or disable the warm-up iterations.
           | Default 'yes'

--threads=N| Comma-separated list of thread counts to use.
           | 0 is treated as the number of logical cores detected.
           | Default [1,]
           | Ex: --threads=1,2,4,0

--pretty   | Output data as human readable columns.
           | This is the default display mode.
           | For easier parsing, see '--csv'.
           | For raw data points, see '--rawdata'
           | Example output, reporting time:
           | > copy(10M, lit(a)); 10.000MB; no windows present
           |         name   threads         best       median         mean       stddev
           |        benny         1    15.3719ms    16.8229ms    16.7728ms   299.5050µs
           | benny_popcnt         1     7.2099ms     7.2128ms     7.2259ms    18.9200µs
    
--csv      | Output data as a csv-like format
           | Times are reported in integer nanoseconds.
           | Throughput is still GB/s, float.
           | Example output: 
           | > copy(10M, lit(a)); 10.000MB; no windows present
           | name,threads,best,median,mean,stddev
           | \"benny\",1,14502165,14664584,14654069,42759
           | \"benny_popcnt\",1,7209638,7211999,7218647,12514

--report=time
           | Report stats as time.
           | --pretty will use human-readable units
           | --csv will use nanoseconds
--report=thrpt
--report=throughput
           | Report stats as GB/s (index of window / time)
--report=raw
           | Output the individual measurements, in chronological order, as time in nanoseconds 
           | Example output:
           | > copy(10M, lit(a)); 10.000MB; no windows present
           | name,threads,times...
           | \"benny\",1,16748483,16793353,16766192,16766403,16846212
           | \"benny_popcnt\",1,8346281,8339932,8341112,8340582,8340382

--skip     | Prefix any argument with --skip to ignore it.

--fns=     | Set, remove or add functions or groups of functions.
--fns-=    | Comma separated values. Duplicates are preserved.
--fns+=    | Default empty
           | Ex: --fns=all  --fns-=benny,gathers,conflicts
";

// "We have Criterion at home":
// justification: very heavy dependency
fn main() {
    const FILE_BUF_BASE_CAPACITY: usize = 512 * 1024 * 1024;

    let mut times = Vec::<Duration>::new();
    let mut file_buf = Vec::<u8>::with_capacity(FILE_BUF_BASE_CAPACITY);

    let mut cli_state = CliOptions::new();

    'arg_loop: for arg in args().skip(1) {
        file_buf.clear();
        file_buf.shrink_to(FILE_BUF_BASE_CAPACITY);

        // the arg parsing code isn't optimized, but it's only dealing with a few dozen strings of
        // a few bytes each.
        if arg == "-h" || arg == "--help" {
            println!("{HELP_MESSAGE}");
            break 'arg_loop;
        } else if let Some(help_type) = arg.strip_prefix("--help=") {
            match help_type {
                "fns" => {
                    println!("\tFunction list:");
                    for (fn_id, _) in ID_TO_FN {
                        println!("{fn_id}");
                    }
                    println!("\tGroup list (selects functions with matching prefix):");
                    for (group_name, group_prefix) in GROUPS {
                        println!("{group_name: <14} => '{group_prefix}'");
                    }
                }
                "dsl" => {
                    println!("{}", input_gen::USAGE_MESSAGE);
                }
                _ => {
                    eprintln!("Error: '{help_type}' is not a valid help category, try one of 'fns', 'dsl'")
                }
            }
            break 'arg_loop;
        } else if arg.starts_with("--skip") {
            // useful for temporarily disabling some option
        } else if let Some(rem_arg) = arg.strip_prefix("--fns") {
            // then =, += or -=
            let Some((modifier, fn_list_str)) = rem_arg.split_once('=') else {
                eprintln!("Error: --fns should be followed by '=' or '-=' or '+='.");
                continue 'arg_loop;
            };
            let mut fn_list = Vec::new();
            for name in fn_list_str.split(',').map(|s| s.trim()) {
                if let Some(pair) = ID_TO_FN.iter().find(|(id, _)| *id == name).copied() {
                    fn_list.push(pair);
                } else if let Some((_, grp_prefix)) =
                    GROUPS.iter().find(|(grp_name, _)| *grp_name == name)
                {
                    fn_list.extend(ID_TO_FN.iter().filter(|(id, _)| id.starts_with(grp_prefix)));
                } else {
                    eprintln!(
                        "Error: {name} in {fn_list_str} is not a recognized fn name or group name"
                    );
                    continue 'arg_loop;
                }
            }
            match modifier {
                "" => {
                    cli_state.fns.clear();
                    cli_state.fns.extend_from_slice(&fn_list);
                }
                "-" => {
                    cli_state.fns.retain(|e| !fn_list.contains(e));
                }
                "+" => {
                    cli_state.fns.extend_from_slice(&fn_list);
                }
                _ => {
                    eprintln!("Error: --fns should be followed by '=' or '-=' or '+='. Got '{modifier}' instead.");
                }
            };
        } else if let Some(rem_arg) = arg.strip_prefix("--threads=") {
            cli_state.mts.clear();
            for num in rem_arg.split(',') {
                match num.parse::<usize>() {
                    Ok(0) => match std::thread::available_parallelism() {
                        Ok(v) => cli_state.mts.push(v.get()),
                        Err(e) => {
                            eprintln!("Error: failed to query available_parallelism, '{e}'")
                        }
                    },
                    Ok(v) => cli_state.mts.push(v),
                    Err(e) => {
                        eprintln!("Error: {e} while parsing {rem_arg} from {arg}");
                        break;
                    }
                }
            }
        } else if let Some(rem_arg) = arg.strip_prefix("--iters=") {
            match rem_arg.parse::<usize>() {
                Ok(v) => cli_state.iter_count = v,
                Err(e) => eprintln!("Error: '{e}' while parsing '{arg}'"),
            }
        } else if let Some(report_type) = arg.strip_prefix("--report=") {
            cli_state.output_mode = match report_type {
                "time" => OutputMode::Time,
                "thrpt" | "throughput" => OutputMode::Throughput,
                "raw" => OutputMode::Raw,
                _ => {
                    eprintln!(
                        "Error: unsupported report arg: `{report_type}`.
Expected one of ['time', 'thrpt', 'throughput', 'raw']"
                    );
                    break;
                }
            };
        } else if let Some(opt) = arg.strip_prefix("--warm-up=") {
            match opt {
                "y" | "yes" | "true" => cli_state.warm_up = true,
                "n" | "no" | "false" => cli_state.warm_up = false,
                _ => {
                    eprintln!(
                        "Error: unsupported value for --warm-up: `{opt}`.
Expected one of ['y', 'yes', 'true', 'n', 'no', 'false']"
                    )
                }
            }
        } else if arg == "--csv" {
            cli_state.output_format = OutputFormat::Csv;
        } else if arg == "--pretty" {
            cli_state.output_format = OutputFormat::Pretty;
        } else if arg == "--just-run" {
            cli_state.just_run = true;
        } else if arg.starts_with("--") {
            eprintln!("Error: unrecognized option: `{arg}`");
        } else {
            // else treat as seq script
            let mut node_arena = new_node_arena();
            match parse(&arg, &mut node_arena, &mut HashSet::new()) {
                Ok("") => {}
                Ok(rest) => {
                    eprintln!("Error: while parsing '{arg}', '{rest}' was left-over, indicating improper formatting");
                    continue 'arg_loop;
                }
                Err(parse_err) => {
                    eprintln!(
                        "Error: while parsing '{arg}', {} in '{}' during a '{}'",
                        parse_err.err_type, parse_err.rest, parse_err.context
                    );
                    continue 'arg_loop;
                }
            };
            if let Err(e) = eval(&node_arena, &mut file_buf) {
                eprintln!("Error: '{e}' while evaluating '{arg}'");
                continue 'arg_loop;
            };
            if cli_state.just_run {
                for (_, f) in cli_state.fns.iter() {
                    for _ in 0..cli_state.iter_count {
                        unsafe { black_box(f(black_box(&file_buf))) };
                    }
                }
            } else {
                run_bench(&arg, &file_buf, &cli_state, &mut times);
            }
        }
    }
}

fn run_bench(arg: &str, bytes: &[u8], cli_options: &CliOptions, times: &mut Vec<Duration>) {
    let benny_output = benny(bytes);
    const KILOBYTE: usize = 1000;
    const MEGABYTE: usize = KILOBYTE * 1000;
    const GIGABYTE: usize = MEGABYTE * 1000;
    let (size_unit, size_scale_factor) = match bytes.len() {
        0..KILOBYTE => ("B", 1.),
        KILOBYTE..MEGABYTE => ("KB", 0.001),
        MEGABYTE..GIGABYTE => ("MB", 0.000_001),
        GIGABYTE.. => ("GB", 0.000_000_001),
    };
    print!(
        "\n> {arg}; {:.3}{size_unit}; ",
        bytes.len() as f64 * size_scale_factor
    );
    if bytes.is_empty() {
        println!();
        return;
    }

    if let Some(idx) = benny_output {
        println!(
            "first window at {:5}%",
            idx as f64 / bytes.len() as f64 * 100.
        );
    } else {
        println!("no windows present");
    }

    match (cli_options.output_format, cli_options.output_mode) {
        (_, OutputMode::Raw) => println!("name,threads,times..."),
        (OutputFormat::Pretty, _) => println!(
            "{: >20}{: >10}{:>13}{:>13}{:>13}{:>13}",
            "name", "threads", "best", "median", "mean", "stddev",
        ),
        (OutputFormat::Csv, _) => println!("name,threads,best,median,mean,stddev"),
    }

    // How many bytes needed to be scanned, used when calculating thrpt.
    // different than how many were actually scanned.
    let search_space_size = benny_output.unwrap_or(bytes.len()) as f64;
    let thrpt_numerator = (search_space_size + 13.) / (GIGABYTE as f64);

    for (name, func) in cli_options.fns.iter().copied() {
        if let Some(unmet_req) = find_unmet_req(name) {
            eprintln!("Warning: fn `{name}` requires feature `{unmet_req:?}`. Skipping.");
            continue;
        }

        for &n_threads in cli_options.mts.iter() {
            let func_output = unsafe { mt(bytes, n_threads, func) };
            if !name.starts_with("load") && func_output != benny_output {
                eprintln!(
                "{name}'s result ({func_output:?}) doesn't match benny's output ({benny_output:?}). ({n_threads} threads)"
            );
                continue;
            }

            times.clear();
            // warm-up - This helps with caches and giving the processor time to leave an idle powerstate
            if cli_options.warm_up {
                for _ in 0..cli_options.iter_count / 2 {
                    black_box(unsafe { mt(black_box(bytes), n_threads, func) });
                }
            }
            times.extend((0..cli_options.iter_count).map(|_| {
                // not designed for inputs with sub-microsecond durations, so the overhead of the
                // timer isn't a problem
                let now = std::time::Instant::now();
                black_box(unsafe { mt(black_box(bytes), n_threads, func) });
                now.elapsed()
            }));

            if cli_options.output_mode == OutputMode::Raw {
                let mut stdout = std::io::stdout().lock();
                write!(&mut stdout, "\"{name}\",{n_threads}").unwrap();
                for time in times.iter() {
                    write!(&mut stdout, ",{}", time.as_nanos()).unwrap();
                }
                writeln!(&mut stdout).unwrap();
                continue;
            }
            times.sort();
            let count: u32 = times.len().try_into().unwrap();
            let best = times[0];
            let median = times[times.len() / 2];
            let mean = times.iter().sum::<Duration>() / count;
            let stddev = Duration::from_secs_f64(
                times
                    .iter()
                    .map(|&v| (v.max(mean) - v.min(mean)).as_secs_f64())
                    .map(|v| v * v)
                    .sum::<f64>()
                    .div(count as f64)
                    .sqrt(),
            );
            let mean_thrpt = thrpt_numerator / mean.as_secs_f64();
            let stddev_thrpt = times
                .iter()
                .map(|&v| thrpt_numerator / v.as_secs_f64())
                .map(|thrpt| thrpt - mean_thrpt)
                .map(|v| v * v)
                .sum::<f64>()
                .div(count as f64)
                .sqrt();

            match (cli_options.output_format, cli_options.output_mode) {
                (_, OutputMode::Raw) => unreachable!("already handled OutputMode::Raw"),
                (OutputFormat::Pretty, OutputMode::Time) => println!(
                    "{name: >20}{n_threads: >10}{best:>13.4?}{median:>13.4?}{mean:>13.4?}{stddev:>13.4?}",
                ),
                (OutputFormat::Pretty, OutputMode::Throughput) => println!(
                    "{name: >20}{n_threads: >10}{best:>13.4}{median:>13.4}{mean:>13.4}{stddev:>13.4}",
                    best = thrpt_numerator / best.as_secs_f64(),
                    median = thrpt_numerator / median.as_secs_f64(),
                    mean = mean_thrpt,
                    stddev = stddev_thrpt,
                ),
                (OutputFormat::Csv, OutputMode::Time) => println!(
                    "\"{name}\",{n_threads},{best},{median},{mean},{stddev}",
                    best = best.as_nanos(),
                    median = median.as_nanos(),
                    mean = mean.as_nanos(),
                    stddev = stddev.as_nanos(),
                ),
                (OutputFormat::Csv, OutputMode::Throughput) => println!(
                    "\"{name}\",{n_threads},{best:.4},{median:.4},{mean:.4},{stddev:.4}",
                    best = thrpt_numerator / best.as_secs_f64(),
                    median = thrpt_numerator / median.as_secs_f64(),
                    mean = mean_thrpt,
                    stddev = stddev_thrpt,
                ),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! Very basic tests, don't rely on these to catch all your bugs
    use rand::SeedableRng;

    use super::input_gen::gen_rand;
    use super::*;

    const PRESENT_INPUTS: &[(&str, usize)] = &[
        ("qwertyuiopasdfghjklo", 0),
        ("qqwertyuiopasdfghjklo", 1),
        ("wqwertyuiopasdfghjklo", 1),
        ("eqwertyuiopasdfghjklo", 1),
        ("rqwertyuiopasdfghjklo", 1),
        ("qqqqqwertyuiopasdfghjklo", 4),
        ("wqqqqwertyuiopasdfghjklo", 4),
        ("awqqqwertyuiopasdfghjklo", 4),
        ("abcqqwertyuiopasdfghjklo", 4),
    ];

    const ABSENT_INPUTS: &[&str] = &[
        "",
        "q",
        "qwert",
        "qwertyuiopasd", // 13
        "qqqqqqqqqqqqqqqq",
        "qwertyuiopqwerthj",
        "qwertyuiopyuiophj",
        "wwqqwertyuiopasd",
        "wwqwqertyuiopasd",
        "wwqweqrtyuiopasd",
    ];

    /// Iterator over id-function pairs supported by this host cpu
    fn supported_fns() -> impl Iterator<Item = (&'static str, unsafe fn(&[u8]) -> Option<usize>)> {
        ID_TO_FN
            .iter()
            .copied()
            .filter(|&(name, _)| !name.starts_with("load") && find_unmet_req(name).is_none())
    }

    #[test]
    fn test_present() {
        for (fn_name, f) in supported_fns() {
            for &(input, pos) in PRESENT_INPUTS {
                let b_input = input.as_bytes();
                assert_eq!(Some(pos), unsafe { f(b_input) }, "{fn_name} on {input}");
            }
        }
    }

    #[test]
    fn test_absent() {
        for (fn_name, f) in supported_fns() {
            for &input in ABSENT_INPUTS {
                let b_input = input.as_bytes();
                assert_eq!(None, unsafe { f(b_input) }, "{fn_name} on {input}");
            }
        }
    }

    #[test]
    fn test_big_present() {
        let mut out = Vec::<u8>::new();
        for seed in [0, 7654, 241, 4531, 6462, 7357344] {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
            gen_rand(&mut out, 1024 * 128, &mut rng);
            // end with unique bytes
            out.extend_from_slice(b"abcdefghijklmnopqrstuvwxyz");
            let reference_answer = benny(&out);
            for (fn_name, f) in supported_fns() {
                assert_eq!(
                    reference_answer,
                    unsafe { f(&out) },
                    "{fn_name} (right) fail w/ seed {seed}"
                );
            }
            out.clear();
        }
    }

    #[test]
    fn test_big_absent() {
        let mut out = Vec::<u8>::new();
        for seed in [0, 234, 7654, 134132, 2352, 16426] {
            let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
            gen_rand(&mut out, 1024 * 128, &mut rng);
            let reference_answer = benny(&out);
            for (fn_name, f) in supported_fns() {
                assert_eq!(
                    reference_answer,
                    unsafe { f(&out) },
                    "{fn_name} (right) fail w/ seed {seed}"
                );
            }
            out.clear();
        }
    }

    #[test]
    fn test_many_present() {
        // even if many solutions are present, the very first window should always be reported.
        let mut out = vec![b'a'; 8192];
        for window_count in [1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 32, 63, 64, 65] {
            for dist in [1, 2, 4, 13, 16, 20, 2305] {
                for seed in [0, 254, 7434, 23421, 235232] {
                    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
                    for _ in 0..window_count {
                        gen_rand(&mut out, dist, &mut rng);
                        out.extend_from_slice(b"qwertyuiopasdf"); // 14 unique characters
                    }
                    for offset in [0, 123, 4096, 8192] {
                        let reference_answer = benny(&out[offset..]);
                        for (fn_name, f) in supported_fns() {
                            assert_eq!(
                                reference_answer,
                                unsafe { f(&out[offset..]) },
                                "{fn_name} (right) fail w/ seed {seed} and count {window_count}"
                            );
                        }
                    }
                    out.truncate(8192);
                }
            }
        }
    }
}
