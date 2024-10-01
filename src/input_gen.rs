//! A DSL for easily generating interesting inputs.
//! A good chunk of this code would be obsolete with a REPL

use crate::benny;
use rand::{distributions::uniform, prelude::*};
use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
    ops::Range,
};

type RNG = rand_chacha::ChaCha8Rng;

pub const USAGE_MESSAGE: &str = "
# Input Gen:
A DSL for producing sequences.
Should be fairly efficient, memoizes sequences when possible.

## Examples 
* 'file(some/file.txt)'
  Load file contents from ./some/file.txt
* 'concat(rng(x, 1234), sparserandom(10M, x), literal(abcdefghijklmn))'
  Create an rng named x with seed 1234, then use x to produce 10 million bytes that 
  don't contain a window of 14 unique characters, then end with 14 unique characters.
  This creates an input where the answer is at the very end.
* 'concat(rep(300K, file(body.txt)), file(tail.txt))'
  Contents of body.txt repeated 300 thousand times, followed by the contents of tail.txt
  
## Seqs:
* literal(a..=z*), alias lit
* file(filepath)
  !! does not escape parens, `filepath` should not contain parens
  Loads the contents of `filepath`. Trims leading and trailing whitespace.
* concat(seq, seq...)
* repeat(n, seq), alias rep
  Repeat `seq` `n` times.
* copy(n, seq)
  Like `repeat` but `seq` is only evaluated once and that first result is copied `n` times.
  Does not evaluate `seq` if `n` is zero.
* rng(label, seed)
  Create an RNG. No bytes are emitted. Overwrites any previous RNG with the same label.
  `label` must 
* denserandom(dist, count, rng_label), alias drand
  Produces a sequence with `count` many windows of 14 unique bytes separated by 
  `dist` bytes not containing a window.
  Essentially a shorthand for `rep(count, concat(srand(dist, rng_label), lit(qwertyuiopasdf)))`
* sparserandom(len, rng_label), alias srand

## Numbers:
* constant: integer literal in [0, int::MAX], e.g. '234'
  Literals can be suffixed with K, M or G for 1000x, etc;
  likewise with Ki, Mi or Gi for 1024x, etc.
* normal(mean, stddev, rng_label)
* uniform(min, max, rng_label)
";

/// Identifier for an RNG instance
type RngId<'a> = &'a str;
/// Like a reference, but via index into an array
/// 0 shall be treated as a special empty sequence
type GenNodeId = usize;
/// Container for the GenNodes produced by parsing a command string
/// The 0th element is a 'null' element.
/// In a larger project, this would be a newtype wrapper for Vec
type GenNodeArena<'command> = Vec<GenNode<'command>>;
type GenNodeArenaSlice<'command> = [GenNode<'command>];

/// The nodeId of the first node
pub const NODE_ARENA_START_IDX: GenNodeId = 1;

/// Returns an Arena ready to be appended to.
/// First entry is a 'null' node
pub fn new_node_arena<'input>() -> GenNodeArena<'input> {
    vec![GenNode::Literal("")]
}

/// Nodes in the DSL for generating sequences.
/// These are stored in a flat AST (nodes reference each other by indexes into an arena)
#[derive(Debug, Clone, Copy)]
pub enum GenNode<'a> {
    /// Just returns the bytes it has
    Literal(&'a str),
    /// Associate this ID to a fresh RNG with this seed
    CreateRng(RngId<'a>, u64),
    /// Random bytes with `count` windows of 14 unique bytes, appearing `dist` bytes apart
    DenseRandom {
        dist: NumberNode<'a>,
        count: NumberNode<'a>,
        rng: RngId<'a>,
    },
    /// `len` many random bytes with no windows
    SparseRandom { len: NumberNode<'a>, rng: RngId<'a> },
    /// Multiple sequences
    Concat { first: GenNodeId, next: GenNodeId },
    /// eval `seq` once (if at all) and memcpy result `count` times
    /// More efficient than Repeat but less flexible.
    Copy {
        count: NumberNode<'a>,
        seq: GenNodeId,
    },
    /// eval `seq` `count` times.
    /// Will cache (re-use) unchanging portions, but re-evaluate the variable portions
    Repeat {
        count: NumberNode<'a>,
        seq: GenNodeId,
    },
    /// Load file contents, trim whitespace
    File(&'a str),
}

#[derive(Debug, Clone, Copy)]
pub enum NumberNode<'a> {
    Const(u64),
    Normal {
        mean: u64,
        stddev: u64,
        rng: RngId<'a>,
    },
    Uniform {
        min: u64,
        max: u64,
        rng: RngId<'a>,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct ParseErr<'a> {
    pub context: &'a str,
    pub err_type: ParseErrType,
    pub rest: &'a str,
}

impl<'a> ParseErr<'a> {
    fn new(context: &'a str, err_type: ParseErrType, rest: &'a str) -> Self {
        Self {
            context,
            err_type,
            rest,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ParseErrType {
    BadCommandName,
    BadLiteral,
    BadNumber,
    MissingComma,
    MissingOpenParen,
    MissingClosingParen,
    UndeclaredRngId,
    InvalidRngId,
}
impl Display for ParseErrType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

/// Parses next seq, recursing as necessary.
/// Returns the remaining string (everything following the matching closing paren for the first
/// open paren) on success
/// Returns error type and approximate source if there is a syntax error.
///
/// example input: "concat(rng(x, 3456), repeat(16, rand(1M, x)), literal(abcdefghijklmno))"
/// See `[USAGE_MESSAGE]` for a description of the accepted input.
pub fn parse<'input, 'node_arena>(
    input: &'input str,
    node_arena: &'node_arena mut GenNodeArena<'input>,
    rng_label_set: &mut HashSet<RngId<'input>>,
) -> Result<&'input str, ParseErr<'input>> {
    type ParseResult<'input, T> = Result<T, (ParseErrType, &'input str)>;
    fn strip_closing_paren(s: &str) -> Result<&str, (ParseErrType, &str)> {
        match s.trim_ascii_start().strip_prefix(')') {
            Some(s) => Ok(s),
            None => Err((ParseErrType::MissingClosingParen, s)),
        }
    }

    fn strip_comma(s: &str) -> Result<&str, (ParseErrType, &str)> {
        match s.trim_ascii_start().strip_prefix(',') {
            Some(s) => Ok(s.trim_ascii_start()),
            None => Err((ParseErrType::MissingComma, s)),
        }
    }

    fn split_comma(s: &str) -> ParseResult<(&str, &str)> {
        match s.split_once(',') {
            Some(v) => Ok(v),
            None => Err((ParseErrType::MissingComma, s)),
        }
    }

    fn parse_int(s: &str) -> Result<u64, (ParseErrType, &str)> {
        let err = (ParseErrType::BadNumber, s);
        let (mut s, is_binary_suffix) = match s.strip_suffix('i') {
            None => (s, false),
            Some(s) => (s, true),
        };
        let multiplier = match s.as_bytes().last().map(u8::to_ascii_uppercase) {
            None => return Err(err),
            Some(b'G') if is_binary_suffix => 1024 * 1024 * 1024,
            Some(b'G') => 1_000_000_000,
            Some(b'M') if is_binary_suffix => 1024 * 1024,
            Some(b'M') => 1_000_000,
            Some(b'K') if is_binary_suffix => 1024,
            Some(b'K') => 1_000,
            _ if is_binary_suffix => return Err(err),
            _ => 1,
        };
        if multiplier != 1 {
            s = &s[..s.len() - 1]
        };
        s.parse::<u64>()
            .ok()
            .and_then(|i| i.checked_mul(multiplier))
            .ok_or(err)
    }

    fn parse_ending_rng_label<'input>(
        input: &'input str,
        rng_label_set: &HashSet<&str>,
    ) -> Result<(RngId<'input>, &'input str), (ParseErrType, &'input str)> {
        let (rng, rest) = input
            .split_once(')')
            .ok_or((ParseErrType::MissingClosingParen, input))?;
        let rng = rng.trim_ascii_end();

        if rng.contains(" ") {
            return Err((ParseErrType::InvalidRngId, rng));
        }
        if !rng_label_set.contains(rng) {
            return Err((ParseErrType::UndeclaredRngId, rng));
        }
        Ok((rng, rest))
    }

    fn parse_number_node<'input>(
        input: &'input str,
        rng_label_set: &HashSet<RngId>,
    ) -> Result<(NumberNode<'input>, &'input str), (ParseErrType, &'input str)> {
        // could be constant (just a number) or "normal(..)" or "uniform(..)"
        let rest = input.trim_ascii_start();
        let (ctor, rest) = if let Some(rest) = rest.strip_prefix("uniform") {
            // (min, max, rng)
            (
                (|min, max, rng| NumberNode::Uniform { min, max, rng })
                    as fn(u64, u64, &str) -> NumberNode,
                rest,
            )
        } else if let Some(rest) = rest.strip_prefix("normal") {
            // (mean, stddev, rng)
            (
                (|mean, stddev, rng| NumberNode::Normal { mean, stddev, rng })
                    as fn(u64, u64, &str) -> NumberNode,
                rest,
            )
        } else {
            let (number, rest) = rest.split_at(
                rest.bytes()
                    .take_while(|b| b.is_ascii_alphanumeric())
                    .count(),
            );
            return Ok((NumberNode::Const(parse_int(number)?), rest));
        };
        let rest = rest
            .trim_ascii_start()
            .strip_prefix('(')
            .ok_or((ParseErrType::MissingOpenParen, rest))?
            .trim_ascii_start();
        let (num1, rest) = split_comma(rest)?;
        let num1 = parse_int(num1.trim_ascii())?;
        let (num2, rest) = split_comma(rest)?;
        let num2 = parse_int(num2.trim_ascii())?;
        let (rng, rest) = parse_ending_rng_label(rest.trim_ascii_start(), rng_label_set)?;
        Ok((ctor(num1, num2, rng), rest))
    }

    let input = input.trim_ascii();
    let Some((command, rest)) = input.split_once('(') else {
        return Err(ParseErr::new("", ParseErrType::MissingOpenParen, input));
    };
    let (command, rest) = (command.trim_ascii_end(), rest.trim_ascii_start());
    let wrap_err =
        |(err_type, rest): (ParseErrType, &'input str)| ParseErr::new(command, err_type, rest);
    match command {
        "lit" | "literal" => {
            // abcd)...
            let Some((literal, rest)) = rest.split_once(')') else {
                return Err(wrap_err((ParseErrType::MissingClosingParen, rest)));
            };
            let literal = literal.trim_ascii_end();
            if literal.bytes().all(|b| b.is_ascii_lowercase()) {
                node_arena.push(GenNode::Literal(literal));
                Ok(rest.trim_ascii_start())
            } else {
                Err(wrap_err((ParseErrType::BadLiteral, literal)))
            }
        }
        "rng" => {
            // label, seed
            let (label, rest) = split_comma(rest).map_err(wrap_err)?;
            let label = label.trim_ascii_end();
            if label.contains(" ") {
                return Err(wrap_err((ParseErrType::InvalidRngId, label)));
            }
            let (seed, rest) = rest
                .trim_ascii_start()
                .split_once(')')
                .ok_or(wrap_err((ParseErrType::MissingClosingParen, rest)))?;
            let seed = parse_int(seed.trim_ascii_end()).map_err(wrap_err)?;
            rng_label_set.insert(label);
            node_arena.push(GenNode::CreateRng(label, seed));
            Ok(rest)
        }
        "rep" | "repeat" | "copy" => {
            // n, seq)...
            let (num_reps, rest) =
                parse_number_node(rest.trim_ascii_end(), rng_label_set).map_err(wrap_err)?;
            node_arena.push(if command == "copy" {
                GenNode::Copy {
                    count: num_reps,
                    seq: node_arena.len() + 1,
                }
            } else {
                GenNode::Repeat {
                    count: num_reps,
                    seq: node_arena.len() + 1,
                }
            });
            let rest = strip_comma(rest.trim_ascii_start())
                .map_err(wrap_err)?
                .trim_ascii_start();
            let rest = parse(rest, node_arena, rng_label_set)?;
            let rest = strip_closing_paren(rest)
                .map_err(wrap_err)?
                .trim_ascii_start();
            Ok(rest)
        }
        "concat" => {
            // seq, seq...)...
            let mut rest = rest;
            loop {
                rest = rest.trim_ascii_start();
                if let Some(r) = rest.strip_prefix(')') {
                    rest = r;
                    break;
                }

                let this_node_idx = node_arena.len();
                node_arena.push(GenNode::Concat {
                    first: node_arena.len() + 1,
                    next: 0,
                });

                rest = parse(rest, node_arena, rng_label_set)?.trim_ascii_start();

                let next_node_idx = node_arena.len();
                // go back and fix 'next' idx of the concat node we pushed
                match &mut node_arena[this_node_idx] {
                    GenNode::Concat { first: _, next } => {
                        *next = next_node_idx;
                    }
                    _ => unreachable!(),
                };

                if rest.starts_with(')') {
                    continue;
                } else if let Some(r) = rest.strip_prefix(',') {
                    rest = r;
                } else {
                    return Err(wrap_err((ParseErrType::MissingComma, rest)));
                }
            }
            // end with 'empty' node. Similar to Null in Lisp
            node_arena.push(GenNode::Literal(""));
            Ok(rest)
        }
        "srand" | "srandom" | "sparserandom" => {
            // len, rng)...
            let (len, rest) = parse_number_node(rest, rng_label_set).map_err(wrap_err)?;
            let rest = strip_comma(rest).map_err(wrap_err)?;
            let (rng, rest) = parse_ending_rng_label(rest, rng_label_set).map_err(wrap_err)?;
            node_arena.push(GenNode::SparseRandom { len, rng });
            Ok(rest.trim_ascii_start())
        }
        "drand" | "drandom" | "denserandom" => {
            // dist, count, rng)...
            let (dist, rest) = parse_number_node(rest, rng_label_set).map_err(wrap_err)?;
            let rest = strip_comma(rest).map_err(wrap_err)?;
            let (count, rest) = parse_number_node(rest, rng_label_set).map_err(wrap_err)?;
            let rest = strip_comma(rest).map_err(wrap_err)?;
            let (rng, rest) = parse_ending_rng_label(rest, rng_label_set).map_err(wrap_err)?;
            node_arena.push(GenNode::DenseRandom { dist, count, rng });
            Ok(rest.trim_ascii_start())
        }
        "file" => {
            // path)...
            let (path, rest) = rest
                .split_once(')')
                .ok_or((ParseErrType::MissingClosingParen, rest))
                .map_err(wrap_err)?;
            node_arena.push(GenNode::File(path));
            Ok(rest.trim_ascii_start())
        }
        c => Err(wrap_err((ParseErrType::BadCommandName, c))),
    }
}

/// Errors that occur while evaluating a GenNode
pub enum GenError {
    IOError(std::io::Error),
    InvalidFileContents,
}
impl Display for GenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GenError::IOError(e) => write!(f, "{e}"),
            GenError::InvalidFileContents => {
                write!(f, "invalid file contents - all characters must be a-z",)
            }
        }
    }
}
impl From<std::io::Error> for GenError {
    fn from(value: std::io::Error) -> Self {
        Self::IOError(value)
    }
}

pub fn eval<'input, 'out_buf>(
    node_arena: &GenNodeArenaSlice<'input>,
    out: &'out_buf mut Vec<u8>,
) -> Result<(), GenError> {
    fn eval_number(number_node: NumberNode, rng_table: &mut RngTable) -> u64 {
        match number_node {
            NumberNode::Const(v) => v,
            NumberNode::Normal { mean, stddev, rng } => rng_table
                .get_mut(rng)
                .unwrap()
                .sample(rand_distr::Normal::new(mean as f64, stddev as f64).unwrap())
                as u64,
            NumberNode::Uniform { min, max, rng } => {
                rng_table.get_mut(rng).unwrap().gen_range(min..max)
            }
        }
    }

    fn repeat_within(out: &mut Vec<u8>, range: Range<usize>, num_reps: usize) {
        if num_reps == 0 {
            return;
        }
        let initial_len = out.len();
        let seq_len = range.len();
        out.reserve(seq_len * num_reps);
        out.extend_from_within(range.clone());
        let mut reps_in_slice = 1;
        // double the slice until we can't double anymore.
        // This way we aren't unreasonably slow on rep(1G, lit(a)).
        while reps_in_slice * 2 < num_reps {
            out.extend_from_within(initial_len..);
            reps_in_slice <<= 1;
        }
        let num_reps_left = num_reps - reps_in_slice;
        out.extend_from_within(initial_len..initial_len + num_reps_left * seq_len);
        assert_eq!(out.len(), initial_len + num_reps * seq_len);
    }
    type RngTable<'input> = HashMap<RngId<'input>, RNG>;

    fn recurse<'input, 'out_buf>(
        current_idx: GenNodeId,
        node_arena: &GenNodeArenaSlice<'input>,
        rng_table: &mut RngTable<'input>,
        memo: &mut HashMap<GenNodeId, Range<usize>>,
        out: &'out_buf mut Vec<u8>,
    ) -> Result<(), GenError> {
        if let Some(range) = memo.get(&current_idx) {
            out.extend_from_within(range.clone());
            return Ok(());
        }
        let initial_len = out.len();
        match node_arena[current_idx] {
            GenNode::Literal(slice) => {
                out.extend_from_slice(slice.as_bytes());
                memo.insert(current_idx, initial_len..out.len());
            }
            GenNode::CreateRng(label, seed) => {
                rng_table.insert(label, RNG::seed_from_u64(seed));
            }
            GenNode::DenseRandom { dist, count, rng } => {
                for _ in 0..eval_number(count, rng_table) {
                    gen_rand(
                        out,
                        eval_number(dist, rng_table).try_into().unwrap(),
                        rng_table.get_mut(rng).unwrap(),
                    );
                    out.extend_from_slice(b"qwertyuiopasdf"); // 14 unique characters
                }
            }
            GenNode::SparseRandom { len, rng } => gen_rand(
                out,
                eval_number(len, rng_table).try_into().unwrap(),
                rng_table.get_mut(rng).unwrap(),
            ),
            GenNode::Concat { first, next } => {
                recurse(first, node_arena, rng_table, memo, out)?;
                recurse(next, node_arena, rng_table, memo, out)?;
                if memo.contains_key(&first) && memo.contains_key(&next) {
                    memo.insert(current_idx, initial_len..out.len());
                }
            }
            GenNode::Copy { count, seq } => {
                let mut rem_iters = eval_number(count, rng_table);
                if rem_iters == 0 {
                    return Ok(());
                }
                recurse(seq, node_arena, rng_table, memo, out)?;
                rem_iters -= 1;
                let range = initial_len..out.len();
                memo.insert(seq, range.clone());
                repeat_within(out, range, rem_iters.try_into().unwrap());
                if matches!(count, NumberNode::Const(_)) {
                    memo.insert(current_idx, initial_len..out.len());
                }
            }
            GenNode::Repeat { count, seq } => {
                let mut rem_iters = eval_number(count, rng_table);
                if rem_iters == 0 {
                    return Ok(());
                }
                recurse(seq, node_arena, rng_table, memo, out)?;
                rem_iters -= 1;

                if let Some(range) = memo.get(&seq) {
                    repeat_within(out, range.clone(), rem_iters.try_into().unwrap());
                    if matches!(count, NumberNode::Const(_)) {
                        memo.insert(current_idx, initial_len..out.len());
                    }
                } else {
                    for _ in 0..rem_iters {
                        recurse(seq, node_arena, rng_table, memo, out)?;
                    }
                }
            }
            GenNode::File(filepath) => {
                let text = std::fs::read_to_string(filepath)?;
                let text = text.trim();
                if !text.bytes().all(|b| b.is_ascii_lowercase()) {
                    return Err(GenError::InvalidFileContents);
                }
                out.extend_from_slice(text.as_bytes());
                memo.insert(current_idx, initial_len..out.len());
            }
        };
        Ok(())
    }

    let mut rng_table = RngTable::new();

    // nodes that can be memoized, are
    let mut memo = HashMap::<GenNodeId, Range<usize>>::new();
    recurse(
        NODE_ARENA_START_IDX,
        node_arena,
        &mut rng_table,
        &mut memo,
        out,
    )
}

/// Appends `len` random bytes to `out`
/// The first and last byte will always be 'a', for the sake of composability.
pub fn gen_rand(out: &mut Vec<u8>, len: usize, rng: &mut RNG) {
    const WINDOW_LEN: usize = 14;

    if len <= 2 {
        for _ in 0..len {
            out.push(b'a');
        }
        return;
    }

    let initial_len = out.len();
    out.reserve(len);

    let mut rand_chars = rng.sample_iter(uniform::Uniform::new_inclusive(b'a', b'z'));

    // first char is 'a'
    let mut counts = [0u8; 26];
    counts[0] = 1;
    let mut initial = [0u8; WINDOW_LEN];
    initial[0] = b'a';
    for i in 1..WINDOW_LEN {
        let byte = rand_chars.next().unwrap();
        initial[i] = byte;
        counts[(byte - b'a') as usize] += 1;
    }
    let mut last_chars = initial;

    // we populated the 14th byte, but don't keep it because it may not satisfy the need for a
    // duplicate
    out.extend_from_slice(&initial[..len.min(WINDOW_LEN - 1)]);

    let mut idx = 13;
    out.extend(
        rand_chars
            .map(|b| {
                // remove previous char at idx
                let old_char = last_chars[idx];
                counts[(old_char - b'a') as usize] -= 1;
                last_chars[idx] = 0;

                let new_char =
                    if counts[b as usize - b'a' as usize] != 0 || counts.iter().any(|c| *c >= 2) {
                        // already have a duplicate, any char is fine
                        b
                    } else {
                        // don't pick next-oldest char because that's a vicious cycle
                        last_chars[(idx + WINDOW_LEN - 1) % 14]
                    };
                counts[(new_char - b'a') as usize] += 1;
                last_chars[idx] = new_char;
                idx = (idx + 1) % 14;
                new_char
            })
            .take(len.saturating_sub(initial.len() - 1).saturating_sub(2)),
    );
    out.truncate(initial_len + len - 2);
    // Goal: make this end in 'a' for easy composition.
    // stop 2 early, copy last so that we're safe w.r.t. no unique window.
    // then end with 'a'
    out.push(out[out.len() - 1]);
    out.push(b'a');

    assert_eq!(out.len(), initial_len + len);
    assert_eq!(out[initial_len], b'a');
    assert_eq!(out[out.len() - 1], b'a');

    if len > 13 + 2 {
        assert_eq!(
            None,
            benny(&out[initial_len..]),
            "appended output should not contain a window. rng: {rng:?} len {len} outlen: {}",
            out.len()
        );
    }
}
