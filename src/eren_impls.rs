use crate::benny;

/// Parallelize search_fn on data.
/// Does not spin up a thread if only 1 thread is needed.
/// SAFETY: Same as `search_fn`
pub unsafe fn mt(
    data: &[u8],
    cpus: usize,
    search_fn: unsafe fn(&[u8]) -> Option<usize>,
) -> Option<usize> {
    if data.len() < 14 {
        return None;
    }
    // To an extent, this also depends on search_fn, but we'll leave that to the caller.
    // This is a decent value to mitigate the perf footgun of spinning up 16 threads for 20 bytes
    // of input (or even 2 threads for 16KB of data).
    const MIN_LEN_PER_THREAD: usize = 1024 * 128;
    let max_num_threads = data.len() / MIN_LEN_PER_THREAD;
    let cpus = cpus.min(max_num_threads);

    // just do it on this thread!
    if cpus <= 1 {
        return unsafe { search_fn(data) };
    }

    // could do this with atomics, but this is an easier pattern for me and the inefficiency is
    // negligible (we lose more on not pooling threads).
    let mut results: Vec<Option<usize>> = vec![None; cpus];
    // <3 scoped threads. just don't ask about all the clean-up code it spits out...
    std::thread::scope(|s| {
        let chunk_size = data.len() / results.len();
        for (idx, out) in results.iter_mut().enumerate() {
            let slice_start = idx * chunk_size;
            let slice_end = if idx == cpus - 1 {
                data.len()
            } else {
                (slice_start + chunk_size + 14).min(data.len())
            };
            let slice = &data[slice_start..slice_end];
            s.spawn(move || *out = unsafe { search_fn(slice).map(|i| i + slice_start) });
        }
    });
    results.into_iter().flatten().next()
}

/// Benny's with a handrolled substitute for popcnt.
/// Faster when there isn't popcnt, slower when there is :)
#[inline(never)]
pub fn benny_alt(input: &[u8]) -> Option<usize> {
    if input.len() < 14 {
        return None;
    }
    let mut filter = 0u32;
    input
        .iter()
        .take(14 - 1)
        .for_each(|c| filter ^= 1 << (c % 32));

    let mut pseudo_count = filter.count_ones();

    input.windows(14).position(|w| {
        let first = w[0];
        let last = w[w.len() - 1];

        let last_1hot = 1 << (last & 31);
        let last_is_pseudo_new = (filter & last_1hot) == 0;
        let res = (pseudo_count == 13) & last_is_pseudo_new;
        filter ^= last_1hot;
        let first_1hot = 1 << (first & 31);
        let first_is_pseudo_old = (filter & first_1hot) != 0;
        filter ^= first_1hot;
        pseudo_count =
            pseudo_count + (last_is_pseudo_new as u32) * 2 - (first_is_pseudo_old as u32) * 2;
        res
    })
}

/// Like Benny's but split the input in half and interleave the execution for better ILP.
/// Also tried this with the no_popcnt variant, it was worse.
/// Also tried with 3x, it was worse. (at best, same speed on large inputs but much larger code
/// size)
#[inline(never)]
pub unsafe fn bbeennnnyy(input: &[u8]) -> Option<usize> {
    if input.len() < 64 {
        return benny(input);
    }

    let l_input = &input[..(input.len() / 2 + 13).min(input.len())];
    let r_input_start_offset = input.len() / 2;
    let r_input = &input[r_input_start_offset..];

    let mut l_filter = l_input
        .iter()
        .take(14 - 1)
        .fold(0u32, |mask, byte| mask ^ 1 << (byte & 31));
    let mut r_filter = r_input
        .iter()
        .take(14 - 1)
        .fold(0u32, |mask, byte| mask ^ 1 << (byte & 31));
    let mut rem_num_iters = l_input.len().min(r_input.len()) - 13;
    let mut offset = 0;

    // while all have remaining, do one iter of each
    while rem_num_iters != 0 {
        let l_first = *unsafe { l_input.get_unchecked(offset) };
        let l_last = *unsafe { l_input.get_unchecked(offset + 13) };
        let r_first = *unsafe { r_input.get_unchecked(offset) };
        let r_last = *unsafe { r_input.get_unchecked(offset + 13) };

        l_filter ^= 1 << (l_last & 31);
        r_filter ^= 1 << (r_last & 31);
        let l_res = l_filter.count_ones() == 14;
        let r_res = r_filter.count_ones() == 14;
        l_filter ^= 1 << (l_first & 31);
        r_filter ^= 1 << (r_first & 31);

        if l_res | r_res {
            break;
        }
        rem_num_iters -= 1;
        offset += 1;
    }
    unsafe {
        benny(l_input.get_unchecked(offset..))
            .map(|v| v + offset)
            .or_else(|| {
                benny(r_input.get_unchecked(offset..)).map(|v| v + offset + r_input_start_offset)
            })
    }
}
