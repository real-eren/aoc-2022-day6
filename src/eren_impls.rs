use crate::benny;
use core::panic;
use std::arch::x86_64::{
    __m128i, _mm256_castsi256_ps, _mm256_movemask_ps, _mm512_conflict_epi32, _mm512_cvtepu8_epi32,
    _mm512_loadu_si512, _mm512_mask2int, _mm512_test_epi32_mask, _mm_loadu_si128, _mm_set_epi8,
    _mm_shuffle_epi8,
};

/// Returns a **reverse-ordered** mask of lanes which matched a later lane
/// if 14th bit is set, the first byte of input matched some later byte
/// You can use `13 - mask.trailing_zeros()` to get the index of the highest byte that matched a
/// later byte
///
/// PERFORMANCE: expects target_features AVX512F, AVX512CD
#[inline(always)]
unsafe fn avx512_window_to_mask(bytes_16: __m128i, reverse_shuffle_first14: __m128i) -> u16 {
    // reverse lanes [0:13], zero out 14 and 15 (last two)
    let rev_bytes_16 = _mm_shuffle_epi8(bytes_16, reverse_shuffle_first14);
    // widen u8*16 to u32*16
    let rev_u32_16 = _mm512_cvtepu8_epi32(rev_bytes_16);
    // compare each lane against *previous* lanes, producing a bitmask of matches for each lane
    // (this is why we had to reverse. so that low bytes end up in high lanes and
    // would be compared with high bytes)
    let rev_bitmask32_16 = _mm512_conflict_epi32(rev_u32_16);
    // get a mask of which lanes are nonzero
    let rev_nonzeros_mask16 =
        _mm512_mask2int(_mm512_test_epi32_mask(rev_bitmask32_16, rev_bitmask32_16));
    rev_nonzeros_mask16 as u16 & 0x3FFF
}

/// SAFETY: uses AVX512F, AVX512CD, and BMI1
#[target_feature(enable = "avx512f,avx512cd,bmi1")]
#[no_mangle]
#[inline]
pub unsafe fn conflict(input: &[u8]) -> Option<usize> {
    if input.len() < 14 {
        return None;
    }
    // this can be made const, but perf-wise it's already outside the loop
    // compiler generates a constant for this
    let reverse_shuffle_first14: __m128i =
        _mm_set_epi8(-1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13);
    let mut idx = 0;
    while let Some(slice) = input.get(idx..(idx + 16)) {
        let rev_dup_lanes_mask = avx512_window_to_mask(
            _mm_loadu_si128(slice.as_ptr().cast()),
            reverse_shuffle_first14,
        );

        if rev_dup_lanes_mask == 0 {
            return Some(idx);
        } else {
            // because we reversed,
            // low bit is last byte/lane
            // high bit is first byte/lane
            // want to find highest non-zero lane -> lowest set bit
            // so count trailing zeros
            // (note: we know at least one set bit bc nonzero)
            let trailing_zeros = rev_dup_lanes_mask.trailing_zeros() as usize;
            // move one past that lane
            idx += 1 + 13 - trailing_zeros;
        }
    }
    // very lazy way of handling the last 1-2 iters
    input.get(idx..).and_then(benny).map(|i| i + idx)
}

/// Stamps out a copy of avx512_ht_n
macro_rules! def_conflict_mc_n {
    ($name: ident, $n: literal) => {
        #[target_feature(enable = "avx512f,avx512cd,bmi1")]
        #[no_mangle]
        pub unsafe fn $name(input: &[u8]) -> Option<usize> {
            conflict_mc_n::<$n>(input)
        }
    };
}

// I instantiate these here so that they're easier to identify in the asm (otherwise we'd only have mangled names)
def_conflict_mc_n!(conflict_mc1b, 1);
def_conflict_mc_n!(conflict_mc2b, 2);
def_conflict_mc_n!(conflict_mc3b, 3);
def_conflict_mc_n!(conflict_mc4b, 4);
def_conflict_mc_n!(conflict_mc5b, 5);
def_conflict_mc_n!(conflict_mc6b, 6);
def_conflict_mc_n!(conflict_mc7b, 7);
def_conflict_mc_n!(conflict_mc8b, 8);
def_conflict_mc_n!(conflict_mc9b, 9);
def_conflict_mc_n!(conflict_mc10b, 10);
def_conflict_mc_n!(conflict_mc11b, 11);
def_conflict_mc_n!(conflict_mc12b, 12);

/// Basically, the algorithm functions like this:
/// Instead of having one cursor into the input that gets advanced left-to-right,
/// there are `NUM_CURSORS` into the input, each getting advanced each iteration.
///
/// SAFETY & PERF: expects target_features AVX512F, AVX512CD, and BMI1
#[inline(always)]
pub unsafe fn conflict_mc_n<const NUM_CURSORS: usize>(input: &[u8]) -> Option<usize> {
    const MIN_SPLIT_LEN: usize = 128;
    if const { NUM_CURSORS == 0 } {
        panic!("need at least one cursor!")
    }

    if input.len() <= MIN_SPLIT_LEN {
        return conflict(input);
    }

    let reverse_shuffle_first14: __m128i =
        _mm_set_epi8(-1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13);

    let inputs: [&[u8]; NUM_CURSORS] = {
        let mut a = [input; NUM_CURSORS];
        for (i, e) in a.iter_mut().enumerate() {
            *e = &input[..((i + 1) * input.len() / NUM_CURSORS + 13).min(input.len())];
        }
        a[NUM_CURSORS - 1] = input;
        a
    };
    let mut indices: [usize; NUM_CURSORS] = {
        let mut a = [0usize; NUM_CURSORS];
        for (i, e) in a.iter_mut().enumerate() {
            *e = i * input.len() / NUM_CURSORS;
        }
        a
    };

    'outer: loop {
        let mut slices = [input as &[u8]; NUM_CURSORS];
        for (i, e) in slices.iter_mut().enumerate() {
            if let Some(s) = inputs[i].get(indices[i]..indices[i] + 16) {
                *e = s;
            } else {
                break 'outer;
            }
        }

        let masks: [u16; NUM_CURSORS] = {
            let mut a = [0u16; NUM_CURSORS];
            for (i, e) in a.iter_mut().enumerate() {
                *e = avx512_window_to_mask(
                    _mm_loadu_si128(slices[i].as_ptr().cast()),
                    reverse_shuffle_first14,
                );
            }
            a
        };

        let e = indices.iter_mut().zip(masks).try_for_each(|(idx, msk)| {
            if msk == 0 {
                Err(())
            } else {
                *idx += 1 + 13 - msk.trailing_zeros() as usize;
                Ok(())
            }
        });
        if e.is_err() {
            break;
        }
    }

    inputs
        .iter()
        .zip(indices.iter().copied())
        .find_map(|(input, index)| input.get(index..).and_then(benny).map(|v| v + index))
}

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

/// Perform `func` over `chunk_len` sized regions of `input`, in order.
/// Returns the first result.
///
/// This is useful for improving the locality of access for funcs that partition the input and process it in
/// parallel.
///
/// SAFETY: same as `func`
#[inline(always)]
unsafe fn chunked(
    input: &[u8],
    chunk_len: usize,
    func: &mut impl FnMut(&[u8]) -> Option<usize>,
) -> Option<usize> {
    let chunk_len = chunk_len.max(1024);
    let mut idx = 0usize;
    while idx < input.len() {
        if let Some(v) = func(&input[idx..input.len().min(idx + chunk_len)]) {
            return Some(v + idx);
        }
        idx += chunk_len - 13;
    }
    None
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
#[target_feature(enable = "popcnt")]
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

#[target_feature(enable = "avx512f,bmi1,avx512vpopcntdq,avx512bw")]
pub unsafe fn gather_avx512_chunked(input: &[u8]) -> Option<usize> {
    chunked(input, 1024 * 256 / 3, &mut |a| unsafe {
        gather_avx512_prefetch(a)
    })
}

#[target_feature(enable = "avx2")]
pub unsafe fn gather_avx2_few_chunked(input: &[u8]) -> Option<usize> {
    chunked(input, 1024 * 512 / 3, &mut |a| unsafe {
        gather_avx2_few_regs(a)
    })
}

#[target_feature(enable = "avx2")]
pub unsafe fn gather_avx2_chunked(input: &[u8]) -> Option<usize> {
    chunked(input, 1024 * 256 / 3, &mut |a| unsafe { gather_avx2(a) })
}

#[target_feature(enable = "avx512f,bmi1,avx512vpopcntdq,avx512bw")]
#[no_mangle]
pub unsafe fn gather_avx512_noprefetch(input: &[u8]) -> Option<usize> {
    gather_avx512_base::<false>(input, false)
}

#[target_feature(enable = "avx512f,bmi1,avx512vpopcntdq,avx512bw")]
#[no_mangle]
pub unsafe fn gather_avx512_prefetch(input: &[u8]) -> Option<usize> {
    gather_avx512_base::<true>(input, true)
}

#[target_feature(enable = "avx512f,bmi1,avx512vpopcntdq,avx512bw")]
unsafe fn gather_avx512_base<const PREFETCH: bool>(input: &[u8], validate: bool) -> Option<usize> {
    use std::arch::x86_64::{
        __m512i, _kor_mask16 as kor, _mm512_and_epi32 as and, _mm512_cmpeq_epi32_mask as cmpeq,
        _mm512_popcnt_epi32 as popcnt, _mm512_set1_epi32, _mm512_setzero_epi32,
        _mm512_sllv_epi32 as shl, _mm512_srli_epi32 as shr, _mm512_ternarylogic_epi32 as tern,
        _mm512_xor_epi32 as xor,
    };
    const OFFSET_SCALE: i32 = 4;
    let gather = std::arch::x86_64::_mm512_i32gather_epi32::<OFFSET_SCALE>;

    if input.len() < 16 * 64 {
        return bbeennnnyy(input);
    }

    // do first 3 iters so that offset[0] can be 3.
    // do next 8 iters so that we can bump up ptr to be aligned for i32
    // could do fewer, but the cost of each additional iter is very low compared to the initial 13
    if let Some(idx) = benny(&input[..13 + 3 + 8]) {
        return Some(idx);
    }

    if input.len() > i32::MAX as usize * OFFSET_SCALE as usize {
        todo!("break giant inputs into chunks to avoid overflowing the offsets");
    }

    let mut ptr = input.as_ptr();
    // we would like aligned loads in each region, and the filter init block advances the pointer
    // by 13 bytes (1 modulo 4). This makes ptr 3 modulo 4
    ptr = ptr.byte_add(ptr.align_offset(4) + 4 - 1);

    let and_v32 = _mm512_set1_epi32(31);
    let ones_v = _mm512_set1_epi32(1);
    let epi32_14_v = _mm512_set1_epi32(14);
    // offset for each region
    let offsets_arr: [i32; 16] = {
        let mut arr = [0i32; 16];
        for i in 1..16 {
            arr[i] = (i * input.len() / 16 / OFFSET_SCALE as usize)
                .try_into()
                .unwrap();
        }
        // now we can safely have each region overshoot by 16 * 3 (we prefetch 2 iterations ahead)
        arr[15] -= if PREFETCH { 48 } else { 32 } / OFFSET_SCALE;
        arr
    };

    // div by 16 for size of one region, 4 bytes per iter. Add 4 iters to handle windows that
    // straddle 2 regions
    let num_iters = input.len() / 16 / 4 + 4;
    assert!(
        !offsets_arr.iter().copied().any(i32::is_negative),
        "offsets, {offsets_arr:?} should all be non-negative"
    );

    let last_region_end =
        offsets_arr[15] as usize * 4 + num_iters * 4 + ptr.offset_from(input.as_ptr()) as usize;

    assert!(
        last_region_end < input.len(),
        "last region is over-extended, {last_region_end} !< {}",
        input.len()
    );

    let offsets_v = _mm512_loadu_si512(offsets_arr.as_ptr());

    // These store 16 lanes of epi32, each lane being 1-hot encoded
    // vi = the i-th byte of each region, each epi32 lane corresponding to a region.
    // we have a window of 17 bytes, meaning we advance each region by 4 bytes on each iter.
    let mut v1: __m512i;
    let mut v2: __m512i;
    let mut v3: __m512i;
    let mut v4: __m512i;
    let mut v5: __m512i;
    let mut v6: __m512i;
    let mut v7: __m512i;
    let mut v8: __m512i;
    let mut v9: __m512i;
    let mut v10: __m512i;
    let mut v11: __m512i;
    let mut v12: __m512i;
    let mut v13: __m512i;
    let mut v14: __m512i;
    let mut v15: __m512i;
    let mut v16: __m512i;
    let mut v17: __m512i;

    let mut filter = {
        debug_assert!(offsets_arr.iter().all(|offset| {
            let effective_addr = ptr.byte_offset(*offset as isize * OFFSET_SCALE as isize);
            input.as_ptr() <= effective_addr && effective_addr < input.as_ptr_range().end
        }));
        let data_1234 = gather(offsets_v, ptr);
        // 8 LSB are the first byte of that region.
        v1 = shl(ones_v, and(and_v32, data_1234));
        v2 = shl(ones_v, and(and_v32, shr::<8>(data_1234)));
        v3 = shl(ones_v, and(and_v32, shr::<16>(data_1234)));
        v4 = shl(ones_v, and(and_v32, shr::<24>(data_1234)));
        ptr = ptr.byte_add(4);
        let data_5678 = gather(offsets_v, ptr);
        v5 = shl(ones_v, and(and_v32, data_5678));
        v6 = shl(ones_v, and(and_v32, shr::<8>(data_5678)));
        v7 = shl(ones_v, and(and_v32, shr::<16>(data_5678)));
        v8 = shl(ones_v, and(and_v32, shr::<24>(data_5678)));
        ptr = ptr.byte_add(4);
        let data_9_10_11_12 = gather(offsets_v, ptr);
        v9 = shl(ones_v, and(and_v32, data_9_10_11_12));
        v10 = shl(ones_v, and(and_v32, shr::<8>(data_9_10_11_12)));
        v11 = shl(ones_v, and(and_v32, shr::<16>(data_9_10_11_12)));
        v12 = shl(ones_v, and(and_v32, shr::<24>(data_9_10_11_12)));
        ptr = ptr.byte_add(4);
        let data_13_x_x_x = gather(offsets_v, ptr);
        v13 = shl(ones_v, and(and_v32, data_13_x_x_x));
        ptr = ptr.byte_add(1);

        [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13]
            .into_iter()
            .fold(_mm512_setzero_epi32(), |acc, v| unsafe { xor(acc, v) })
    };

    let mut mask1: u16 = 0;
    let mut mask2: u16 = 0;
    let mut mask3: u16 = 0;
    let mut mask4: u16 = 0;

    assert_eq!(
        (ptr as usize) % 4, 0,
        "To improve the gather performance, ptr should be aligned for i32 after initializing the filter"
    );

    /*
    load gets
     data: |aaaa|bbbb|cccc|dddd|eeee|ffff|gggg|hhhh|... until 16th letter
    4 bytes of 16 regions.
    we need 1hot encoding in each lane, meaning one byte of input becomes an i32.
    Split this interleaved data into 4 with shr:
     data: |4321|...
    becomes
     v14: |___1|...
     v15: |___2|...
     v16: |___3|...
     v17: |___4|...

    after that, it's 'just' benny's algo mapped to vertical SIMD ops.
    One iteration will have consumed 4 bytes in 16 regions -> 64 bytes consumed.
    Keep in mind that each iteration of this is FAR more expensive than benny's.

    According to Agner Fog's instruction tables, _mm512_i32gather_epi32 has (latency, R-thrpt) of (81, 17).
    Based on a few test runs, the CPUs instruction re-ordering was not enough to hide the latency. IPC was low (~1.1), but cache misses weren't too high either (~2%)

    So we "pre-fetch" 2 iterations ahead to give the gathers time to complete. It's like async in the hardware :).
    I also tried with 3 iterations ahead, it did not yield further improvements.

    diagram:
    | 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|
    \=====================================/|==========/|==========/|==========/
     v1 - v13 carried over from prev iter  | data      |           |
                                                       |next_data  |
                                                                   |next_next_data
    We get the first values of v1-v13 when building the filter before the loop.
    Right before the loop, we start the gathers for 'data' and 'next_data'
    The current iter starts the gather for the next next iterations,
    and we are always 2 gathers ahead of the current working window.
    As a reminder, this is to hide the high latency of gather, especially on Zen4.
    */
    // if no PREFETCH, then just ignore these values in the inner loop. Safe to load these here
    // because we ruled out small inputs,
    // and the compiler shouldn't dedicate registers to these two since they're unused
    let mut data = gather(offsets_v, ptr);
    let mut next_data = gather(offsets_v, ptr.byte_add(4));
    for _ in 0..num_iters {
        debug_assert!(offsets_arr.iter().all(|offset| {
            let effective_addr = ptr.byte_offset(*offset as isize * OFFSET_SCALE as isize);
            input.as_ptr() <= effective_addr && effective_addr < input.as_ptr_range().end
        }));
        // if no prefetch, load the data for this iteration (0 byte offset)
        let next_next_data = gather(offsets_v, ptr.byte_add(if PREFETCH { 8 } else { 0 }));
        if !PREFETCH {
            data = next_next_data;
        }
        if validate {
            use std::arch::x86_64::*;
            if 0 != _mm512_cmpgt_epi8_mask(
                _mm512_sub_epi8(data, _mm512_set1_epi8(b'a' as i8)),
                _mm512_set1_epi8(25),
            ) {
                // todo: return Result instead
                panic!("input contains characters outside the range a-z")
            }
        }

        // now every byte is 0 - 25 (assuming valid input). (very easy to add validation here, too!)

        v14 = shl(ones_v, and(and_v32, data));
        v15 = shl(ones_v, and(and_v32, shr::<8>(data)));
        v16 = shl(ones_v, and(and_v32, shr::<16>(data)));
        v17 = shl(ones_v, and(and_v32, shr::<24>(data)));

        if PREFETCH {
            data = next_data;
            next_data = next_next_data;
        }
        // prevent/limit re-ordering. LLVM wants to move this to after the branch on masks
        std::arch::asm!("");

        const TERN_TABLE: i32 = 150;
        // xor in 14th, count, xor out 1st
        // the adjacent xors get turned into vpternlogd
        filter = xor(filter, v14);
        mask1 = cmpeq(popcnt(filter), epi32_14_v);
        filter = tern::<TERN_TABLE>(filter, v1, v15);
        mask2 = cmpeq(popcnt(filter), epi32_14_v);
        filter = tern::<TERN_TABLE>(filter, v2, v16);
        mask3 = cmpeq(popcnt(filter), epi32_14_v);
        filter = tern::<TERN_TABLE>(filter, v3, v17);
        mask4 = cmpeq(popcnt(filter), epi32_14_v);
        filter = xor(filter, v4);

        if kor(kor(mask1, mask2), kor(mask3, mask4)) != 0 {
            break;
        }

        ptr = ptr.byte_add(4);

        /*
        shift the top 13 registers down, opening up the top 4 for the next batch
        start of iter (after load)
        | 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|13|14|15|16|17|
        shifted out 1-4, in 14-17
        rotate down by 4 to get
        | 5| 6| 7| 8| 9|10|11|12|13|14|15|16|17|__|__|__|__|
        */
        v1 = v5;
        v2 = v6;
        v3 = v7;
        v4 = v8;
        v5 = v9;
        v6 = v10;
        v7 = v11;
        v8 = v12;
        v9 = v13;
        v10 = v14;
        v11 = v15;
        v12 = v16;
        v13 = v17;
    }

    let ptr_offset = ptr.offset_from(input.as_ptr()) as usize;
    // We've either found a solution or ran out of input.
    // if the found solution wasn't in the first region, we have to finish checking all earlier regions
    let mask = mask1 | mask2 | mask3 | mask4;
    if mask != 0 {
        // 0th bit is first region
        0..mask.trailing_zeros() + 1
    } else {
        0..0
    }
    .find_map(|region_idx| {
        let start =
            ptr_offset - 13 + offsets_arr[region_idx as usize] as usize * OFFSET_SCALE as usize;
        let end = if region_idx == 15 {
            usize::MAX
        } else {
            offsets_arr[region_idx as usize + 1] as usize * OFFSET_SCALE as usize + 64
        }
        .min(input.len());
        gather_avx512_base::<PREFETCH>(&input[start..end], validate).map(|v| v + start)
    })
    .or_else(|| {
        let benny_start = last_region_end - 13 - 16;
        benny(&input[benny_start..]).map(|v| v + benny_start)
    })
}

#[target_feature(enable = "avx2")]
pub unsafe fn slow_mm256_popcnt_epi32(v: std::arch::x86_64::__m256i) -> std::arch::x86_64::__m256i {
    let arr: [i32; 8] = std::mem::transmute_copy(&v);
    let arr = arr.map(|i| i.count_ones() as i32);
    std::mem::transmute_copy(&arr)
}

/// Like `gather_benny_avx512`, but for AVX2.
/// Based on Eren's `benny_no_popcnt` variant of `benny`, since AVX2 lacks vpopcnt
#[target_feature(enable = "avx2")]
#[no_mangle]
pub unsafe fn gather_avx2(input: &[u8]) -> Option<usize> {
    use std::arch::x86_64::__m256i;
    use std::arch::x86_64::{
        _mm256_add_epi32 as add, _mm256_and_si256 as and, _mm256_andnot_si256 as andnot,
        _mm256_cmpeq_epi32 as cmpeq, _mm256_loadu_epi32, _mm256_set1_epi32, _mm256_setzero_si256,
        _mm256_sllv_epi32 as shl, _mm256_srli_epi32 as shr, _mm256_sub_epi32 as sub,
        _mm256_xor_si256 as xor,
    };
    const OFFSET_SCALE: i32 = 4;
    let gather = std::arch::x86_64::_mm256_i32gather_epi32::<OFFSET_SCALE>;
    let movemask = |v| unsafe { _mm256_movemask_ps(_mm256_castsi256_ps(v)) as u8 };

    if input.len() < 8 * 1024 {
        return bbeennnnyy(input);
    }

    // do first 3 iters so that offset[0] can be 3.
    // do next 8 iters so that we can bump up ptr to be aligned for i32
    // could do fewer, but the cost of each additional iter is very low compared to the initial 13
    if let Some(idx) = benny(&input[..13 + 3 + 8]) {
        return Some(idx);
    }

    if input.len() > i32::MAX as usize * OFFSET_SCALE as usize {
        todo!("break giant inputs into chunks to avoid overflowing the offsets");
    }

    let mut ptr = input.as_ptr();
    // we would like aligned loads in each region, and the filter init block advances the pointer
    // by 13 bytes (1 modulo 4). This makes ptr 3 modulo 4
    ptr = ptr.byte_add(ptr.align_offset(4) + 4 - 1);

    let and_v32 = _mm256_set1_epi32(31);
    let ones_v = _mm256_set1_epi32(1);
    let epi32_13_v = _mm256_set1_epi32(13);
    // offset for each region
    let offsets_arr: [i32; 8] = {
        let mut arr = [0i32; 8];
        for i in 1..8 {
            arr[i] = (i * input.len() / 8 / OFFSET_SCALE as usize)
                .try_into()
                .unwrap();
        }
        // now we can safely have each region overshoot by 16 (see next comment for why)
        arr[7] -= 48 / OFFSET_SCALE;
        arr
    };

    // div by 16 for size of one region, 4 bytes per iter. Add 4 iters to handle windows that
    // straddle 2 regions
    let num_iters = input.len() / 8 / 4 + 4;
    assert!(
        !offsets_arr.iter().copied().any(i32::is_negative),
        "offsets, {offsets_arr:?} should all be non-negative"
    );

    let last_region_end = offsets_arr[7] as usize * OFFSET_SCALE as usize
        + num_iters * 4
        + ptr.offset_from(input.as_ptr()) as usize;

    assert!(
        last_region_end < input.len(),
        "last region is over-extended, {last_region_end} !< {}",
        input.len()
    );

    let offsets_v = _mm256_loadu_epi32(offsets_arr.as_ptr().cast());

    // These store 8 lanes of epi32, each lane being 1-hot encoded
    // vi = the i-th byte of each region, each epi32 lane corresponding to a region.
    // we have a window of 17 bytes, meaning we advance each region by 4 bytes on each iter.
    let mut v1: __m256i;
    let mut v2: __m256i;
    let mut v3: __m256i;
    let mut v4: __m256i;
    let mut v5: __m256i;
    let mut v6: __m256i;
    let mut v7: __m256i;
    let mut v8: __m256i;
    let mut v9: __m256i;
    let mut v10: __m256i;
    let mut v11: __m256i;
    let mut v12: __m256i;
    let mut v13: __m256i;
    let mut v14: __m256i;
    let mut v15: __m256i;
    let mut v16: __m256i;
    let mut v17: __m256i;

    let mut filter = {
        debug_assert!(offsets_arr.iter().all(|offset| {
            let effective_addr = ptr.byte_offset(*offset as isize * OFFSET_SCALE as isize);
            input.as_ptr() <= effective_addr && effective_addr < input.as_ptr_range().end
        }));
        let data_1234 = gather(ptr.cast(), offsets_v);
        // 8 LSB are the first byte of that region.
        v1 = shl(ones_v, and(and_v32, data_1234));
        v2 = shl(ones_v, and(and_v32, shr::<8>(data_1234)));
        v3 = shl(ones_v, and(and_v32, shr::<16>(data_1234)));
        v4 = shl(ones_v, and(and_v32, shr::<24>(data_1234)));
        ptr = ptr.byte_add(4);
        let data_5678 = gather(ptr.cast(), offsets_v);
        v5 = shl(ones_v, and(and_v32, data_5678));
        v6 = shl(ones_v, and(and_v32, shr::<8>(data_5678)));
        v7 = shl(ones_v, and(and_v32, shr::<16>(data_5678)));
        v8 = shl(ones_v, and(and_v32, shr::<24>(data_5678)));
        ptr = ptr.byte_add(4);
        let data_9_10_11_12 = gather(ptr.cast(), offsets_v);
        v9 = shl(ones_v, and(and_v32, data_9_10_11_12));
        v10 = shl(ones_v, and(and_v32, shr::<8>(data_9_10_11_12)));
        v11 = shl(ones_v, and(and_v32, shr::<16>(data_9_10_11_12)));
        v12 = shl(ones_v, and(and_v32, shr::<24>(data_9_10_11_12)));
        ptr = ptr.byte_add(4);
        let data_13_x_x_x = gather(ptr.cast(), offsets_v);
        v13 = shl(ones_v, and(and_v32, data_13_x_x_x));
        ptr = ptr.byte_add(1);

        [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13]
            .into_iter()
            .fold(_mm256_setzero_si256(), |acc, v| unsafe { xor(acc, v) })
    };

    let mut mask1: u8 = 0;
    let mut mask2: u8 = 0;
    let mut mask3: u8 = 0;
    let mut mask4: u8 = 0;

    assert_eq!(
        (ptr as usize) % 4, 0,
        "To improve the gather performance, ptr should be aligned for i32 after initializing the filter"
    );

    let mut pseudo_count = slow_mm256_popcnt_epi32(filter);
    let mut data = gather(ptr.cast(), offsets_v);
    for _ in 0..num_iters {
        debug_assert!(offsets_arr.iter().all(|offset| {
            let effective_addr = ptr.byte_offset(*offset as isize * OFFSET_SCALE as isize);
            input.as_ptr() <= effective_addr && effective_addr < input.as_ptr_range().end
        }));
        let next_data = gather(ptr.byte_add(4).cast(), offsets_v);

        // now every byte is 0 - 25 (assuming valid input). (very easy to add validation here, too!)

        v14 = shl(ones_v, and(and_v32, data));
        v15 = shl(ones_v, and(and_v32, shr::<8>(data)));
        v16 = shl(ones_v, and(and_v32, shr::<16>(data)));
        v17 = shl(ones_v, and(and_v32, shr::<24>(data)));

        // prevent/limit re-ordering. LLVM wants to move this to after the branch on masks
        data = next_data;
        std::arch::asm!("");

        // xor in new, count, xor out oldest

        // corresponds to res in `benny_no_popcnt`
        let v14_is_pseudo_new = cmpeq(andnot(filter, v14), v14);
        mask1 = movemask(and(cmpeq(pseudo_count, epi32_13_v), v14_is_pseudo_new));
        filter = xor(filter, v14);
        let v1_is_pseudo_old = cmpeq(and(filter, v1), v1);
        filter = xor(filter, v1);
        // cmpeq sets true lanes to 0xFF..., aka -1
        let delta_count = sub(v14_is_pseudo_new, v1_is_pseudo_old);
        pseudo_count = sub(pseudo_count, add(delta_count, delta_count));

        let v15_is_pseudo_new = cmpeq(andnot(filter, v15), v15);
        mask2 = movemask(and(cmpeq(pseudo_count, epi32_13_v), v15_is_pseudo_new));
        filter = xor(filter, v15);
        let v2_is_pseudo_old = cmpeq(and(filter, v2), v2);
        filter = xor(filter, v2);
        let delta_count = sub(v15_is_pseudo_new, v2_is_pseudo_old);
        pseudo_count = sub(pseudo_count, add(delta_count, delta_count));

        let v16_is_pseudo_new = cmpeq(andnot(filter, v16), v16);
        mask3 = movemask(and(cmpeq(pseudo_count, epi32_13_v), v16_is_pseudo_new));
        filter = xor(filter, v16);
        let v3_is_pseudo_old = cmpeq(and(filter, v3), v3);
        filter = xor(filter, v3);
        let delta_count = sub(v16_is_pseudo_new, v3_is_pseudo_old);
        pseudo_count = sub(pseudo_count, add(delta_count, delta_count));

        let v17_is_pseudo_new = cmpeq(andnot(filter, v17), v17);
        mask4 = movemask(and(cmpeq(pseudo_count, epi32_13_v), v17_is_pseudo_new));
        filter = xor(filter, v17);
        let v4_is_pseudo_old = cmpeq(and(filter, v4), v4);
        filter = xor(filter, v4);
        let delta_count = sub(v17_is_pseudo_new, v4_is_pseudo_old);
        pseudo_count = sub(pseudo_count, add(delta_count, delta_count));

        if (mask1 | mask2 | mask3 | mask4) != 0 {
            break;
        }

        ptr = ptr.byte_add(4);

        /*
        shift the top 13 registers down, opening up the top 4 for the next batch
        start of iter (after load)
        | 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|13|14|15|16|17|
        shifted out 1-4, in 14-17
        rotate down by 4 to get
        | 5| 6| 7| 8| 9|10|11|12|13|14|15|16|17|__|__|__|__|
        */
        v1 = v5;
        v2 = v6;
        v3 = v7;
        v4 = v8;
        v5 = v9;
        v6 = v10;
        v7 = v11;
        v8 = v12;
        v9 = v13;
        v10 = v14;
        v11 = v15;
        v12 = v16;
        v13 = v17;
    }

    let ptr_offset = ptr.offset_from(input.as_ptr()) as usize;
    // We've either found a solution or ran out of input.
    // if the found solution wasn't in the first region, we have to finish checking all earlier regions
    let mask = mask1 | mask2 | mask3 | mask4;
    if mask != 0 {
        // 0th bit is first region
        0..mask.trailing_zeros() + 1
    } else {
        0..0
    }
    .find_map(|region_idx| {
        let start =
            ptr_offset - 13 + offsets_arr[region_idx as usize] as usize * OFFSET_SCALE as usize;
        let end = if region_idx == 7 {
            usize::MAX
        } else {
            offsets_arr[region_idx as usize + 1] as usize * OFFSET_SCALE as usize + 64
        }
        .min(input.len());
        gather_avx2(&input[start..end]).map(|v| v + start)
    })
    .or_else(|| {
        let benny_start = last_region_end - 13 - 16;
        benny(&input[benny_start..]).map(|v| v + benny_start)
    })
}

/// Like `gather_benny_avx512`, but for AVX2.
/// Based on Eren's `benny_no_popcnt` variant of `benny`, since AVX2 lacks vpopcnt
/// Since AVX2 only grants 16 registers, the compiler has to spill several registers every
/// iteration. By doing it manually, we can spill fewer.
#[target_feature(enable = "avx2")]
#[no_mangle]
pub unsafe fn gather_avx2_few_regs(input: &[u8]) -> Option<usize> {
    use std::arch::x86_64::{
        __m256i, _mm256_add_epi32 as add, _mm256_and_si256 as and, _mm256_andnot_si256 as andnot,
        _mm256_cmpeq_epi32 as cmpeq, _mm256_loadu_epi32, _mm256_set1_epi32, _mm256_setzero_si256,
        _mm256_sllv_epi32 as shl, _mm256_srli_epi32 as shr, _mm256_sub_epi32 as sub,
        _mm256_xor_si256 as xor,
    };
    const OFFSET_SCALE: i32 = 4;
    let gather = std::arch::x86_64::_mm256_i32gather_epi32::<OFFSET_SCALE>;
    let movemask = |v| unsafe { _mm256_movemask_ps(_mm256_castsi256_ps(v)) as u8 };

    if input.len() < 8 * 1024 {
        return bbeennnnyy(input);
    }

    // do first 3 iters so that offset[0] can be 3.
    // do next 8 iters so that we can bump up ptr to be aligned for i32
    // could do fewer, but the cost of each additional iter is very low compared to the initial 13
    if let Some(idx) = benny(&input[..13 + 3 + 8]) {
        return Some(idx);
    }

    if input.len() > i32::MAX as usize * OFFSET_SCALE as usize {
        todo!("break giant inputs into chunks to avoid overflowing the offsets");
    }

    let mut ptr = input.as_ptr();
    // we would like aligned loads in each region, and the filter init block advances the pointer
    // by 13 bytes (1 modulo 4). This makes ptr 3 modulo 4
    ptr = ptr.byte_add(ptr.align_offset(4) + 4 - 1);

    let and_v32 = _mm256_set1_epi32(31);
    let ones_v = _mm256_set1_epi32(1);
    let epi32_13_v = _mm256_set1_epi32(13);
    // offset for each region
    let offsets_arr: [i32; 8] = {
        let mut arr = [0i32; 8];
        for i in 1..8 {
            arr[i] = (i * input.len() / 8 / OFFSET_SCALE as usize)
                .try_into()
                .unwrap();
        }
        // now we can safely have each region overshoot by 16 (see next comment for why)
        arr[7] -= 48 / OFFSET_SCALE;
        arr
    };

    // div by 16 for size of one region, 4 bytes per iter. Add 4 iters to handle windows that
    // straddle 2 regions
    let num_iters = input.len() / 8 / 4 + 4;
    assert!(
        !offsets_arr.iter().copied().any(i32::is_negative),
        "offsets, {offsets_arr:?} should all be non-negative"
    );

    let last_region_end = offsets_arr[7] as usize * OFFSET_SCALE as usize
        + num_iters * 4
        + ptr.offset_from(input.as_ptr()) as usize;

    assert!(
        last_region_end < input.len(),
        "last region is over-extended, {last_region_end} !< {}",
        input.len()
    );

    let offsets_v = _mm256_loadu_epi32(offsets_arr.as_ptr().cast());

    // points to oldest v
    let mut v_idx = 0usize;
    // wrap-around
    let mut v_last14: [__m256i; 16] = [_mm256_setzero_si256(); 16];

    let mut filter = {
        debug_assert!(offsets_arr.iter().all(|offset| {
            let effective_addr = ptr.byte_offset(*offset as isize * OFFSET_SCALE as isize);
            input.as_ptr() <= effective_addr && effective_addr < input.as_ptr_range().end
        }));
        let data_1234 = gather(ptr.cast(), offsets_v);
        // 8 LSB are the first byte of that region.
        let v1 = shl(ones_v, and(and_v32, data_1234));
        let v2 = shl(ones_v, and(and_v32, shr::<8>(data_1234)));
        let v3 = shl(ones_v, and(and_v32, shr::<16>(data_1234)));
        let v4 = shl(ones_v, and(and_v32, shr::<24>(data_1234)));
        ptr = ptr.byte_add(4);
        let data_5678 = gather(ptr.cast(), offsets_v);
        let v5 = shl(ones_v, and(and_v32, data_5678));
        let v6 = shl(ones_v, and(and_v32, shr::<8>(data_5678)));
        let v7 = shl(ones_v, and(and_v32, shr::<16>(data_5678)));
        let v8 = shl(ones_v, and(and_v32, shr::<24>(data_5678)));
        ptr = ptr.byte_add(4);
        let data_9_10_11_12 = gather(ptr.cast(), offsets_v);
        let v9 = shl(ones_v, and(and_v32, data_9_10_11_12));
        let v10 = shl(ones_v, and(and_v32, shr::<8>(data_9_10_11_12)));
        let v11 = shl(ones_v, and(and_v32, shr::<16>(data_9_10_11_12)));
        let v12 = shl(ones_v, and(and_v32, shr::<24>(data_9_10_11_12)));
        ptr = ptr.byte_add(4);
        let data_13_x_x_x = gather(ptr.cast(), offsets_v);
        let v13 = shl(ones_v, and(and_v32, data_13_x_x_x));
        ptr = ptr.byte_add(1);

        v_last14
            .iter_mut()
            .zip([v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13])
            .for_each(|(out, v)| *out = v);

        [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13]
            .into_iter()
            .fold(_mm256_setzero_si256(), |acc, v| unsafe { xor(acc, v) })
    };

    let mut mask1: u8 = 0;
    let mut mask2: u8 = 0;
    let mut mask3: u8 = 0;
    let mut mask4: u8 = 0;

    assert_eq!(
        (ptr as usize) % 4, 0,
        "To improve the gather performance, ptr should be aligned for i32 after initializing the filter"
    );

    let mut pseudo_count = slow_mm256_popcnt_epi32(filter);
    let mut data = gather(ptr.cast(), offsets_v);
    for _ in 0..num_iters {
        debug_assert!(offsets_arr.iter().all(|offset| {
            let effective_addr = ptr.byte_offset(*offset as isize * OFFSET_SCALE as isize);
            input.as_ptr() <= effective_addr && effective_addr < input.as_ptr_range().end
        }));
        let next_data = gather(ptr.byte_add(4).cast(), offsets_v);

        // now every byte is 0 - 25 (assuming valid input). (very easy to add validation here, too!)

        let v14 = shl(ones_v, and(and_v32, data));
        let v15 = shl(ones_v, and(and_v32, shr::<8>(data)));
        let v16 = shl(ones_v, and(and_v32, shr::<16>(data)));
        let v17 = shl(ones_v, and(and_v32, shr::<24>(data)));

        let v1 = v_last14[v_idx];
        v_idx = (v_idx + 1) & 15;
        let v2 = v_last14[v_idx];
        v_idx = (v_idx + 1) & 15;
        let v3 = v_last14[v_idx];
        v_idx = (v_idx + 1) & 15;
        let v4 = v_last14[v_idx];
        v_idx = (v_idx + 1) & 15;

        {
            v_last14[(v_idx + 9) & 15] = v14;
            v_last14[(v_idx + 10) & 15] = v15;
            v_last14[(v_idx + 11) & 15] = v16;
            v_last14[(v_idx + 12) & 15] = v17;
        }
        // prevent/limit re-ordering. LLVM wants to move this to after the branch on masks
        data = next_data;
        std::arch::asm!("");

        // xor in new, count, xor out oldest

        // corresponds to res in `benny_no_popcnt`
        let v14_is_pseudo_new = cmpeq(andnot(filter, v14), v14);
        mask1 = movemask(and(cmpeq(pseudo_count, epi32_13_v), v14_is_pseudo_new));
        filter = xor(filter, v14);
        let v1_is_pseudo_old = cmpeq(and(filter, v1), v1);
        filter = xor(filter, v1);
        // cmpeq sets true lanes to 0xFF..., aka -1
        let delta_count = sub(v14_is_pseudo_new, v1_is_pseudo_old);
        pseudo_count = sub(pseudo_count, add(delta_count, delta_count));

        let v15_is_pseudo_new = cmpeq(andnot(filter, v15), v15);
        mask2 = movemask(and(cmpeq(pseudo_count, epi32_13_v), v15_is_pseudo_new));
        filter = xor(filter, v15);
        let v2_is_pseudo_old = cmpeq(and(filter, v2), v2);
        filter = xor(filter, v2);
        let delta_count = sub(v15_is_pseudo_new, v2_is_pseudo_old);
        pseudo_count = sub(pseudo_count, add(delta_count, delta_count));

        let v16_is_pseudo_new = cmpeq(andnot(filter, v16), v16);
        mask3 = movemask(and(cmpeq(pseudo_count, epi32_13_v), v16_is_pseudo_new));
        filter = xor(filter, v16);
        let v3_is_pseudo_old = cmpeq(and(filter, v3), v3);
        filter = xor(filter, v3);
        let delta_count = sub(v16_is_pseudo_new, v3_is_pseudo_old);
        pseudo_count = sub(pseudo_count, add(delta_count, delta_count));

        let v17_is_pseudo_new = cmpeq(andnot(filter, v17), v17);
        mask4 = movemask(and(cmpeq(pseudo_count, epi32_13_v), v17_is_pseudo_new));
        filter = xor(filter, v17);
        let v4_is_pseudo_old = cmpeq(and(filter, v4), v4);
        filter = xor(filter, v4);
        let delta_count = sub(v17_is_pseudo_new, v4_is_pseudo_old);
        pseudo_count = sub(pseudo_count, add(delta_count, delta_count));

        if (mask1 | mask2 | mask3 | mask4) != 0 {
            break;
        }
        ptr = ptr.byte_add(4);
    }

    let ptr_offset = ptr.offset_from(input.as_ptr()) as usize;
    // We've either found a solution or ran out of input.
    // if the found solution wasn't in the first region, we have to finish checking all earlier regions
    let mask = mask1 | mask2 | mask3 | mask4;
    if mask != 0 {
        // 0th bit is first region
        0..mask.trailing_zeros() + 1
    } else {
        0..0
    }
    .find_map(|region_idx| {
        let start =
            ptr_offset - 13 + offsets_arr[region_idx as usize] as usize * OFFSET_SCALE as usize;
        let end = if region_idx == 7 {
            usize::MAX
        } else {
            offsets_arr[region_idx as usize + 1] as usize * OFFSET_SCALE as usize + 64
        }
        .min(input.len());
        gather_avx2_few_regs(&input[start..end]).map(|v| v + start)
    })
    .or_else(|| {
        let benny_start = last_region_end - 13 - 16;
        benny(&input[benny_start..]).map(|v| v + benny_start)
    })
}
