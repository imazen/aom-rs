//! Differential harness for intra-block-copy DV prediction
//! (`aom_entropy::dv_ref`) vs the REAL C: [`find_dv_ref_mvs`] against the
//! actually-exported (non-static) `av1_find_mv_refs` + `av1_find_best_ref_mvs`
//! (called directly over a synthetic MI grid, not transcribed);
//! [`find_ref_dv`] / [`is_dv_valid`] against the real `static inline`
//! `av1_find_ref_dv` / `av1_is_dv_valid`. A fourth test composes the three
//! verified C primitives into a manual oracle for
//! [`aom_entropy::dv_ref::assign_and_validate_dv`] — our own glue code
//! chaining `read_intrabc_info`'s caller-side sequencing
//! (`decodemv.c::read_intrabc_info` + `assign_dv`), not itself a single C
//! function, so this cross-checks the WIRING between the three primitives
//! independently of their own correctness.

use aom_entropy::dv_ref::{
    assign_and_validate_dv, find_dv_ref_mvs, find_ref_dv, is_dv_valid, DvNbr, DvTileBounds,
};
use aom_sys_ref::{self as c, RefDvNbr};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn range(&mut self, lo: i64, hi: i64) -> i64 {
        // [lo, hi)
        lo + (self.next() % ((hi - lo) as u64)) as i64
    }
}

const DIM: usize = c::REF_DV_GRID_DIM; // 128

/// Fill a `DIM x DIM` flat grid with a realistic-but-adversarial mix of
/// candidate content: intrabc-DV candidates, plain-intra candidates (garbage
/// mv, must be ignored), and out-of-KEY-frame-envelope inter-ref candidates
/// (exercises `is_inter_block`'s `ref_frame[0] > INTRA_FRAME` arm and
/// `process_single_ref_mv_candidate`'s general form — both provably dead on
/// our KEY-frame-only envelope, but validated here against the REAL C so the
/// port is correct beyond just the reachable slice, matching this module's
/// "ported in full, not hand-simplified" discipline).
fn random_grid(rng: &mut Rng) -> Vec<RefDvNbr> {
    let mut g = vec![RefDvNbr::default(); DIM * DIM];
    for cell in g.iter_mut() {
        let bsize = rng.range(0, 22) as u8;
        let class = rng.next() % 100;
        if class < 35 {
            // intrabc DV candidate.
            cell.bsize = bsize;
            cell.ref_frame0 = 0; // INTRA_FRAME
            cell.ref_frame1 = -1; // NONE_FRAME
            cell.use_intrabc = true;
            cell.mode = 0; // DC_PRED
            cell.mv0_row = (rng.range(-256, 256) * 8) as i16;
            cell.mv0_col = (rng.range(-256, 256) * 8) as i16;
        } else if class < 70 {
            // plain intra: is_inter_block must gate this out; garbage mv/mode.
            cell.bsize = bsize;
            cell.ref_frame0 = 0; // INTRA_FRAME
            cell.ref_frame1 = -1;
            cell.use_intrabc = false;
            cell.mode = rng.range(0, 13) as u8;
            cell.mv0_row = rng.range(-30000, 30000) as i16;
            cell.mv0_col = rng.range(-30000, 30000) as i16;
            cell.mv1_row = rng.range(-30000, 30000) as i16;
            cell.mv1_col = rng.range(-30000, 30000) as i16;
        } else if class < 90 {
            // out-of-envelope real inter ref: is_inter_block true via
            // ref_frame[0] > INTRA_FRAME (never true on a real KEY frame,
            // ported+verified anyway).
            cell.bsize = bsize;
            cell.ref_frame0 = rng.range(1, 8) as i8; // LAST_FRAME..ALTREF_FRAME
            cell.ref_frame1 = if rng.next().is_multiple_of(2) {
                -1
            } else {
                rng.range(1, 8) as i8
            };
            cell.use_intrabc = false;
            cell.mode = rng.range(0, 22) as u8; // may include GLOBALMV(15)/GLOBAL_GLOBALMV(21)
            cell.mv0_row = (rng.range(-256, 256) * 8) as i16;
            cell.mv0_col = (rng.range(-256, 256) * 8) as i16;
            cell.mv1_row = (rng.range(-256, 256) * 8) as i16;
            cell.mv1_col = (rng.range(-256, 256) * 8) as i16;
        } else {
            // NONE_FRAME sentinel cell.
            cell.bsize = bsize;
            cell.ref_frame0 = -1;
            cell.ref_frame1 = -1;
            cell.use_intrabc = false;
            cell.mode = 0;
        }
    }
    g
}

fn grid_fn(grid: &[RefDvNbr], mi_row: i32, mi_col: i32) -> impl Fn(i32, i32) -> DvNbr + '_ {
    move |row_off: i32, col_off: i32| {
        let r = mi_row + row_off;
        let cc = mi_col + col_off;
        assert!(
            r >= 0 && (r as usize) < DIM && cc >= 0 && (cc as usize) < DIM,
            "grid probe ({row_off},{col_off}) from ({mi_row},{mi_col}) landed at ({r},{cc}), outside [0,{DIM})"
        );
        let cell = grid[(r as usize) * DIM + (cc as usize)];
        DvNbr {
            bsize: cell.bsize as usize,
            ref_frame0: cell.ref_frame0 as i32,
            ref_frame1: cell.ref_frame1 as i32,
            use_intrabc: cell.use_intrabc,
            mode: cell.mode as i32,
            mv0_row: cell.mv0_row as i32,
            mv0_col: cell.mv0_col as i32,
            mv1_row: cell.mv1_row as i32,
            mv1_col: cell.mv1_col as i32,
        }
    }
}

/// `MI_SIZE_WIDE`/`MI_SIZE_HIGH` (`common_data.h`), needed here purely to
/// size a valid random case (NOT re-exported from `dv_ref` — kept
/// intentionally private there).
const MI_SIZE_WIDE: [i32; 22] = [
    1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 1, 4, 2, 8, 4, 16,
];
const MI_SIZE_HIGH: [i32; 22] = [
    1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 4, 1, 8, 2, 16, 4,
];

#[derive(Clone, Copy, Debug)]
struct Case {
    mi_row: i32,
    mi_col: i32,
    bsize: usize,
    own_partition: usize,
    tile: DvTileBounds,
    frame_mi_rows: i32,
    frame_mi_cols: i32,
    mib_size: i32,
}

/// A random, self-consistent (position/size/tile/frame all fit inside the
/// `[0, DIM)` grid with margin for the scan's reach) test case.
fn random_case(rng: &mut Rng) -> Case {
    let bsize = rng.range(0, 22) as usize;
    let width_mi = MI_SIZE_WIDE[bsize];
    let height_mi = MI_SIZE_HIGH[bsize];
    // Safe position range: [24, 64) leaves >=16 mi of margin above/left of the
    // lowest position for the scan's ~8-mi reach, and up to 63+32=95 < 128 for
    // the block's own footprint (BLOCK_128X128 = 32 mi) to the bottom/right.
    let mi_row = rng.range(24, 64) as i32;
    let mi_col = rng.range(24, 64) as i32;
    let own_partition = rng.range(0, 10) as usize;
    let mib_size = if rng.next().is_multiple_of(2) { 16 } else { 32 };

    let frame_mi_rows = rng.range((mi_row + height_mi) as i64, (DIM as i64) + 1) as i32;
    let frame_mi_cols = rng.range((mi_col + width_mi) as i64, (DIM as i64) + 1) as i32;

    let tile = if rng.next() % 100 < 70 {
        DvTileBounds {
            mi_row_start: 0,
            mi_row_end: frame_mi_rows,
            mi_col_start: 0,
            mi_col_end: frame_mi_cols,
        }
    } else {
        let mi_row_start = rng.range(0, (mi_row + 1) as i64) as i32;
        let mi_row_end = rng.range((mi_row + height_mi) as i64, (frame_mi_rows as i64) + 1) as i32;
        let mi_col_start = rng.range(0, (mi_col + 1) as i64) as i32;
        let mi_col_end = rng.range((mi_col + width_mi) as i64, (frame_mi_cols as i64) + 1) as i32;
        DvTileBounds {
            mi_row_start,
            mi_row_end,
            mi_col_start,
            mi_col_end,
        }
    };

    Case {
        mi_row,
        mi_col,
        bsize,
        own_partition,
        tile,
        frame_mi_rows,
        frame_mi_cols,
        mib_size,
    }
}

/// The DV reference-MV quad `find_dv_ref_mvs` returns:
/// `(nearest_row, nearest_col, near_row, near_col)` in 1/8-pel units.
type DvRefMvs = (i32, i32, i32, i32);

fn run_one(case: Case, grid: &[RefDvNbr]) -> (DvRefMvs, DvRefMvs) {
    let up_available = case.mi_row > case.tile.mi_row_start;
    let left_available = case.mi_col > case.tile.mi_col_start;

    let rust_grid = grid_fn(grid, case.mi_row, case.mi_col);
    let rust_out = find_dv_ref_mvs(
        case.mi_row,
        case.mi_col,
        case.bsize,
        case.own_partition,
        up_available,
        left_available,
        case.tile,
        case.frame_mi_rows,
        case.frame_mi_cols,
        case.mib_size,
        rust_grid,
    );

    let c_out = c::ref_find_dv_ref_mvs(
        case.mi_row,
        case.mi_col,
        case.bsize,
        case.own_partition,
        up_available,
        left_available,
        case.tile.mi_row_start,
        case.tile.mi_row_end,
        case.tile.mi_col_start,
        case.tile.mi_col_end,
        case.frame_mi_rows,
        case.frame_mi_cols,
        case.mib_size,
        grid,
    );
    (rust_out, c_out)
}

#[test]
fn find_dv_ref_mvs_matches_c() {
    let mut rng = Rng(0xd7a5_1c0d_e000_0001);
    // Each case's dominant cost is the O(DIM^2) grid FFI marshal
    // (`ref_find_dv_ref_mvs` re-flattens 9 parallel arrays every call
    // regardless of grid reuse), so N grids are shared across a batch of
    // position/geometry variations rather than 1:1.
    let n_grids = 200;
    let cases_per_grid = 15;
    let mut total = 0u32;
    for g in 0..n_grids {
        let mut grid_rng =
            Rng(0xbeef_cafe_0000_0001 ^ (g as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let grid = random_grid(&mut grid_rng);
        for _ in 0..cases_per_grid {
            let case = random_case(&mut rng);
            let (rust_out, c_out) = run_one(case, &grid);
            assert_eq!(rust_out, c_out, "case {total}: {case:?}");
            total += 1;
        }
    }
    eprintln!("find_dv_ref_mvs_matches_c: {total} cases x {n_grids} distinct grids byte/value-identical vs C");
}

#[test]
fn find_ref_dv_matches_c() {
    let mut rng = Rng(0x5eed_f14d_2ef0_0001);
    for _ in 0..50_000 {
        let mib_size = if rng.next().is_multiple_of(2) { 16 } else { 32 };
        let mi_row = rng.range(0, 512) as i32;
        let tile_mi_row_start = rng.range(0, (mi_row + 1).max(1) as i64) as i32;

        let rust_out = find_ref_dv(tile_mi_row_start, mib_size, mi_row);
        let c_out = c::ref_find_ref_dv(mi_row, mib_size, tile_mi_row_start);
        assert_eq!(
            rust_out, c_out,
            "mi_row={mi_row} mib_size={mib_size} tile_mi_row_start={tile_mi_row_start}"
        );
    }
}

#[test]
fn is_dv_valid_matches_c() {
    let mut rng = Rng(0x1234_5678_9abc_def0);
    for i in 0..100_000 {
        let bsize = rng.range(0, 22) as usize;
        let mi_row = rng.range(0, 256) as i32;
        let mi_col = rng.range(0, 256) as i32;
        let tile_mi_row_start = rng.range(0, (mi_row + 1) as i64) as i32;
        let tile_mi_row_end = rng.range((mi_row + 1) as i64, 300) as i32;
        let tile_mi_col_start = rng.range(0, (mi_col + 1) as i64) as i32;
        let tile_mi_col_end = rng.range((mi_col + 1) as i64, 300) as i32;
        let mib_size_log2 = if rng.next().is_multiple_of(2) { 4 } else { 5 };
        let is_chroma_ref = rng.next().is_multiple_of(2);
        let num_planes = if rng.next().is_multiple_of(3) { 1 } else { 3 };
        let ss_x = (rng.next() % 2) as i32;
        let ss_y = (rng.next() % 2) as i32;
        // DVs: mostly plausible small multiples of 8, occasionally
        // non-multiples (exercise the SCALE_PX_TO_MV reject) and occasionally
        // large (exercise the tile/SB-wavefront rejects) — but ALWAYS within
        // `int16_t` range, matching the real `MV.row`/`.col` representation
        // (the C shim's `av1_is_dv_valid` takes a `const MV dv` with
        // `int16_t` fields; feeding it a wider value than `int16_t` can hold
        // would silently truncate on the C side only, an apples-to-oranges
        // comparison against Rust's untruncated `i32` — not a real
        // algorithmic input any real caller could produce).
        let (dv_row, dv_col) = if i % 5 == 0 {
            (
                rng.range(-32000, 32000) as i32,
                rng.range(-32000, 32000) as i32,
            )
        } else {
            (
                (rng.range(-4000, 4000) * 8) as i32,
                (rng.range(-4000, 4000) * 8) as i32,
            )
        };

        let tile = DvTileBounds {
            mi_row_start: tile_mi_row_start,
            mi_row_end: tile_mi_row_end,
            mi_col_start: tile_mi_col_start,
            mi_col_end: tile_mi_col_end,
        };
        let rust_out = is_dv_valid(
            dv_row,
            dv_col,
            mi_row,
            mi_col,
            bsize,
            tile,
            mib_size_log2,
            is_chroma_ref,
            num_planes,
            ss_x,
            ss_y,
        );
        let c_out = c::ref_is_dv_valid(
            dv_row,
            dv_col,
            mi_row,
            mi_col,
            bsize,
            tile_mi_row_start,
            tile_mi_row_end,
            tile_mi_col_start,
            tile_mi_col_end,
            mib_size_log2,
            is_chroma_ref,
            num_planes,
            ss_x,
            ss_y,
        );
        assert_eq!(
            rust_out, c_out,
            "dv=({dv_row},{dv_col}) mi=({mi_row},{mi_col}) bsize={bsize} tile={tile:?} \
             mib_log2={mib_size_log2} chroma_ref={is_chroma_ref} planes={num_planes} ss=({ss_x},{ss_y})"
        );
    }
}

/// Composes the three verified C primitives into a manual oracle for
/// [`assign_and_validate_dv`] — cross-checks OUR wiring (dv_ref selection:
/// `nearestmv==0 ? nearmv : nearestmv`, then the `find_ref_dv` fallback,
/// truncation, `+ diff`, final validate), not the primitives themselves.
#[test]
fn assign_and_validate_dv_matches_composed_c() {
    let mut rng = Rng(0xfeed_a55e_1234_5678);
    let n_grids = 50;
    let cases_per_grid = 15;
    let mut total = 0u32;
    for g in 0..n_grids {
        let mut grid_rng =
            Rng(0xa11a_0000_0000_0001 ^ (g as u64).wrapping_mul(0x1234_5678_9abc_def1));
        let grid = random_grid(&mut grid_rng);
        for _ in 0..cases_per_grid {
            let case = random_case(&mut rng);
            let up_available = case.mi_row > case.tile.mi_row_start;
            let left_available = case.mi_col > case.tile.mi_col_start;

            let (c_nr, c_nc, c_rr, c_rc) = c::ref_find_dv_ref_mvs(
                case.mi_row,
                case.mi_col,
                case.bsize,
                case.own_partition,
                up_available,
                left_available,
                case.tile.mi_row_start,
                case.tile.mi_row_end,
                case.tile.mi_col_start,
                case.tile.mi_col_end,
                case.frame_mi_rows,
                case.frame_mi_cols,
                case.mib_size,
                &grid,
            );

            // A diff mostly in the plausible MV_SUBPEL_NONE-coded range
            // (multiples of 8, per `read_mv_component`'s structural
            // guarantee — see `assign_and_validate_dv`'s doc), occasionally
            // wild to exercise the `is_mv_valid` bound.
            let (diff_row, diff_col) = if rng.next().is_multiple_of(5) {
                (
                    rng.range(-40_000, 40_000) as i32,
                    rng.range(-40_000, 40_000) as i32,
                )
            } else {
                (
                    (rng.range(-2000, 2000) * 8) as i32,
                    (rng.range(-2000, 2000) * 8) as i32,
                )
            };

            let mib_size_log2 = if case.mib_size == 32 { 5 } else { 4 };
            let is_chroma_ref = rng.next().is_multiple_of(2);
            let num_planes = if rng.next().is_multiple_of(3) { 1 } else { 3 };
            let ss_x = (rng.next() % 2) as i32;
            let ss_y = (rng.next() % 2) as i32;

            // Manual oracle, composed from the three verified C primitives.
            let mut dv_ref = if (c_nr, c_nc) == (0, 0) {
                (c_rr, c_rc)
            } else {
                (c_nr, c_nc)
            };
            if dv_ref == (0, 0) {
                dv_ref = c::ref_find_ref_dv(case.mi_row, case.mib_size, case.tile.mi_row_start);
            }
            let valid_dv_ref = (dv_ref.1 & 7) == 0 && (dv_ref.0 & 7) == 0;
            dv_ref = ((dv_ref.0 >> 3) * 8, (dv_ref.1 >> 3) * 8);
            let mv_row = ((dv_ref.0 + diff_row) >> 3) * 8;
            let mv_col = ((dv_ref.1 + diff_col) >> 3) * 8;
            const MV_UPP: i32 = 1 << 14;
            const MV_LOW: i32 = -(1 << 14);
            let is_mv_valid =
                mv_row > MV_LOW && mv_row < MV_UPP && mv_col > MV_LOW && mv_col < MV_UPP;
            let expected = if valid_dv_ref
                && is_mv_valid
                && c::ref_is_dv_valid(
                    mv_row,
                    mv_col,
                    case.mi_row,
                    case.mi_col,
                    case.bsize,
                    case.tile.mi_row_start,
                    case.tile.mi_row_end,
                    case.tile.mi_col_start,
                    case.tile.mi_col_end,
                    mib_size_log2,
                    is_chroma_ref,
                    num_planes,
                    ss_x,
                    ss_y,
                ) {
                Some((mv_row, mv_col))
            } else {
                None
            };

            let actual = assign_and_validate_dv(
                (c_nr, c_nc),
                (c_rr, c_rc),
                diff_row,
                diff_col,
                case.tile.mi_row_start,
                case.mib_size,
                case.mi_row,
                case.mi_col,
                case.bsize,
                case.tile,
                mib_size_log2,
                is_chroma_ref,
                num_planes,
                ss_x,
                ss_y,
            );

            assert_eq!(
                actual, expected,
                "case {total}: nearest=({c_nr},{c_nc}) near=({c_rr},{c_rc}) diff=({diff_row},{diff_col}) {case:?}"
            );
            total += 1;
        }
    }
    eprintln!(
        "assign_and_validate_dv_matches_composed_c: {total} cases matched the composed-C oracle"
    );
}
