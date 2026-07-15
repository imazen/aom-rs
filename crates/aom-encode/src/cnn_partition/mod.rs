//! Port of the speed>=1 **intra CNN partition prune**
//! (`av1/encoder/partition_strategy.c` `intra_mode_cnn_partition`) — the
//! learned model that, on an all-intra frame, prunes `PARTITION_SPLIT` and/or
//! the non-split partitions at each 64×64/32×32/16×16/8×8 block before the RD
//! partition search runs. Gated on `part_sf.intra_cnn_based_part_prune_level`
//! (0 at speed 0; `allow_screen_content_tools ? 0 : 2` at speed 1).
//!
//! Isolation (see `isolate_vgrad256_cq32_cnn_partition_prune`) proved this is
//! the single delta behind the last cpu-used=1 all-intra byte divergence
//! (`vgrad 256×256 cq32`): at qindex 128 the CNN forbids square-split on every
//! sub-block of SB(0,0), which the port's unpruned search must reproduce.
//!
//! Structure (built up in chunks, each diffed against the real C):
//! - [`nn`] — `av1_nn_predict` + `prec_reduce` (the branch DNN forward pass).
//! - (next) the CNN conv engine, weight tables, feature assembly + decision,
//!   then integration into `rd_pick_partition_real`.

pub mod nn;
