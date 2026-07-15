// Reference oracle for the FORWARD quantization-matrix *selector*
// (av1_get_qmatrix / av1_qm_init, av1/common/quant_common.c). The QM bases
// `wt_matrix_ref` are file-static (not exported), but av1_qm_init packs pointers
// to them into `quant_params->gqmatrix[q][c][t]`. Reading those back yields the
// genuine C bytes for a (level, plane group, tx size) cell, validating the Rust
// `aom_quant::qmatrix` selector's table + packing offsets + plane/level indexing
// + 64-point aliasing against the REAL init loop, not a transcription.
#include <stdint.h>
#include "av1/common/blockd.h"          /* av1_get_adjusted_tx_size, TX_SIZE */
#include "av1/common/common_data.h"     /* tx_size_2d */
#include "av1/common/av1_common_int.h"  /* CommonQuantParams (+ gqmatrix) */
#include "av1/common/quant_common.h"    /* av1_qm_init, NUM_QM_LEVELS, qm_val_t */

// Copy the forward QM matrix bytes for (level q, plane group c, tx size t) into
// `out` (capacity out_cap). Returns the matrix length == tx_size_2d[adjusted(t)],
// or -1 if the cell is the flat NULL matrix (q == NUM_QM_LEVELS-1), or -2 if the
// length would overflow out_cap.
int shim_qm_gqmatrix(int q, int c, int t, uint8_t *out, int out_cap) {
  static CommonQuantParams qp;  // ~large; static avoids a big stack frame
  av1_qm_init(&qp, 3);          // populate all 3 plane groups (c in {0,1,2})
  const qm_val_t *m = qp.gqmatrix[q][c][t];
  if (m == NULL) return -1;
  int len = tx_size_2d[av1_get_adjusted_tx_size((TX_SIZE)t)];
  if (len > out_cap) return -2;
  for (int i = 0; i < len; ++i) out[i] = (uint8_t)m[i];
  return len;
}
