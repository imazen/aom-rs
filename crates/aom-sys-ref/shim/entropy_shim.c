/* Thin ABI-stable shim over libaom's od_ec_enc / od_ec_dec so Rust need not
 * mirror the C struct layout. Oracle use only. */
#include <stdlib.h>
#include <stdint.h>
#include "aom_dsp/entenc.h"
#include "aom_dsp/entdec.h"

void *shim_enc_new(uint32_t size) {
  od_ec_enc *e = (od_ec_enc *)malloc(sizeof(od_ec_enc));
  od_ec_enc_init(e, size);
  return e;
}
void shim_enc_bool(void *e, int val, unsigned f) {
  od_ec_encode_bool_q15((od_ec_enc *)e, val, f);
}
void shim_enc_cdf(void *e, int s, const uint16_t *icdf, int nsyms) {
  od_ec_encode_cdf_q15((od_ec_enc *)e, s, icdf, nsyms);
}
const unsigned char *shim_enc_done(void *e, uint32_t *nbytes) {
  return od_ec_enc_done((od_ec_enc *)e, nbytes);
}
void shim_enc_free(void *e) {
  od_ec_enc_clear((od_ec_enc *)e);
  free(e);
}

void *shim_dec_new(const unsigned char *buf, uint32_t sz) {
  od_ec_dec *d = (od_ec_dec *)malloc(sizeof(od_ec_dec));
  od_ec_dec_init(d, buf, sz);
  return d;
}
int shim_dec_bool(void *d, unsigned f) {
  return od_ec_decode_bool_q15((od_ec_dec *)d, f);
}
int shim_dec_cdf(void *d, const uint16_t *icdf, int nsyms) {
  return od_ec_decode_cdf_q15((od_ec_dec *)d, icdf, nsyms);
}
void shim_dec_free(void *d) { free(d); }
