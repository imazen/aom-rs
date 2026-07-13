/* Shim over cdef_filter_8_{0,1,2,3}_c. Oracle use only. */
#include <stdint.h>
void cdef_filter_8_0_c(void*,int,const uint16_t*,int,int,int,int,int,int,int,int);
void cdef_filter_8_1_c(void*,int,const uint16_t*,int,int,int,int,int,int,int,int);
void cdef_filter_8_2_c(void*,int,const uint16_t*,int,int,int,int,int,int,int,int);
void cdef_filter_8_3_c(void*,int,const uint16_t*,int,int,int,int,int,int,int,int);

void shim_cdef_filter8(int variant, uint8_t* dst, int dstride, const uint16_t* in,
                       int pri, int sec, int dir, int prid, int secd, int cshift,
                       int bw, int bh) {
  switch (variant) {
    case 0: cdef_filter_8_0_c(dst,dstride,in,pri,sec,dir,prid,secd,cshift,bw,bh); break;
    case 1: cdef_filter_8_1_c(dst,dstride,in,pri,sec,dir,prid,secd,cshift,bw,bh); break;
    case 2: cdef_filter_8_2_c(dst,dstride,in,pri,sec,dir,prid,secd,cshift,bw,bh); break;
    default: cdef_filter_8_3_c(dst,dstride,in,pri,sec,dir,prid,secd,cshift,bw,bh); break;
  }
}
