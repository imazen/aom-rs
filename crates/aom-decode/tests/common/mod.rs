//! Shared test helpers. Sub-directory of `tests/`, so it is NOT compiled as its
//! own test binary — only pulled in via `mod common;` from other test files.

/// Self-contained streaming MD5 (RFC 1321) for golden-hash decode anchors. The
/// same construction the conformance gate uses to reproduce libaom's shipped
/// per-frame hashes.
pub mod md5 {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    const K: [u32; 64] = [
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613,
        0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193,
        0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d,
        0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122,
        0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
        0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244,
        0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
        0xeb86d391,
    ];

    pub struct Md5 {
        a: [u32; 4],
        buf: [u8; 64],
        buf_len: usize,
        total: u64,
    }
    impl Default for Md5 {
        fn default() -> Self {
            Self::new()
        }
    }
    impl Md5 {
        pub fn new() -> Self {
            Md5 {
                a: [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476],
                buf: [0; 64],
                buf_len: 0,
                total: 0,
            }
        }
        pub fn update(&mut self, mut data: &[u8]) {
            self.total = self.total.wrapping_add(data.len() as u64);
            if self.buf_len > 0 {
                let need = 64 - self.buf_len;
                let take = need.min(data.len());
                self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
                self.buf_len += take;
                data = &data[take..];
                if self.buf_len == 64 {
                    let block = self.buf;
                    self.process(&block);
                    self.buf_len = 0;
                }
            }
            while data.len() >= 64 {
                let mut block = [0u8; 64];
                block.copy_from_slice(&data[..64]);
                self.process(&block);
                data = &data[64..];
            }
            if !data.is_empty() {
                self.buf[..data.len()].copy_from_slice(data);
                self.buf_len = data.len();
            }
        }
        pub fn finish(mut self) -> String {
            let bitlen = self.total.wrapping_mul(8);
            let mut pad = vec![0x80u8];
            while (self.total.wrapping_add(pad.len() as u64)) % 64 != 56 {
                pad.push(0);
            }
            pad.extend_from_slice(&bitlen.to_le_bytes());
            self.update(&pad);
            debug_assert_eq!(self.buf_len, 0);
            let mut out = String::with_capacity(32);
            for word in self.a {
                for byte in word.to_le_bytes() {
                    out.push_str(&format!("{byte:02x}"));
                }
            }
            out
        }
        fn process(&mut self, chunk: &[u8; 64]) {
            let mut m = [0u32; 16];
            for i in 0..16 {
                m[i] = u32::from_le_bytes([
                    chunk[i * 4],
                    chunk[i * 4 + 1],
                    chunk[i * 4 + 2],
                    chunk[i * 4 + 3],
                ]);
            }
            let [mut a, mut b, mut c, mut d] = self.a;
            for i in 0..64 {
                let (f, g) = match i {
                    0..=15 => ((b & c) | (!b & d), i),
                    16..=31 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                    32..=47 => (b ^ c ^ d, (3 * i + 5) % 16),
                    _ => (c ^ (b | !d), (7 * i) % 16),
                };
                let f = f.wrapping_add(a).wrapping_add(K[i]).wrapping_add(m[g]);
                a = d;
                d = c;
                c = b;
                b = b.wrapping_add(f.rotate_left(S[i]));
            }
            self.a[0] = self.a[0].wrapping_add(a);
            self.a[1] = self.a[1].wrapping_add(b);
            self.a[2] = self.a[2].wrapping_add(c);
            self.a[3] = self.a[3].wrapping_add(d);
        }
    }

    /// MD5 of `data` as a 32-char lowercase hex string.
    pub fn hex(data: &[u8]) -> String {
        let mut m = Md5::new();
        m.update(data);
        m.finish()
    }

    /// MD5 over decoded planes in libaom `md5_helper.h::Add(aom_image_t*)` order:
    /// each plane's cropped rows, low byte then high byte per sample (the
    /// little-endian `u16` layout the conformance goldens hash).
    pub fn planes_hex(planes: &[&[u16]]) -> String {
        let mut m = Md5::new();
        let mut row = Vec::new();
        for p in planes {
            row.clear();
            for &s in *p {
                row.push((s & 0xff) as u8);
                row.push((s >> 8) as u8);
            }
            m.update(&row);
        }
        m.finish()
    }

    #[test]
    fn md5_known_vectors() {
        assert_eq!(hex(b""), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(hex(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
    }
}
