"""Requantize selected S2 Pro tensors from a GGUF (e.g. q4_k_m) to Q4_0 with
MSE-searched scales, keeping every other tensor byte-for-byte.

Targets: slow/fast `attention.wo`, `feed_forward.w1`, `feed_forward.w3`.
Everything else (wqkv, w2, embeddings, codec, norms) is copied as-is.

Usage:
    python tools/quantize_q4m.py --src q4_k_m.gguf --out s2-q4m.gguf
"""
import argparse
import re
import struct
from pathlib import Path

import numpy as np
import gguf
from gguf.quants import dequantize

Q4_0 = int(gguf.GGMLQuantizationType.Q4_0)
TARGET_RE = re.compile(
    r"^(fast_)?layers\.\d+\.(attention\.wo|feed_forward\.w1|feed_forward\.w3)\.weight$")


def read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8")


def read_value(f, vtype):
    if vtype in (0, 1):
        return struct.unpack("<B", f.read(1))[0]
    if vtype in (2, 3):
        return struct.unpack("<H", f.read(2))[0]
    if vtype in (4, 5):
        return struct.unpack("<I", f.read(4))[0]
    if vtype == 6:
        return struct.unpack("<f", f.read(4))[0]
    if vtype == 7:
        return struct.unpack("<?", f.read(1))[0]
    if vtype in (10, 11):
        return struct.unpack("<Q", f.read(8))[0]
    if vtype in (12,):
        return struct.unpack("<d", f.read(8))[0]
    if vtype == 8:
        return read_str(f)
    if vtype == 9:
        (sub,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        return [read_value(f, sub) for _ in range(n)]
    raise ValueError(f"unknown gguf value type {vtype}")


def skip_value(f, vtype):
    if vtype in (0, 1):
        f.read(1)
    elif vtype in (2, 3):
        f.read(2)
    elif vtype in (4, 5):
        f.read(4)
    elif vtype == 6:
        f.read(4)
    elif vtype == 7:
        f.read(1)
    elif vtype in (10, 11, 12):
        f.read(8)
    elif vtype == 8:
        read_str(f)
    elif vtype == 9:
        (sub,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            skip_value(f, sub)
    else:
        raise ValueError(f"unknown gguf value type {vtype}")


def nbytes_and_nelem(dims, qtype):
    n = 1
    for d in dims:
        n *= d
    block, size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(qtype)]
    return (n + block - 1) // block * size, n


def quantize_q4_0_mse(x):
    x = np.asarray(x, dtype=np.float32)
    assert x.size % 32 == 0, x.size
    b = x.reshape(-1, 32)
    absmax = np.max(np.abs(b), axis=1)

    best_err = np.full(b.shape[0], np.inf, dtype=np.float32)
    best_q = np.zeros((b.shape[0], 32), dtype=np.uint8)
    best_d = np.zeros(b.shape[0], dtype=np.float16)

    # 32 candidate clipping thresholds between 0.6 and 1.0 of absmax
    for c in range(32):
        clip = absmax * (0.6 + 0.4 * c / 31.0)
        d = (clip / 8.0).astype(np.float16)
        d32 = np.maximum(d.astype(np.float32), 1e-12)
        q = np.clip(np.rint(b / d32[:, None]) + 8.0, 0, 15).astype(np.uint8)
        rec = (q.astype(np.float32) - 8.0) * d32[:, None]
        err = np.mean((rec - b) ** 2, axis=1)
        better = err < best_err
        if np.any(better):
            best_err[better] = err[better]
            best_q[better] = q[better]
            best_d[better] = d[better]

    qs = best_q[:, :16] | (best_q[:, 16:] << 4)
    out = np.empty((b.shape[0], 18), dtype=np.uint8)
    out[:, :2] = best_d.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = qs
    return out.reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)

    with open(src, "rb") as f:
        magic, version, n_tensors, n_kv = struct.unpack("<4sIQQ", f.read(24))
        assert magic == b"GGUF", magic

        kv_start = f.tell()
        align = 32
        for _ in range(n_kv):
            key = read_str(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            if key == "general.alignment" and vtype == 4:
                align = read_value(f, vtype)
            else:
                skip_value(f, vtype)
        kv_end = f.tell()
        f.seek(kv_start)
        kv_bytes = f.read(kv_end - kv_start)

        tensors = []
        for _ in range(n_tensors):
            name = read_str(f)
            (nd,) = struct.unpack("<I", f.read(4))
            dims = tuple(struct.unpack("<" + "Q" * nd, f.read(8 * nd)))
            (ttype,) = struct.unpack("<I", f.read(4))
            (offset,) = struct.unpack("<Q", f.read(8))
            tensors.append((name, dims, ttype, offset))

        data_start = (f.tell() + align - 1) // align * align

        layout = []
        cur = 0
        converted = 0
        for name, dims, ttype, offset in tensors:
            nb_in, n = nbytes_and_nelem(dims, ttype)
            if TARGET_RE.match(name):
                nb_out = (n // 32) * 18
                new_type = Q4_0
                converted += 1
            else:
                nb_out = nb_in
                new_type = ttype
            layout.append((name, dims, new_type, ttype, offset, nb_in, nb_out, cur))
            cur = (cur + nb_out + align - 1) // align * align

    with open(src, "rb") as fin, open(out, "wb") as fout:
        fout.write(struct.pack("<4sIQQ", b"GGUF", version, n_tensors, n_kv))
        fout.write(kv_bytes)

        for name, dims, new_type, _, _, _, _, off in layout:
            b = name.encode("utf-8")
            fout.write(struct.pack("<Q", len(b)))
            fout.write(b)
            fout.write(struct.pack("<I", len(dims)))
            fout.write(struct.pack("<" + "Q" * len(dims), *dims))
            fout.write(struct.pack("<I", new_type))
            fout.write(struct.pack("<Q", off))

        fout.write(b"\x00" * ((-fout.tell()) % align))
        data_out_start = fout.tell()

        for name, dims, new_type, src_type, src_off, nb_in, nb_out, off in layout:
            fout.seek(data_out_start + off)
            fin.seek(data_start + src_off)
            raw = fin.read(nb_in)
            if TARGET_RE.match(name):
                x = dequantize(np.frombuffer(raw, dtype=np.uint8),
                               gguf.GGMLQuantizationType(src_type))
                fout.write(quantize_q4_0_mse(x).tobytes())
            else:
                fout.write(raw)

    print(f"converted {converted}/{n_tensors} tensors -> {out} ({out.stat().st_size/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
