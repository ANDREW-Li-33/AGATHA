"""Tiny protobuf/TFRecord *encoder* for tests: builds fake Waymo v1 frames byte by byte."""

import struct


def varint(n: int) -> bytes:
    if n < 0:
        n += 1 << 64
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def f_varint(num: int, v: int) -> bytes:
    return varint(num << 3 | 0) + varint(v)


def f_double(num: int, v: float) -> bytes:
    return varint(num << 3 | 1) + struct.pack("<d", v)


def f_bytes(num: int, data: bytes) -> bytes:
    return varint(num << 3 | 2) + varint(len(data)) + data


def f_doubles_unpacked(num: int, values) -> bytes:
    return b"".join(f_double(num, v) for v in values)


def f_doubles_packed(num: int, values) -> bytes:
    return f_bytes(num, struct.pack(f"<{len(values)}d", *values))


def transform(T) -> bytes:
    return f_doubles_unpacked(1, [float(v) for v in T.reshape(-1)])


def map_point(x, y, z) -> bytes:
    return f_double(1, x) + f_double(2, y) + f_double(3, z)


def tfrecord(records) -> bytes:
    out = bytearray()
    for r in records:
        out += struct.pack("<Q", len(r)) + b"\0\0\0\0" + r + b"\0\0\0\0"  # CRCs not checked
    return bytes(out)
