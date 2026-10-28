"""
Minimal message format between the pi0.5 server (Python 3.12, numpy 2) and the LIBERO-PRO clients (Python 3.8,
numpy 1.22): an 8-byte big-endian header length, a JSON header (fields + array specs), then the arrays' raw bytes.
No pickle, so the two numpy versions never need to read each other's objects.
"""
import json
import socket
import struct

import numpy as np


def send_msg(sock: socket.socket, header: dict, arrays: dict = None) -> None:
    arrays = arrays or {}
    specs, blobs = [], []
    for name, a in arrays.items():
        a = np.ascontiguousarray(a)
        specs.append([name, a.dtype.str, list(a.shape)])
        blobs.append(a.tobytes())
    head = json.dumps(dict(header, _arrays=specs)).encode()
    sock.sendall(struct.pack(">Q", len(head)) + head + b"".join(blobs))


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 22))
        if not chunk:
            raise ConnectionError("socket closed")
        buf += chunk
    return bytes(buf)


def recv_msg(sock: socket.socket):
    (n,) = struct.unpack(">Q", _recv_exact(sock, 8))
    header = json.loads(_recv_exact(sock, n))
    arrays = {}
    for name, dtype, shape in header.pop("_arrays"):
        dt = np.dtype(dtype)
        size = int(np.prod(shape)) * dt.itemsize
        arrays[name] = np.frombuffer(_recv_exact(sock, size), dtype=dt).reshape(shape).copy()
    return header, arrays
