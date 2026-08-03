"""Client for the shared per-GPU Mage text-encoder server.

Used by eval workers when `MAGE_ENCODER_SOCKET` is set: instead of loading the
8.3GB Mage text encoder per worker, the worker asks the shared server (one per
GPU) to encode (instruction, current frame) into the text context.

The server runs the identical `encode_edit_conditions` the model would run
online, so the returned (context, mask) is bit-identical to local encoding.
"""
from __future__ import annotations

import pickle
import socket
import struct

import torch


def _recvall(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("encoder server closed the connection")
        buf.extend(chunk)
    return bytes(buf)


def _recv_msg(sock):
    (length,) = struct.unpack("<Q", _recvall(sock, 8))
    return pickle.loads(_recvall(sock, length))


def _send_msg(sock, obj):
    payload = pickle.dumps(obj)
    sock.sendall(struct.pack("<Q", len(payload)) + payload)


def ping(socket_path: str, timeout: float = 5.0) -> bool:
    """Return True if the encoder server at `socket_path` answers a ping."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(socket_path)
        _send_msg(s, {"op": "ping"})
        return bool(_recv_msg(s).get("ok"))
    finally:
        s.close()


def encode_context(socket_path: str, instruction: str, image: torch.Tensor,
                   timeout: float = 120.0):
    """Ask the shared encoder server to encode (instruction, image).

    Args:
        socket_path: unix socket of this GPU's encoder server.
        instruction: the prompt string the model would encode (DEFAULT_PROMPT-formatted).
        image: Tensor[B,3,H,W] in any value range; the server denormalizes exactly
            like the model's `_prepare_mage_infer_context`.
    Returns:
        (context[B,L,D], mask[B,L]) as CPU tensors.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"`image` must be a torch.Tensor, got {type(image)}")
    import time as _time
    last_err = None
    for _attempt in range(10):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout)
        try:
            s.connect(socket_path)
            _send_msg(s, {"op": "encode", "instruction": str(instruction),
                          "image": image.detach().to("cpu")})
            resp = _recv_msg(s)
            s.close()
            break
        except (BlockingIOError, ConnectionRefusedError, FileNotFoundError) as exc:
            s.close()
            last_err = exc
            _time.sleep(0.2 * (_attempt + 1))  # backoff: 0.2, 0.4, ..., 2.0s
    else:
        raise ConnectionError(
            f"failed to connect to encoder server at {socket_path} after 10 retries: {last_err}")

    if "error" in resp:
        raise RuntimeError(f"mage encoder server error: {resp['error']}")
    return resp["context"], resp["mask"]
