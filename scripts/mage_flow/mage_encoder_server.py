#!/usr/bin/env python3
"""Per-GPU shared Mage text-encoder server for eval.

Loads ONLY the Mage text encoder (Qwen3-VL, ~8.3GB) + processor, then listens
on a unix socket and serves `encode_edit_conditions` to eval workers running on
the same GPU. Workers can thus share one encoder instead of each loading their
own copy, cutting per-worker GPU memory so more workers fit per card.

The encode performed here is identical to what the model's
`_prepare_mage_infer_context` does online (same image denormalization, same
PIL conversion, same `encode_edit_conditions` call) -> the returned context is
bit-identical to local online encoding. Zero accuracy change.

Protocol (length-prefixed pickle over a unix stream socket):
    request {"op": "ping"}                          -> {"ok": True}
    request {"op": "encode", "instruction": str,
             "image": Tensor[B,3,H,W]}              -> {"context": Tensor[B,L,D],
                                                       "mask": Tensor[B,L]}
    on error                                         -> {"error": str}
"""
from __future__ import annotations

import argparse
import os
import pickle
import socketserver
import struct

import torch
from PIL import Image

from imagewam.models.backbones.mage_flow_video_expert import MageFlowVideoExpert


def _recvall(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed by peer")
        buf.extend(chunk)
    return bytes(buf)


def _recv_msg(sock):
    (length,) = struct.unpack("<Q", _recvall(sock, 8))
    return pickle.loads(_recvall(sock, length))


def _send_msg(sock, obj):
    payload = pickle.dumps(obj)
    sock.sendall(struct.pack("<Q", len(payload)) + payload)


class _State:
    """Holds the loaded encoder expert. Populated once at startup."""
    expert: MageFlowVideoExpert | None = None


def _encode(instruction: str, image: torch.Tensor):
    """Replicate `_prepare_mage_infer_context`'s encode path exactly."""
    expert = _State.expert
    refs = image.detach().float().cpu()
    if float(refs.min()) < 0:
        refs = (refs + 1.0) * 0.5
    refs = refs.clamp(0, 1)
    pil_refs = [
        Image.fromarray((x.permute(1, 2, 0).numpy() * 255).astype("uint8"))
        for x in refs
    ]
    context, mask = expert.encode_edit_conditions(
        [instruction] * refs.shape[0],
        [[im] for im in pil_refs],
        device=expert.device,
    )
    return context.detach().cpu(), mask.detach().cpu()


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            req = _recv_msg(self.request)
        except Exception as exc:  # malformed / closed
            return
        op = req.get("op")
        if op == "ping":
            _send_msg(self.request, {"ok": True})
        elif op == "encode":
            try:
                ctx, msk = _encode(req["instruction"], req["image"])
                _send_msg(self.request, {"context": ctx, "mask": msk})
            except Exception as exc:
                _send_msg(self.request, {"error": f"{type(exc).__name__}: {exc}"})
        else:
            _send_msg(self.request, {"error": f"unknown op: {op!r}"})


class _UnixServer(socketserver.UnixStreamServer):
    """Serial (single-threaded) server: one request handled at a time."""
    allow_reuse_address = True
    request_queue_size = 64  # large backlog so concurrent workers don't get EAGAIN on connect


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True, help="Mage-Flow backbone repo/dir")
    p.add_argument(
        "--mage-flow-src-path",
        default=os.environ.get("IMAGEWAM_MAGE_FLOW_SRC_PATH", "third_party/Mage"),
    )
    p.add_argument("--socket", required=True, help="unix socket path to listen on")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    torch_dtype = getattr(torch, args.dtype)
    print(f"[mage-encoder-server] loading text encoder from {args.model_path} "
          f"on {args.device} ({args.dtype})...", flush=True)
    _State.expert = MageFlowVideoExpert.from_pretrained(
        args.model_path,
        args.mage_flow_src_path,
        device=args.device,
        torch_dtype=torch_dtype,
        load_text_encoder=True,
        text_encoder_only=True,
    )
    _State.expert.eval()
    print("[mage-encoder-server] text encoder ready.", flush=True)

    if os.path.exists(args.socket):
        os.unlink(args.socket)
    server = _UnixServer(args.socket, _Handler)
    print(f"[mage-encoder-server] listening on {args.socket}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if os.path.exists(args.socket):
            os.unlink(args.socket)


if __name__ == "__main__":
    main()
