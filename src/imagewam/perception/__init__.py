"""Object-addr perception runtime (ported from the flux2b branch).

Offline cache building (training) and online perception (eval) share the SAME
pipeline: Qwen3 noun phrases -> SAM3 grounding -> DINOv3 masked-pool identity.
See online_addr.py for the eval entry point and identity.py for the exact
identity function (full-frame 518 resize + masked-average-pool, NOT crops).
"""
from . import identity, model_loaders, noun_phrase, sam3_grounding, online_addr  # noqa: F401
