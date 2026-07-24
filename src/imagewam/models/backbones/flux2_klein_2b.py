import re


DEFAULT_DOUBLE_INDICES = (0, 2, 4)
DEFAULT_SINGLE_INDICES = (0, 2, 4, 6, 9, 11, 13, 15, 17, 19)


def estimate_flux2_params(hidden_size: int, double_layers: int, single_layers: int) -> int:
    """Estimate the full FLUX.2 parameter count for a shallow Klein model."""
    # Block parameters include the two RMSNorm scales in each attention module.
    double = double_layers * (26 * hidden_size**2 + 4 * (hidden_size // 24))
    single = single_layers * (13 * hidden_size**2 + 2 * (hidden_size // 24))
    non_block = (
        128 * hidden_size
        + 256 * hidden_size
        + hidden_size * hidden_size
        + 7680 * hidden_size
        + 2 * 6 * hidden_size * hidden_size
        + 3 * hidden_size * hidden_size
        + 2 * hidden_size * hidden_size
        + 128 * hidden_size
    )
    return double + single + non_block


_BLOCK_KEY = re.compile(r"^(double_blocks|single_blocks)\.(\d+)(\..+)$")


def remap_block_key(key: str, target_index: int) -> str:
    match = _BLOCK_KEY.match(key)
    if match is None:
        raise ValueError(f"Not a FLUX.2 block key: {key}")
    return f"{match.group(1)}.{target_index}{match.group(3)}"


def interpolation_sources(source_depth: int, target_depth: int) -> tuple[tuple[int, int, float], ...]:
    """Return neighboring source layers and alpha for center-aligned interpolation."""
    result = []
    for target_index in range(target_depth):
        position = (target_index + 0.5) * source_depth / target_depth - 0.5
        position = max(0.0, min(float(source_depth - 1), position))
        left = int(position)
        right = min(left + 1, source_depth - 1)
        result.append((left, right, position - left))
    return tuple(result)
