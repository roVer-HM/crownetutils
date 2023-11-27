_to_bps = {
    "bps": 1,
    "kbps": 1e3,
    "Mbps": 1e6,
    "Gbps": 1e9,
    "B/s": 8,
    "Bps": 8,
    "kB/s": 8e3,
    "kBps": 8e3,
    "MB/s": 8e6,
    "MBps": 8e6,
    "GB/s": 8e9,
    "GBps": 8e9,
}


def str_to_bps(sval: str) -> int:
    """parce bits per seconds based on string value with unit string"""
    val = None
    for _unit, _factor in _to_bps.items():
        if sval.endswith(_unit):
            val = float(sval[: -len(_unit)]) * _factor
            break
    if val is None:
        raise ValueError(
            f"unit not found. Got unit '{sval}'. Supported units are: {_to_bps.keys()}"
        )

    return int(val)
