from __future__ import annotations

import argparse
from pathlib import Path

from ..configuration import load_json_config, resolve_config_path


def parse_args_with_config(parser: argparse.ArgumentParser, *, section: str) -> argparse.Namespace:
    parser.add_argument("--config", type=Path, help="Version-controlled JSON defaults")
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path)
    known, _ = preliminary.parse_known_args()
    if known.config is not None:
        known.config = resolve_config_path(known.config)
        raw = load_json_config(known.config)
        defaults = raw.get(section)
        if not isinstance(defaults, dict):
            raise ValueError(f"missing configuration section: {section}")
        valid = {action.dest for action in parser._actions}
        unexpected = sorted(set(defaults) - valid)
        if unexpected:
            raise ValueError(f"unsupported {section} configuration keys: {unexpected}")
        parser.set_defaults(**defaults)
    parsed = parser.parse_args()
    if parsed.config is not None:
        parsed.config = resolve_config_path(parsed.config)
    return parsed
