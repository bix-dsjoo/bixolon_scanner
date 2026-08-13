from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args_with_config(parser: argparse.ArgumentParser, *, section: str) -> argparse.Namespace:
    parser.add_argument("--config", type=Path, help="Version-controlled JSON defaults")
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path)
    known, _ = preliminary.parse_known_args()
    if known.config is not None:
        raw = json.loads(known.config.read_text(encoding="utf-8"))
        defaults = raw.get(section)
        if not isinstance(defaults, dict):
            raise ValueError(f"missing configuration section: {section}")
        valid = {action.dest for action in parser._actions}
        unexpected = sorted(set(defaults) - valid)
        if unexpected:
            raise ValueError(f"unsupported {section} configuration keys: {unexpected}")
        parser.set_defaults(**defaults)
    return parser.parse_args()
