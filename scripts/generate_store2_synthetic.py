from __future__ import annotations

import argparse
import json
from pathlib import Path

from bixolon_scanner.configuration import load_json_config
from bixolon_scanner.training.store2_dataset import (
    Store2SyntheticRecipe,
    generate_store2_synthetic_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate approved store-2 composites")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    args = parser.parse_args()
    recipe = Store2SyntheticRecipe(**load_json_config(args.recipe))
    result = generate_store2_synthetic_dataset(
        args.source_root,
        args.prepared_root,
        args.output_root,
        recipe=recipe,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
