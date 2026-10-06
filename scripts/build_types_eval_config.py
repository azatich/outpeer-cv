"""Create a separate evaluation configuration for a named type-model run."""

import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--name", default="taco_n_types_30")
parser.add_argument("--baseline", action="append", default=[], help="Compare an earlier run on the same frozen validation split; repeat for multiple runs")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", args.name):
    parser.error("name must be a simple run name")
if any(not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", name) or name == args.name for name in args.baseline) or len(set(args.baseline)) != len(args.baseline):
    parser.error("baseline must be a different simple run name")
config = yaml.safe_load((ROOT / "configs/multiclass/evaluation.yaml").read_text())
config["models"] = [{"name": args.name, "weights": f"models/b/{args.name}/best.pt"}]
if args.baseline:
    config["models"][0:0] = [{"name": name, "weights": f"models/b/{name}/best.pt"} for name in args.baseline]
output = args.output if args.output.is_absolute() else ROOT / args.output
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("x", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
