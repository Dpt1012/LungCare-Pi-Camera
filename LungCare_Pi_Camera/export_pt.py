"""Export YOUR CNN state_dict to the tensor-only TorchScript contract."""
import argparse
import importlib
import json
from pathlib import Path
import sys

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--factory", required=True, help="module:ClassOrFactory, e.g. my_network:UNet")
    p.add_argument("--kwargs", default="{}", help="Actual architecture constructor arguments as JSON")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("model_scripted.pt"))
    args = p.parse_args()
    sys.path.insert(0, str(Path.cwd()))
    module, name = args.factory.split(":", 1)
    model = getattr(importlib.import_module(module), name)(**json.loads(args.kwargs))
    # A saved whole Python model is intentionally not unpickled here.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint
    if isinstance(checkpoint, dict):
        state = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    model.load_state_dict(state, strict=True)
    model.cpu().eval()
    c = json.loads(args.config.read_text(encoding="utf-8"))
    shape = (1, c["channels"], c["height"], c["width"])
    with torch.inference_mode():
        example = model(torch.zeros(shape))
        if not isinstance(example, torch.Tensor) or example.ndim != 4 or tuple(example.shape[:2]) != (1, 1):
            raise ValueError("Need tensor [1,1,H,W]. Wrap dict/tuple/multiclass models before export.")
        # Scripting preserves control flow; fail explicitly if architecture needs adaptation.
        scripted = torch.jit.script(model)
        for x in (torch.zeros(shape), torch.rand(shape)):
            torch.testing.assert_close(scripted(x), model(x))
        torch.jit.save(scripted, str(args.out))
    print(f"Saved {args.out}. Check preprocessing against training before inference.")


if __name__ == "__main__":
    main()
