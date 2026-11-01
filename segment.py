"""2D research segmentation on Raspberry Pi 4. No trained weights included."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time

import cv2
import numpy as np


def read_config(path):
    c = json.loads(Path(path).read_text(encoding="utf-8"))
    if c["engine"] not in ("torchscript", "onnx"):
        raise ValueError("engine must be torchscript or onnx")
    if c["source_kind"] not in ("photo", "dicom"):
        raise ValueError("source_kind must be photo or dicom")
    if c["channels"] not in (1, 3) or c["resize"] != "stretch":
        raise ValueError("Only 1/3 channels and stretch resizing are supported")
    for key in ("height", "width", "threads"):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f"Invalid {key}")
    if not 0 < c["threshold"] < 1 or c["output_kind"] not in ("logits", "probabilities"):
        raise ValueError("Invalid threshold/output_kind")
    for key in ("mean", "std"):
        if len(c[key]) != c["channels"] or not np.isfinite(c[key]).all():
            raise ValueError(f"{key} must have one finite value per channel")
    if min(c["std"]) <= 0:
        raise ValueError("std must be positive")
    if not math.isfinite(c["window_center"]) or not math.isfinite(c["window_width"]) or c["window_width"] <= 0:
        raise ValueError("Invalid CT window")
    return c


def load_image(path, c):
    """Return RGB float32 [0,1], optional mm/pixel, and validity mask."""
    if c["source_kind"] == "photo":
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.uint8:
            raise ValueError("Photo must be a readable 8-bit PNG/JPEG/BMP")
        if raw.ndim == 2:
            rgb = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
        elif raw.ndim == 3 and raw.shape[2] == 3:
            rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Unsupported photo channels; remove alpha explicitly")
        return rgb.astype(np.float32) / 255, None, np.ones(rgb.shape[:2], bool)

    import pydicom
    ds = pydicom.dcmread(str(path))
    if ds.get("Modality") != "CT" or int(ds.get("NumberOfFrames", 1)) != 1:
        raise ValueError("Only classic single-frame CT DICOM is supported")
    if "LOCALIZER" in "\\".join(ds.get("ImageType", [])).upper():
        raise ValueError("Localizer is not a CT slice")
    if ds.get("PhotometricInterpretation") not in ("MONOCHROME1", "MONOCHROME2"):
        raise ValueError("Only monochrome CT is supported")
    if "RescaleSlope" not in ds or "RescaleIntercept" not in ds or "ModalityLUTSequence" in ds:
        raise ValueError("Need explicit linear CT rescale slope/intercept")
    if str(ds.get("RescaleType", "HU")).upper() != "HU":
        raise ValueError("CT rescale units must be HU")
    if "RealWorldValueMappingSequence" in ds:
        raise ValueError("Real-world value mapping is unsupported")
    slope, intercept = float(ds.RescaleSlope), float(ds.RescaleIntercept)
    if not np.isfinite([slope, intercept]).all() or slope <= 0:
        raise ValueError("Invalid CT rescale parameters")
    raw = ds.pixel_array
    if raw.ndim != 2:
        raise ValueError("Expected one 2D CT slice")
    valid = np.ones(raw.shape, bool)
    if "PixelPaddingValue" in ds:
        a = float(ds.PixelPaddingValue)
        b = float(ds.get("PixelPaddingRangeLimit", a))
        valid &= ~((raw >= min(a, b)) & (raw <= max(a, b)))
    hu = raw.astype(np.float32) * slope + intercept
    # Window definition is deliberately explicit and must match model training.
    low = c["window_center"] - c["window_width"] / 2
    gray = np.clip((hu - low) / c["window_width"], 0, 1)
    gray[~valid] = 0
    spacing = None
    if "PixelSpacing" in ds:
        spacing = [float(x) for x in ds.PixelSpacing]
        if len(spacing) != 2 or not np.isfinite(spacing).all() or min(spacing) <= 0:
            raise ValueError("Invalid PixelSpacing")
    # HU processing ignores MONOCHROME1 display inversion by design.
    return np.repeat(gray[:, :, None], 3, axis=2), spacing, valid


def prepare(rgb, c):
    small = cv2.resize(rgb, (c["width"], c["height"]), interpolation=cv2.INTER_LINEAR)
    if c["channels"] == 1:
        small = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)[:, :, None]
    mean = np.asarray(c["mean"], np.float32)
    std = np.asarray(c["std"], np.float32)
    return np.ascontiguousarray(((small - mean) / std).transpose(2, 0, 1)[None])


class Predictor:
    def __init__(self, model, c):
        self.c = c
        if c["engine"] == "torchscript":
            import torch
            self.torch = torch
            torch.set_num_threads(c["threads"])
            self.net = torch.jit.load(str(model), map_location="cpu").eval()
        else:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = c["threads"]
            options.inter_op_num_threads = 1
            self.net = ort.InferenceSession(str(model), sess_options=options,
                                            providers=["CPUExecutionProvider"])
            if len(self.net.get_inputs()) != 1 or len(self.net.get_outputs()) != 1:
                raise ValueError("ONNX must have exactly one input and one output")

    def predict(self, x):
        if self.c["engine"] == "torchscript":
            with self.torch.inference_mode():
                y = self.net(self.torch.from_numpy(x))
            if not isinstance(y, self.torch.Tensor):
                raise ValueError("Model must return one tensor; export a wrapper for dict/tuple output")
            y = y.detach().cpu().numpy()
        else:
            y = self.net.run(None, {self.net.get_inputs()[0].name: x})[0]
        return decode(y, self.c)


def decode(y, c):
    if y.ndim != 4 or y.shape[:2] != (1, 1) or min(y.shape[2:]) < 1:
        raise ValueError(f"Expected binary segmentation [1,1,H,W], got {y.shape}")
    if not np.isfinite(y).all():
        raise ValueError("Model returned NaN/Inf")
    y = y[0, 0].astype(np.float32)
    if c["output_kind"] == "logits":
        y = 1 / (1 + np.exp(-np.clip(y, -80, 80)))
    elif y.min() < 0 or y.max() > 1:
        raise ValueError("Probability output outside [0,1]; check output_kind")
    return y


def run_one(path, out, c, predictor, roi=None):
    rgb, spacing, valid = load_image(path, c)
    h, w = rgb.shape[:2]
    x, y, rw, rh = roi or (0, 0, w, h)
    if min(x, y) < 0 or min(rw, rh) < 1 or x + rw > w or y + rh > h:
        raise ValueError("ROI lies outside image")
    t0 = time.perf_counter()
    p = predictor.predict(prepare(rgb[y:y+rh, x:x+rw], c))
    p = cv2.resize(p, (rw, rh), interpolation=cv2.INTER_LINEAR)
    probability = np.zeros((h, w), np.float32)
    probability[y:y+rh, x:x+rw] = p
    probability[~valid] = 0
    mask = (probability >= c["threshold"]).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    regions = []
    for label in range(1, n):
        sx, sy, sw, sh, area = [int(v) for v in stats[label]]
        region = {"bbox_xywh_px": [sx, sy, sw, sh], "area_px": area}
        if spacing is not None:
            region["area_mm2"] = area * spacing[0] * spacing[1]
        regions.append(region)
    display = np.rint(rgb * 255).astype(np.uint8)
    display[mask > 0] = (0.55 * display[mask > 0] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
    overlay = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
    cv2.putText(overlay, "RESEARCH ONLY - NOT A DIAGNOSIS", (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    # Content-based IDs avoid copying patient names from source filenames.
    ident = hashlib.sha256(path.read_bytes()).hexdigest()
    dest = out / ident
    dest.mkdir(parents=True, exist_ok=True)
    for name, image in (("overlay.png", overlay), ("mask.png", mask * 255)):
        if not cv2.imwrite(str(dest / name), image):
            raise OSError(f"Cannot write {name}")
    np.save(dest / "score.npy", probability)
    report = {"status": "inference_completed", "diagnosis": None,
              "notice": "Research only. Empty mask does not rule out disease. Scores are not cancer risk.",
              "source_sha256": ident, "config": c, "roi_xywh": [x, y, rw, rh],
              "shape_hw": [h, w], "pixel_spacing_row_col_mm": spacing,
              "region_count_2d": len(regions), "regions": regions,
              "processing_seconds": round(time.perf_counter() - t0, 3)}
    (dest / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{ident[:12]}: {len(regions)} regions (2D); {dest}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    modes = parser.add_subparsers(dest="mode", required=True)
    batch = modes.add_parser("batch", help="Process one file or a directory of slices")
    batch.add_argument("input", type=Path)
    capture = modes.add_parser("capture", help="Call a vendor-specific single-frame capture command")
    capture.add_argument("--command-json", type=Path, required=True)
    capture.add_argument("--count", type=int, default=1, help="0 = repeat until Ctrl+C")
    capture.add_argument("--interval", type=float, default=2)
    capture.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    c = read_config(args.config)
    if args.mode == "capture" and c["source_kind"] != "photo":
        parser.error("Camera capture requires source_kind=photo")
    predictor = Predictor(args.model, c)
    # Fingerprint model once; copy into each per-slice report for reproducibility.
    with args.model.open("rb") as f:
        c["model_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "batch":
        if not args.input.exists():
            parser.error("Input path does not exist")
        if args.input.is_file():
            paths = [args.input]
        else:
            # All files in DICOM mode, including extensionless DICOM; use a clean input directory.
            paths = sorted(p for p in args.input.rglob("*") if p.is_file() and
                           (c["source_kind"] == "dicom" or p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")))
            if args.out.resolve().is_relative_to(args.input.resolve()):
                parser.error("Output directory must be outside input directory")
        if not paths:
            parser.error("No input files")
        errors = []
        for index, path in enumerate(paths):
            try:
                run_one(path, args.out, c, predictor, args.roi)
            except Exception as exc:
                print(f"FAILED input #{index + 1}: {exc}", flush=True)
                errors.append({"input_index_1based": index + 1, "error": str(exc)})
        (args.out / "batch_status.json").write_text(json.dumps(
            {"total": len(paths), "failed": len(errors), "errors": errors}, indent=2), encoding="utf-8")
        return 1 if errors else 0

    command = json.loads(args.command_json.read_text(encoding="utf-8"))
    if not isinstance(command, list) or not command or not all(isinstance(s, str) for s in command):
        parser.error("command JSON must be a non-empty array of strings")
    if not any("{output}" in s for s in command):
        parser.error("Capture command must contain {output}")
    if args.count < 0 or not math.isfinite(args.interval) or args.interval < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("Invalid count/interval/timeout")
    i = 0
    try:
        while args.count == 0 or i < args.count:
            with tempfile.TemporaryDirectory(prefix="ct_capture_") as tmp:
                image = Path(tmp) / "frame.jpg"
                subprocess.run([s.replace("{output}", str(image)) for s in command],
                               check=True, timeout=args.timeout, shell=False)
                if not image.is_file() or image.stat().st_size == 0:
                    raise RuntimeError("Camera command did not produce a JPEG at {output}")
                run_one(image, args.out, c, predictor, args.roi)
            i += 1
            if args.count == 0 or i < args.count:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Capture stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
