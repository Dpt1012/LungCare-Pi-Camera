import copy
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from segment import Predictor, decode, load_image, read_config, run_one


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.c = read_config(Path(__file__).resolve().parents[1] / "config.example.json")

    def tearDown(self):
        self.temp.cleanup()

    def test_real_torchscript_roi_alignment_and_report(self):
        # Identity probabilities expose ROI coordinate errors without a clinical model.
        net = torch.jit.script(torch.nn.Identity())
        model = self.root / "identity.pt"
        net.save(str(model))
        c = copy.deepcopy(self.c)
        c.update(height=8, width=8, output_kind="probabilities")
        image = np.zeros((20, 24), np.uint8)
        image[5:13, 7:15] = 255
        source = self.root / "input.png"
        cv2.imwrite(str(source), image)
        run_one(source, self.root / "results", c, Predictor(model, c), (7, 5, 8, 8))
        result = next((self.root / "results").iterdir())
        mask = cv2.imread(str(result / "mask.png"), cv2.IMREAD_GRAYSCALE)
        np.testing.assert_array_equal(mask, image)
        report = json.loads((result / "result.json").read_text())
        self.assertEqual(report["regions"][0]["bbox_xywh_px"], [7, 5, 8, 8])
        self.assertEqual(report["regions"][0]["area_px"], 64)
        self.assertNotIn("area_mm2", report["regions"][0])
        self.assertIsNone(report["diagnosis"])

    def test_invalid_outputs_and_sigmoid(self):
        np.testing.assert_allclose(decode(np.zeros((1, 1, 2, 2)), self.c), 0.5)
        for bad in (np.zeros((1, 2, 8, 8)), np.full((1, 1, 2, 2), np.nan)):
            with self.assertRaises(ValueError):
                decode(bad, self.c)
        c = dict(self.c, output_kind="probabilities")
        with self.assertRaises(ValueError):
            decode(np.full((1, 1, 2, 2), 2.0), c)

    def test_checkpoint_export_and_capture_cli(self):
        project = Path(__file__).resolve().parents[1]
        net = torch.nn.Conv2d(1, 1, 1)
        with torch.no_grad():
            net.weight.fill_(1)
            net.bias.fill_(-0.5)
        checkpoint = self.root / "weights.pt"
        torch.save({"state_dict": net.state_dict()}, checkpoint)
        model = self.root / "scripted.pt"
        config = self.root / "config.json"
        config.write_text(json.dumps(self.c))
        exported = subprocess.run([
            sys.executable, str(project / "export_pt.py"), "--checkpoint", str(checkpoint),
            "--factory", "torch.nn:Conv2d", "--kwargs",
            '{"in_channels":1,"out_channels":1,"kernel_size":1}',
            "--config", str(config), "--out", str(model)], capture_output=True, text=True)
        self.assertEqual(exported.returncode, 0, exported.stderr)
        source = self.root / "frame.jpg"
        cv2.imwrite(str(source), np.full((32, 32), 255, np.uint8))
        # Simulate the vendor command contract, not physical camera behavior.
        command = self.root / "capture.json"
        command.write_text(json.dumps([sys.executable, "-c",
            "import shutil,sys; shutil.copyfile(sys.argv[1],sys.argv[2])", str(source), "{output}"]))
        out = self.root / "capture_results"
        captured = subprocess.run([
            sys.executable, str(project / "segment.py"), "--model", str(model),
            "--config", str(config), "--out", str(out), "capture",
            "--command-json", str(command)], capture_output=True, text=True)
        self.assertEqual(captured.returncode, 0, captured.stderr)
        report = json.loads(next(out.glob("*/result.json")).read_text())
        self.assertEqual(report["regions"][0]["area_px"], 32 * 32)
        self.assertEqual(len(report["config"]["model_sha256"]), 64)

    def make_dicom(self):
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        ds = FileDataset(str(self.root / "slice.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.Modality = "CT"
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.Rows, ds.Columns = 2, 2
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit, ds.PixelRepresentation = 15, 1
        ds.RescaleSlope, ds.RescaleIntercept = 2, -1000
        ds.PixelSpacing = [0.7, 0.8]
        ds.PixelPaddingValue = -2000
        ds.PixelData = np.array([[-2000, 0], [500, 1000]], dtype="<i2").tobytes()
        return ds

    def test_dicom_hu_padding_and_reject_multiframe(self):
        ds = self.make_dicom()
        path = self.root / "slice.dcm"
        ds.save_as(path, enforce_file_format=True)
        c = dict(self.c, source_kind="dicom", window_center=0, window_width=2000)
        rgb, spacing, valid = load_image(path, c)
        np.testing.assert_allclose(rgb[:, :, 0], [[0, 0], [0.5, 1]])
        self.assertEqual(spacing, [0.7, 0.8])
        self.assertFalse(valid[0, 0])
        ds.NumberOfFrames = 2
        ds.save_as(path, enforce_file_format=True)
        with self.assertRaises(ValueError):
            load_image(path, c)


if __name__ == "__main__":
    unittest.main()
