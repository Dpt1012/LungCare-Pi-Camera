# Lung CT Slice Segmentation on Raspberry Pi 4 + OV5642

This is a research program that runs a 2D CNN segmentation model on the CPU. The
input is individual DICOM CT slices, or JPEG/PNG photos of a CT slice. The output
is a binary mask, an image with the model's predicted region highlighted in red,
a score map, and a JSON file for each image.

**No trained model is included.** Lesion-detection ability cannot be determined
from the name "CNN segmen" alone. You need the weights file, the exact
architecture code, the meaning of the foreground class, and the preprocessing
pipeline used during training.
A model that segments the whole lung does not automatically become a model that
segments lung nodules/tumors. The red mask is only the region the model
predicted, not a confirmation of cancer. An empty mask does not rule out
disease; pixel scores are not cancer probabilities.

## 1. OV5642 and CT images

The OV5642 captures visible light from film on a lightbox or from a screen
displaying the CT; the camera does not perform a CT scan of the body. Use the
original DICOM whenever available.
Photographed images do not carry reliable HU values or mm/pixel scale from the
original CT.

The sensor name OV5642 alone does not identify the board/driver. This version
connects to the camera through **a command that captures one JPEG using the
active driver**; it does not assume `cv2.VideoCapture(0)` can read over SPI. No
SPI driver or OV5642 register table is included, as none has been verified on
your module.

If you are using an ArduCAM Mini OV5642 with FIFO, SPI is needed to transfer
images and I2C is usually needed to configure the sensor. Check the schematic of
your exact board and driver before wiring, especially voltage and the CS pin. Do
not treat a bare DVP OV5642 as an SPI camera.
The legacy ArduCAM driver referenced below has BCM2835/2836 configurations; do
not assume it runs unmodified on a Pi 4 or current Raspberry Pi OS.

The camera image must contain **a single slice**, be straight, in focus, free of
glare, correctly oriented, and have no UI/text inside the region fed to the
network. If you photograph an entire film sheet with multiple panels, select an
ROI for each panel or crop them separately first. `--roi X Y W H` selects a fixed
rectangle; it does not correct perspective or find film panels automatically.
A model trained on DICOM should not be assumed suitable for photographed images;
it must be trained/evaluated on data acquired the same way.

## 2. Installation on Pi 4

Prefer Raspberry Pi OS **64-bit**, with Python 3.11–3.13 for the TorchScript path
(the TorchScript API is deprecated; do not choose Python 3.14 for deploying this
path).
Create the environment:

```bash
cd pi4_ct_seg
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Choose one runtime:

```bash
# To run a TorchScript .pt:
python -m pip install torch

# Or, if you have exported an ONNX model:
python -m pip install onnxruntime
```

The wheel must match your Python version and Linux ARM64. If pip reports no
wheel available, use a Python/runtime combination supported by the vendor; do
not install x86 wheels on the Pi. No FPS speed is promised: measure it directly
with your model. Start at 256x256 only if that matches training, use
`threads=2`, and process one slice at a time to limit RAM. Downscaling images
may lose small nodules.

## 3. The CNN .pt file

The application loads **TorchScript**, not an arbitrary `state_dict` directly.
If your current file is a PyTorch checkpoint, export it on the training machine
using the same code/library versions:

```bash
python export_pt.py --checkpoint best.pt \
  --factory my_network:UNet \
  --kwargs '{"in_channels":1,"out_channels":1}' \
  --config config.example.json --out model_scripted.pt
```

`my_network:UNet` and the parameters above are only **example architecture
names**. Replace them with your real module/class/parameters, and place that
module in the current directory or on the Python path.
The script supports a plain state_dict or a dict with a `state_dict`/
`model_state_dict` key, loads strictly, and exports with `torch.jit.script`.
Checkpoints in other formats, with a `module.` prefix, or models that cannot be
scripted need adjustments based on the training code. The architecture is never
guessed from the .pt file.

Model contract supported by the application:

- Input: one float32 NCHW tensor `[1,C,H,W]`, C=1 or 3; C=3 uses RGB order.
- Output: one tensor `[1,1,h,w]` for the **suspected lesion/nodule** class.
- If the output is logits, the application applies sigmoid; if the output is
  already sigmoided, set `output_kind="probabilities"`. Not directly supported:
  2-class softmax logits, tuple/dict outputs, 3D/2.5D models, YOLO, or
  lung-only models.
- The network must work with the **stretch resize** and normalization below. If
  training used letterboxing, z-score, CLAHE, or a different window, modify the
  preprocessing to match that pipeline before trusting the results.

## 4. Configuration matching training

Copy `config.example.json` to `config.json` and edit:

| Field | Meaning |
|---|---|
| engine | `torchscript` or `onnx` |
| source_kind | `photo` for 8-bit/camera images, `dicom` for original CT |
| height, width | Network input size |
| channels | 1 grayscale or 3 RGB |
| mean, std | One value per channel: `(pixel_0_1 - mean) / std` |
| output_kind | `logits` or `probabilities` |
| threshold | Pixel score threshold; 0.5 is only a sample value, choose it on validation data |
| window_center, window_width | HU window for DICOM; ignored for camera images |
| threads | Number of CPU threads for the runtime |

DICOM: `HU = pixel * RescaleSlope + RescaleIntercept`, then
`clip((HU - (center - width/2)) / width, 0, 1)`. The default center=-600,
width=1500 is a sample configuration, not a standard for every model.
Pixel padding is excluded from the mask. Enhanced/multiframe DICOM, flagged
localizers, non-CT data, or data missing rescale values are rejected.
Compressed DICOM may require an additional pydicom decoder depending on the
TransferSyntax; a read error is never treated as "no lesion found". Axial
slices are not yet filtered automatically by orientation; the input folder must
contain the correct series/plane as used in training.

## 5. Running on a folder of images/slices

8-bit PNG/JPEG images, with `source_kind=photo`:

```bash
python segment.py --model model_scripted.pt --config config.json \
  --out results batch input_photos
```

Select only one slice region in a larger image (pixel coordinates of the
original image):

```bash
python segment.py --model model_scripted.pt --config config.json \
  --out results --roi 100 80 600 600 batch input_photos
```

DICOM: change `source_kind` to `dicom` and match the preprocessing parameters to
the model:

```bash
python segment.py --model model_scripted.pt --config config.json \
  --out results batch input_dicom
```

You can replace the folder with a path to a single file. DICOM folders are read
recursively, including files without extensions; use a folder containing only
the series to process. The output folder must be outside the input folder. The
program processes each slice independently: it does not sort by anatomical
position, reconstruct a 3D volume, or count unique tumors across slices. One
lesion may appear in multiple slices.

## 6. Capturing from the OV5642 and segmenting

First confirm your board's driver can capture **one image**, save it as JPEG,
and exit with code 0 once written. Create `capture_command.json` containing an
argv array. Example of the wrapper interface you need to provide:

```json
["/home/pi/ov5642_driver/capture_one", "--output", "{output}"]
```

**The path/program and `--output` flag above are examples, not a driver included
in this package.** Replace them entirely with the real command for your
driver/module. If the driver only streams, you need a wrapper that grabs exactly
one JPEG frame and exits.
The application passes a temporary file path via `{output}`, runs the command
without a shell, and waits up to 30 seconds. Do not use pipes, `sudo`, or
GUI-launching commands in this JSON. The account running the program needs
SPI/I2C permissions as described in the driver's instructions.

```bash
# Capture/process one image:
python segment.py --model model_scripted.pt --config config.json \
  --out results capture --command-json capture_command.json

# Loop until Ctrl+C; change the CT slice before each capture:
python segment.py --model model_scripted.pt --config config.json \
  --out results --roi 100 80 600 600 \
  capture --command-json capture_command.json --count 0 --interval 3
```

The interval is counted after capture and inference finish; it is not an FPS
guarantee. The program does not yet control a film-moving mechanism or advance
slices on a screen automatically.

## 7. Outputs and limitations

Each image gets a folder named by the SHA256 of the input content, containing:

- `overlay.png`: the image with the red region and a research label.
- `mask.png`: same size as the source image; 0 background, 255 foreground.
- `score.npy`: float32 scores after sigmoid, same size as the image, 0 outside the ROI.
- `result.json`: config, model/image fingerprints, ROI, 2D connected regions,
  bounding boxes, and pixel areas. Area in mm² is only available for DICOM with
  PixelSpacing.

`batch_status.json` reports failed files, and the exit code is non-zero if any
errors occurred; an empty mask after successful inference is a distinct state.
Small components are not filtered by default, to avoid silently discarding small
nodules. The number of connected regions is not the number of cancers.
Identical images write to the same folder; use a new `--out` for each model or
configuration change to keep earlier results.

Processing is local; images are not uploaded anywhere. Original DICOM files are
not modified. Hashing file names is not full de-identification: camera images
may still contain patient text/names.
Temporary captures in capture mode are deleted after processing; only the
outputs above are kept.

Before any use beyond experimentation, evaluate on a held-out patient set:
Dice/IoU for masks, per-lesion sensitivity, and false positives per scan.
Compare against radiologists, check the camera/DICOM data domain, and measure
time/RAM on a real Pi.
This package has not been clinically validated and must not be used on its own
to make treatment decisions.

## 8. Software tests

```bash
python -m unittest discover -s tests -v
```

Tests use synthetic models/data to check TorchScript inference, ROI alignment,
DICOM rescaling, pixel padding, and rejection of invalid outputs. The synthetic
model only tests the code and must not be used to detect disease.

## References

- ArduCAM SPI reference: https://github.com/ArduCAM/RaspberryPi/tree/master/SPI_Camera
- pydicom pixel data: https://pydicom.github.io/pydicom/stable/guides/user/working_with_pixel_data.html
- ONNX Runtime ARM64/Python: https://onnxruntime.ai/docs/get-started/with-python
- PyTorch TorchScript: https://docs.pytorch.org/docs/stable/jit.html
- Lung nodule information: https://www.cancer.org/cancer/types/lung-cancer/detection-diagnosis-staging/lung-nodules.html
