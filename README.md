# Segmentation lát CT phổi trên Raspberry Pi 4 + OV5642

Đây là chương trình nghiên cứu chạy CNN segmentation 2D bằng CPU. Đầu vào là
từng lát CT DICOM hoặc ảnh JPEG/PNG chụp lại một lát CT. Đầu ra là mask nhị phân,
ảnh tô đỏ vùng mô hình dự đoán, bản đồ điểm số và JSON cho từng ảnh.

**Chưa kèm mô hình đã huấn luyện.** Không thể xác định khả năng nhận diện tổn
thương khi chỉ biết tên “CNN segmen”. Bạn cần file trọng số, đúng mã kiến trúc,
ý nghĩa lớp foreground và quy trình tiền xử lý lúc huấn luyện.
Mô hình phân vùng toàn bộ phổi không tự trở thành mô hình phân vùng nốt/u phổi.
Mask đỏ chỉ là vùng mô hình dự đoán, không phải xác nhận ung thư. Mask trống
không loại trừ bệnh; điểm số pixel không phải xác suất mắc ung thư.

## 1. OV5642 và ảnh CT

OV5642 chụp ánh sáng nhìn thấy từ phim trên hộp đèn hoặc màn hình hiển thị CT;
camera không tự chụp CT bên trong người. Nên dùng DICOM gốc nếu có.
Ảnh chụp lại không có HU hoặc tỉ lệ mm/pixel đáng tin cậy từ CT gốc.

Tên cảm biến OV5642 chưa xác định được board/driver. Bản này kết nối camera
qua **lệnh chụp một ảnh JPEG của driver đang hoạt động**, không giả định
`cv2.VideoCapture(0)` đọc được SPI. Không kèm driver SPI hay bảng thanh ghi
OV5642 chưa được kiểm tra trên module của bạn.

Nếu là ArduCAM Mini OV5642 có FIFO, cần SPI để truyền ảnh và thường I2C để
cấu hình cảm biến. Đối chiếu sơ đồ của đúng board và driver trước khi đấu dây,
đặc biệt điện áp và chân CS. Không coi OV5642 dạng DVP trần là camera SPI.
Driver ArduCAM cũ bên dưới có cấu hình BCM2835/2836; không giả định chạy nguyên
trạng trên Pi 4 hoặc Raspberry Pi OS hiện tại.

Ảnh camera phải chứa **một lát cắt**, đã cân thẳng, đủ nét, không lóa, đúng chiều
và không có giao diện/chữ bên trong vùng đưa vào mạng. Nếu chụp cả tấm phim có
nhiều ô, chọn ROI từng ô hoặc cắt riêng trước. `--roi X Y W H` chọn một hình chữ
nhật cố định, không tự sửa phối cảnh hoặc tự tìm các ô phim.
Mô hình huấn luyện trên DICOM chưa được coi là phù hợp với ảnh chụp lại;
cần huấn luyện/đánh giá trên dữ liệu cùng cách thu nhận.

## 2. Cài trên Pi 4

Ưu tiên Raspberry Pi OS **64-bit**, Python 3.11–3.13 cho nhánh TorchScript
(API TorchScript đã deprecated; không chọn Python 3.14 cho triển khai nhánh này).
Tạo môi trường:

```bash
cd pi4_ct_seg
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Chọn một runtime:

```bash
# Để chạy TorchScript .pt:
python -m pip install torch

# Hoặc, nếu đã xuất mô hình ONNX:
python -m pip install onnxruntime
```

Wheel phải tương thích phiên bản Python và Linux ARM64 của máy. Nếu pip báo
không có wheel, cần dùng cặp Python/runtime được nhà cung cấp hỗ trợ; không
cài wheel x86 lên Pi. Không có cam kết tốc độ FPS: đo trực tiếp với mô hình
của bạn. Bắt đầu kích thước 256x256 chỉ khi phù hợp lúc huấn luyện, dùng
`threads=2`, xử lý từng lát để giới hạn RAM. Thu nhỏ ảnh có thể mất nốt nhỏ.

## 3. File .pt của CNN

Ứng dụng đọc **TorchScript**, không đọc trực tiếp một `state_dict` bất kỳ.
Nếu file hiện tại là checkpoint PyTorch, thực hiện xuất trên máy đã huấn luyện
có cùng phiên bản code/thư viện:

```bash
python export_pt.py --checkpoint best.pt \
  --factory my_network:UNet \
  --kwargs '{"in_channels":1,"out_channels":1}' \
  --config config.example.json --out model_scripted.pt
```

`my_network:UNet` và các tham số trên chỉ là **ví dụ tên kiến trúc**. Thay bằng
module/lớp/tham số thật, đặt module đó trong thư mục hiện tại hoặc Python path.
Script hỗ trợ state_dict trực tiếp hoặc dict có khóa `state_dict`/
`model_state_dict`, nạp strict và xuất bằng `torch.jit.script`.
Checkpoint dùng định dạng khác, prefix `module.`, hoặc model không script được
cần điều chỉnh theo mã huấn luyện. Không tự đoán kiến trúc từ file .pt.

Hợp đồng mô hình mà ứng dụng hỗ trợ:

- Input một tensor float32 NCHW `[1,C,H,W]`, C=1 hoặc 3; C=3 dùng thứ tự RGB.
- Output một tensor `[1,1,h,w]` cho lớp **tổn thương/nốt nghi ngờ**.
- Output logits thì ứng dụng dùng sigmoid; output đã sigmoid phải chọn
  `output_kind="probabilities"`. Không hỗ trợ trực tiếp logits 2 lớp softmax,
  output tuple/dict, model 3D/2.5D, YOLO hoặc lung-only model.
- Mạng phải làm việc với **resize stretch** và chuẩn hóa dưới đây. Nếu lúc
  huấn luyện dùng letterbox, z-score, CLAHE hoặc cửa sổ khác, phải sửa tiền xử lý
  theo đúng pipeline trước khi dùng kết quả.

## 4. Cấu hình đúng với lúc huấn luyện

Sao chép `config.example.json` thành `config.json`, sửa:

| Trường | Ý nghĩa |
|---|---|
| engine | `torchscript` hoặc `onnx` |
| source_kind | `photo` cho ảnh 8-bit/camera, `dicom` cho CT gốc |
| height, width | Kích thước input mạng |
| channels | 1 grayscale hoặc 3 RGB |
| mean, std | Một giá trị mỗi kênh: `(pixel_0_1 - mean) / std` |
| output_kind | `logits` hoặc `probabilities` |
| threshold | Ngưỡng điểm pixel; 0.5 chỉ là giá trị mẫu, cần chọn trên validation |
| window_center, window_width | Cửa sổ HU dùng cho DICOM; bỏ qua với ảnh camera |
| threads | Số luồng CPU của runtime |

DICOM: `HU = pixel * RescaleSlope + RescaleIntercept`, sau đó
`clip((HU - (center - width/2)) / width, 0, 1)`. Mặc định center=-600,
width=1500 là cấu hình mẫu, không phải chuẩn áp dụng cho mọi mô hình.
Pixel padding bị loại khỏi mask. DICOM Enhanced/multiframe, localizer được
đánh dấu, dữ liệu không CT hoặc thiếu rescale sẽ bị từ chối. DICOM nén có thể
cần decoder pydicom bổ sung theo TransferSyntax; lỗi đọc không được coi là
“không thấy tổn thương”. Chưa tự lọc axial theo orientation; thư mục đầu vào
phải được chọn đúng series/plane như lúc huấn luyện.

## 5. Chạy trên thư mục ảnh/lát cắt

Ảnh PNG/JPEG 8-bit, cấu hình `source_kind=photo`:

```bash
python segment.py --model model_scripted.pt --config config.json \
  --out results batch input_photos
```

Chỉ chọn vùng một lát cắt trong ảnh lớn (tọa độ pixel ảnh gốc):

```bash
python segment.py --model model_scripted.pt --config config.json \
  --out results --roi 100 80 600 600 batch input_photos
```

DICOM: đổi `source_kind` thành `dicom`, khớp các tham số tiền xử lý với model:

```bash
python segment.py --model model_scripted.pt --config config.json \
  --out results batch input_dicom
```

Có thể thay thư mục bằng đường dẫn một file. Thư mục DICOM được đọc đệ quy,
bao gồm file không có đuôi; dùng thư mục chỉ chứa series cần xử lý. Thư mục
kết quả phải nằm ngoài thư mục input. Chương trình chạy độc lập từng lát,
không sắp xếp theo vị trí giải phẫu, không dựng thể tích 3D, không đếm số u
duy nhất qua nhiều lát. Một tổn thương có thể xuất hiện ở nhiều lát.

## 6. Chụp từ OV5642 rồi segmentation

Trước hết xác nhận driver của board có thể chụp **một ảnh** và lưu JPEG,
thoát với mã 0 sau khi ghi xong. Tạo `capture_command.json` chứa mảng argv.
Ví dụ giao diện wrapper bạn cần cung cấp:

```json
["/home/pi/ov5642_driver/capture_one", "--output", "{output}"]
```

**Đường dẫn/chương trình và cờ `--output` trên là ví dụ, chưa phải driver được
kèm trong gói.** Thay toàn bộ bằng lệnh thật của driver/module bạn dùng.
Nếu driver chỉ streaming, cần wrapper lấy đúng một frame JPEG rồi thoát.
Ứng dụng truyền đường dẫn file tạm qua `{output}`, gọi không qua shell và chờ
tối đa 30 giây. Không dùng pipe, `sudo` hoặc lệnh bật GUI trong JSON này.
Tài khoản chạy chương trình cần có quyền SPI/I2C theo hướng dẫn driver.

```bash
# Chụp/xử lý một ảnh:
python segment.py --model model_scripted.pt --config config.json \
  --out results capture --command-json capture_command.json

# Lặp đến khi Ctrl+C; thay lát CT trước mỗi lần chụp:
python segment.py --model model_scripted.pt --config config.json \
  --out results --roi 100 80 600 600 \
  capture --command-json capture_command.json --count 0 --interval 3
```

Khoảng nghỉ tính sau khi chụp và suy luận xong, không phải cam kết FPS. Chương
trình chưa điều khiển cơ cấu di chuyển phim hoặc tự chuyển lát trên màn hình.

## 7. Kết quả và giới hạn

Mỗi ảnh có thư mục theo SHA256 nội dung đầu vào, chứa:

- `overlay.png`: ảnh có vùng đỏ và nhãn nghiên cứu.
- `mask.png`: cùng kích thước ảnh nguồn; 0 nền, 255 foreground.
- `score.npy`: điểm float32 sau sigmoid, cùng kích thước ảnh, ngoài ROI = 0.
- `result.json`: config, fingerprint model/ảnh, ROI, vùng liên thông 2D,
  bbox và diện tích pixel. Diện tích mm² chỉ có với DICOM có PixelSpacing.

`batch_status.json` báo file lỗi và mã thoát khác 0 nếu có lỗi; mask trống sau
suy luận thành công là một trạng thái khác. Không lọc thành phần nhỏ mặc định
để tránh tự ý bỏ nốt nhỏ. Số vùng liên thông không phải số ung thư.
Ảnh giống hệt sẽ ghi cùng thư mục; dùng `--out` mới cho mỗi lần đổi mô hình
hoặc cấu hình để giữ kết quả trước đó.

Xử lý cục bộ, không tải ảnh lên mạng. DICOM gốc không bị sửa. Hash tên file
không phải khử định danh hoàn chỉnh: ảnh camera có thể còn chữ/tên bệnh nhân.
Ảnh chụp tạm trong chế độ capture được xóa sau xử lý, chỉ lưu các kết quả trên.

Trước khi sử dụng ngoài thử nghiệm, đánh giá trên tập bệnh nhân tách biệt:
Dice/IoU cho mask, độ nhạy theo tổn thương và false positives/scan. So sánh với
bác sĩ, kiểm tra miền dữ liệu camera/DICOM và thời gian/RAM trên Pi thật.
Gói này chưa được xác nhận lâm sàng và không dùng độc lập để quyết định điều trị.

## 8. Kiểm tra phần mềm

```bash
python -m unittest discover -s tests -v
```

Test dùng mô hình/dữ liệu tổng hợp để kiểm tra suy luận TorchScript, alignment
ROI, rescale DICOM, pixel padding và từ chối output không hợp lệ. Mô hình
tổng hợp chỉ kiểm tra code, không dùng để phát hiện bệnh.

## Tài liệu tham khảo

- ArduCAM SPI reference: https://github.com/ArduCAM/RaspberryPi/tree/master/SPI_Camera
- pydicom pixel data: https://pydicom.github.io/pydicom/stable/guides/user/working_with_pixel_data.html
- ONNX Runtime ARM64/Python: https://onnxruntime.ai/docs/get-started/with-python
- PyTorch TorchScript: https://docs.pytorch.org/docs/stable/jit.html
- Ý nghĩa nốt phổi: https://www.cancer.org/cancer/types/lung-cancer/detection-diagnosis-staging/lung-nodules.html
