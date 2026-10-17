# 条码、QR 与 OCR

源码：`src/cv_code/`。可视化：`tests/test_code_viz.py`。不弹窗。

## QR

`decode_qr` 用 OpenCV `QRCodeDetector`（先多码，再单码）。不读模型文件。微信 QR 需要 caffemodel，这里不用。

```python
hits = cvter.decode_qr(image)
for h in hits:
    print(h.text, h.format, h.polygon)  # polygon：Nx2
```

## 一维码

`decode_barcode` 先试 OpenCV `BarcodeDetector`。OpenCV 4.x 在没有超分模型时，合成的 EAN/UPC 经常 `detectAndDecode` 直接失败。这时如果编译机有 `zbar.h`（Makefile 链 `-lzbar`），会落到 zbar。两边都没有时 `barcode_available()` 为 false，返回空列表。

`decode_codes` 是 QR 和一维码的结果接在一起。

## OCR

```python
r = cvter.recognize_text(image, lang="eng")
print(r.backend, r.found, r.text)
```

编进 Tesseract（`pkg-config tesseract` 成功，编译加 `-DCVTER_WITH_TESSERACT`）且能加载 `eng` 时，`backend` 为 `tesseract`，页面模式是单行。否则 `backend` 为 `none`，`found` 为 false，不报错。
