import numpy as np
import os
import cv2
import numpy as np

def load_tensor(file):

    with open(file, "rb") as f:
        binary_data = f.read()

    magic_number, ndims, dtype = np.frombuffer(binary_data, np.uint32, count=3, offset=0)

    assert magic_number == 0xFCCFE2E2, f"{file} not a tensor file."

    dims = np.frombuffer(binary_data, np.uint32, count=ndims, offset=12)

    if dtype == 0:
        np_dtype = np.float32
    elif dtype == 1:
        np_dtype = np.float16
    elif dtype == 3:
        np_dtype = np.uint8
    else:
        raise RuntimeError(f"Unsupported dtype = {dtype}")

    tensor = np.frombuffer(
        binary_data,
        np_dtype,
        offset=(ndims + 3) * 4
    ).reshape(*dims)

    return tensor.copy()   # 防止 read-only

if __name__ == "__main__":
    for i in range(10):
        bin_path = f"/home/ps/workspace/trt/cvter/workspace/{i}_orig_box_segment.bin"
        if not os.path.exists(bin_path):
            print(f"{bin_path} not exists, skip")
            continue
        tensor = load_tensor(bin_path)
        print(tensor.shape)
        tensor = tensor.copy()
        print(tensor.max())
        # tensor = np.transpose(tensor, (1, 2, 0))
        # tensor = tensor.squeeze()
        # tensor[tensor < 0.40] = 0
        # tensor[tensor >= 0.40] = 1.0
        # image = (tensor * 255.0).astype(np.uint8)
        # print(image.max())
        # # print(tensor)
        # # exit()
        # # image = tensor* 255.0
        # # image = image.astype(np.uint8)
        # # image = image[:, :, ::-1]  # RGB to BGR
        cv2.imwrite(f"test_orig_{i}.png", tensor)  
    # print(tensor)

    # tensor = load_tensor("/home/ps/workspace/trt/cvter/workspace/cls_infer_output_0.bin")
    # print(tensor.max())
    # print(tensor.argmax())
    