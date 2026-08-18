import onnx
import onnx.helper as helper
import sys
import os
from ultralytics import YOLO

r"""
trtexec.exe --onnx="D:\opt\aidt_dev\models\tuidian.transd_dy.onnx" \
                --saveEngine="D:\opt\aidt_dev\models\tuidian.transd_dy_fp16.trtmodel" \
                --minShapes=input_image:1x3x1024x1024 \
                --optShapes=input_image:1x3x4096x4096 \
                --maxShapes=input_image:1x3x4096x4096 \
                --fp16 \
                --device=0 \
                --workspace=20480 \
                --preview=+fasterDynamicShapes0805
"""


def transd(file):

    prefix, suffix = os.path.splitext(file)
    dst = prefix + ".transd" + suffix
    model = onnx.load(file)
    node  = model.graph.node[-1]

    old_output = node.output[0]
    node.output[0] = "pre_transpose"

    for specout in model.graph.output:
        if specout.name == old_output:
            shape0 = specout.type.tensor_type.shape.dim[0]
            shape1 = specout.type.tensor_type.shape.dim[1]
            shape2 = specout.type.tensor_type.shape.dim[2]
            new_out = helper.make_tensor_value_info(
                specout.name,
                specout.type.tensor_type.elem_type,
                [0, 0, 0]
            )
            new_out.type.tensor_type.shape.dim[0].CopyFrom(shape0)
            new_out.type.tensor_type.shape.dim[2].CopyFrom(shape1)
            new_out.type.tensor_type.shape.dim[1].CopyFrom(shape2)
            specout.CopyFrom(new_out)

    model.graph.node.append(
        helper.make_node("Transpose", ["pre_transpose"], [old_output], perm=[0, 2, 1])
    )

    print(f"Model save to {dst}")
    onnx.save(model, dst)
if __name__ == "__main__":
    pth = r"/home/ps/workspace/trt/trt-sahi-yolo/workspace/pretrain/yolo11s-cls.pt"
    model = YOLO(pth)
    success = model.export(format="onnx", imgsz=224, device='0', dynamic=False, simplify=True)  # export the model to ONNX format
    # print(success)
    # success = model.export(format="engine", imgsz=2048, device='0', dynamic=False, simplify=True, half=False)
    # transd(pth.replace(".pt", ".onnx"))
