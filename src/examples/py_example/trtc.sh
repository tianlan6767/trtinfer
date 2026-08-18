cd /home/ps/workspace/trt/lean/TensorRT-10.11.0.33/targets/x86_64-linux-gnu/bin
export LD_LIBRARY_PATH=/home/ps/workspace/trt/lean/TensorRT-10.11.0.33/targets/x86_64-linux-gnu/lib:$LD_LIBRARY_PATH
./trtexec --onnx="/home/ps/workspace/trt/cvter/workspace/pretrain/yolov8s-seg.transd.onnx" \
            --saveEngine="/home/ps/workspace/trt/cvter/workspace/pretrain/yolov8s-seg.transd.trtmodel" \
            --minShapes=images:1x3x640x640 \
            --optShapes=images:1x3x640x640 \
            --maxShapes=images:18x3x640x640 \
            --fp16 \
            --device=0 