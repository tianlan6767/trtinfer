cd /home/ps/workspace/trt/lean/TensorRT-10.11.0.33/targets/x86_64-linux-gnu/bin
export LD_LIBRARY_PATH=/home/ps/workspace/trt/lean/TensorRT-10.11.0.33/targets/x86_64-linux-gnu/lib:$LD_LIBRARY_PATH
./trtexec --onnx="/home/ps/workspace/trt/cvter/workspace/pretrain/yolo11s-cls-dy.onnx" \
            --saveEngine="/home/ps/workspace/trt/cvter/workspace/pretrain/yolo11s-cls-dy-fp32.engine" \
            --minShapes=images:1x3x224x224 \
            --optShapes=images:1x3x224x224 \
            --maxShapes=images:18x3x224x224 \
            # --fp32 \
            # --device=0 