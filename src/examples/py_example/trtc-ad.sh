cd /home/ps/workspace/trt/lean/TensorRT-10.11.0.33/targets/x86_64-linux-gnu/bin
export LD_LIBRARY_PATH=/home/ps/workspace/trt/lean/TensorRT-10.11.0.33/targets/x86_64-linux-gnu/lib:$LD_LIBRARY_PATH
./trtexec --onnx="/home/ps/workspace/trt/cvter/workspace/ad/ad2/net_100.onnx" \
            --saveEngine="/home/ps/workspace/trt/cvter/workspace/ad/ad2/net_100.trtmodel" \
            --minShapes=images:1x3x512x512 \
            --optShapes=images:1x3x512x512 \
            --maxShapes=images:1x3x512x512 \
            --fp16 \
            --device=0 