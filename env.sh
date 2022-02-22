
cuda_version=9.0
#tensorrt_version=7.2.3.4 # 7.2.3.4, 8.2.1.8
if [[ 9.0 == $cuda_version ]]; then
    export PATH=/usr/local/cuda-9.0/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH
else
    echo "invalid cuda+tensorrt version combination!"
fi

export PYTHONPATH=/home/ubuntu/jliu/codes/facenet/src