#!/bin/sh
# 1th round, set HF_HOME
export HF_HOME=~/huggingface

# >=2th round, set OFFLINE MODE and use the HF_HOME that 1th round set.
#export HF_DATASETS_OFFLINE=1
#export TRANSFORMERS_OFFLINE=1
#export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_CACHE=$HF_HOME/hub
export TOKENIZERS_PARALLELISM=false
#export ONECCL_BINDINGS_FOR_PYTORCH_ENV_VERBOSE=1
#export CCL_PROCESS_LAUNCHER=none
#export CCL_ZE_IPC_EXCHANGE=drmfd
#export CCL_LOG_LEVEL=error
#export DNNL_VERBOSE=1
#export PROFILE=1
#export TORCH_CCL_DISABLE_ALLREDUCE=1
#export CCL_SYCL_OUTPUT_EVENT=1
#export CCL_USE_EXTERNAL_QUEUE=1
#export CCL_REDUCE_SCATTER_TOPO_READ=0
#export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
#export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=1000
#export CCL_ATL_TRANSPORT=mpi
#export CCL_ZE_DEVICE_CHUNK_COUNT=1
#export CCL_ZE_DEVICE_CACHE_POLICY=chunk
#export FI_PROVIDER=tcp
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
export COL_MAJOR=0
export ENABLE_SDP_FUSION=1
export TORCH_LLM_ALLREDUCE=1
#export IPEX_DEEPSPEED_OPTIMIZE=1
#export LD_PRELOAD=/opt/intel/oneapi/dnnl/2023.0.0/cpu_dpcpp_gpu_dpcpp/lib/libdnnl.so

mpirun -np 2 --prepend-rank python tests/pipeline_xpu.py
#python tests/serve_xpu.py
