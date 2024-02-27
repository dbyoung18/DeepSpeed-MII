#!/bin/bash

: ${TYPE=${1:-"pipeline"}}
: ${TP=${2:-1}}
: ${DP=${3:-1}}
: ${MODEL:="/datadisk/share/llama2-7b"}
: ${LAUNCH:="python"}
: ${SKIP_DECODE:=false}

function set_env() {
  # 1th round, set HF_HOME
  export HF_HOME=/datadisk/share/huggingface

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
}

function main() {
  # set_env

  case ${LAUNCH} in
  "mpi") CMD="mpirun -np ${TP} --prepend-rank python" ;;
  "ipdb") CMD="ipdb3" ;;
  *) CMD="python -u" ;;
  esac

  case ${TYPE} in
  "pipeline") CMD+=" non-persistent/pipeline.py --model ${MODEL} --tp ${TP}" ;;
  "serve") CMD+=" persistent/serve.py --model ${MODEL} --tp ${TP} --dp ${DP}" ;;
  "client") CMD+=" persistent/client.py" ;;
  "terminate") CMD+=" persistent/terminate.py" ;;
  "oneshot") CMD+=" persistent/oneshot.py --model ${MODEL} --tp ${TP} --dp ${DP}" ;;
  "stream") CMD+=" persistent/stream.py --model ${MODEL} --tp ${TP} --dp ${DP}" ;;
  esac

  [ ${SKIP_DECODE} == true ] && CMD+=" --skip_decode"

  echo ${CMD}
  exec ${CMD}
}

main
