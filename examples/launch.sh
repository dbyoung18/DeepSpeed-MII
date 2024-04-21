#!/bin/bash

: ${TYPE=${1:-"pipeline"}}
: ${TP=${2:-1}}
: ${DP=${3:-1}}
: ${MODEL:="/datadisk/share/llama2-7b"}
: ${LAUNCH:="python"}
: ${SKIP_DECODE:=false}

function set_env() {
  export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
  export TORCH_LLM_ALLREDUCE=1
}

function main() {
  set_env

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
