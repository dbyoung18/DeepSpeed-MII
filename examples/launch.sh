#!/bin/bash

: ${MODEL=${1:-${HOME}/llama2-7b-chat}} # model name or model path
: ${LAUNCHER=${2:-mpi}}                 # python(TP=1 or server only) | mpi | deepspeed
: ${TYPE=${3:-pipeline}}                # pipeline | serve | stream(serve)
: ${TP=${4:-1}}                         # tensor-parallel(ranks/instance)
: ${DP=${5:-1}}                         # data-parallel(num instances)

function set_env() {
  echo -e "\033[32m==> Setting IPEX Runtime ENV \033[0m"
  set -x
  export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
  export TORCH_LLM_ALLREDUCE=1
  set +x
}

function main() {
  set_env

  # backup proxy
  export http_proxy_bak=${http_proxy}
  export https_proxy_bak=${https_proxy}
  unset http_proxy https_proxy

  case ${TYPE} in
  "pipeline")
    case ${LAUNCHER} in
    "python") CMD="python none-persistent.py --model ${MODEL} --tp ${TP}" ;; # TP=1 only
    "mpi") CMD="mpirun -np ${TP} python none-persistent.py --model ${MODEL} --tp ${TP}" ;;
    "deepspeed") CMD="deepspeed --num_gpus ${TP} none-persistent.py --model ${MODEL} --tp ${TP}" ;;
    "ipdb")
      if [[ ${TP} == 1 ]]; then
        CMD="ipdb3 none-persistent.py --model ${MODEL} --tp ${TP}"
      else
        CMD="mpirun -s $((TP - 1)) -np $((TP - 1)) --prepend-rank python none-persistent.py --model ${MODEL} --tp ${TP} : -np 1 ipdb3 none-persistent.py --model ${MODEL} --tp ${TP}"
      fi
      ;;
    esac
    ;;
  "serve")
    CMD="python persistent.py --model ${MODEL} --tp ${TP}"
    ;;
  "stream")
    CMD="python persistent.py --model ${MODEL} --tp ${TP} --stream"
    ;;
  esac

  echo CMD=${CMD}
  eval ${CMD}

  # restore proxy
  export http_proxy=${http_proxy_bak}
  export https_proxy=${https_proxy_bak}
  unset http_proxy_bak https_proxy_bak
}

main
