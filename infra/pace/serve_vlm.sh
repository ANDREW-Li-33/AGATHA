#!/bin/bash
# usage: serve_vlm.sh start|stop|status
# The model for agatha-vlm --model local: vLLM serves [local]'s model (Gemma 4) on 127.0.0.1, so the
# frames sent to it never leave this node. Run it from ~/agatha inside a GPU job, after build_vllm.sbatch,
# in a shell that stays open, such as the OnDemand session's terminal: Slurm ends what an srun step started
# when that step ends. Or start it and run Stage 2 in one step:
#   srun --jobid=<job> --overlap bash -c \
#     'bash infra/pace/serve_vlm.sh start && bash infra/pace/stage2_cutin.sbatch 1 local'
# For a whole scene list, stage2_list.sh starts it, runs each scene, and stops it.
# It reads configs/stage2.toml, or VLM_CONFIG: Pass 2's max_images and [local]'s id, max_tokens and image
# budget, so the server takes the requests agatha-vlm makes, and the port in [local]'s base_url, 8000 when it
# names none. ICE's nodes are shared: when another user's server has the port, a config names another.
# [local]'s temperature isn't the server's: each request carries it, so setting() is never asked for it.
# Context: vLLM takes the longest the GPUs hold (--max-model-len auto), and start prints it beside what
# Pass 2's longest request needs. A longer request fails, and a rerun asks again. MAX_MODEL_LEN=<tokens>
# asks for a length instead, and vLLM then won't start unless it fits.
# Precision by GPU: BF16 takes about 70 GB, the weights and 20% more, so it runs when the GPUs used hold
# 80 GB or more. Otherwise FP8, which vLLM makes from the BF16 weights as it loads them, with no
# calibration, and an FP8 KV cache, which holds twice the context; KV_CACHE_DTYPE=auto keeps that at
# BF16. A QAT checkpoint comes quantized.
# Request logging is off, so the log in $WOD_HOME/logs holds no request. Only labels and counts print.
set -euo pipefail
source infra/pace/env.sh
export PATH="$HOME/scratch/venvs/vllm/bin:$PATH"
export HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1  # the weights are on scratch already, and vLLM reports nothing home
# FlashInfer's sampler compiles itself at start, which needs ninja and nvcc, and ICE's GPU nodes have neither:
# vLLM 0.32.0 stopped there on ICE on 2026-11-04. PyTorch's sampler needs neither.
export VLLM_USE_FLASHINFER_SAMPLER=0
config=${VLM_CONFIG:-configs/stage2.toml}
host=$(hostname -s)
pidfile=$WOD_HOME/logs/vllm-$host.pid  # one per node: a server answers only on its own
log=$WOD_HOME/logs/vllm-$host.log

setting() {  # one key of one [section] of the config
  awk -v section="[$1]" -v key="$2" '/^\[/ {inside = $1 == section}
    inside && $1 == key {gsub(/"/, "", $3); print $3}' "$config"
}
# The port agatha-vlm sends to, so the server listens there: on 2026-11-04 another user's vLLM had 8000 on ICE.
port=8000
port_in_url='^https?://[^/]*:([0-9]+)(/|$)'
if [[ $(setting local base_url 2> /dev/null) =~ $port_in_url ]]; then  # stop and status run without a config too
  port=${BASH_REMATCH[1]}
fi
url=http://127.0.0.1:$port
# The server this script started: the process in the pid file, if it's still vLLM. When a job ends, Slurm
# ends the server but leaves the file, and another process can take its number later. -ww: the whole command.
ours() { [[ $(ps -ww -o args= -p "$(cat "$pidfile" 2> /dev/null)" 2> /dev/null) == *"vllm serve"* ]]; }
healthy() { curl -sf --noproxy '*' -m 5 -o /dev/null "$url/health"; }  # never through a proxy: this node only
served_len() {  # the context vLLM took, from its list of models
  curl -sf --noproxy '*' -m 5 "$url/v1/models" | grep -o '"max_model_len": *[0-9]*' | grep -o '[0-9]*$' || true
}

launch() {
  model=$(setting local id)
  max_tokens=$(setting local max_tokens)
  image_tokens=$(setting local image_tokens)
  max_images=$(setting pass2 max_images)
  if ! [[ $model && $max_tokens =~ ^[0-9]+$ && $image_tokens =~ ^[0-9]+$ && $max_images =~ ^[0-9]+$ ]]; then
    echo "vllm: $config needs [local] id, max_tokens and image_tokens, and [pass2] max_images" >&2
    exit 2
  fi
  if ! mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2> /dev/null) || [ -z "$mib" ]; then
    echo "vllm: no GPU here: run it inside a GPU job" >&2
    exit 1
  fi
  # Tensor parallelism splits the attention heads, so it takes 1, 2 or 4 GPUs: 4 global KV heads split no further.
  gpus=$(grep -c . <<< "$mib")
  tp=1
  while [ $((tp * 2)) -le "$gpus" ] && [ "$tp" -lt 4 ]; do
    tp=$((tp * 2))
  done
  memory=$(head -n "$tp" <<< "$mib" | awk '{total += $1} END {print total}')  # MiB
  if [[ $model == *-qat-* ]]; then
    precision=(--dtype bfloat16) label=w4a16 kv=auto  # the checkpoint's own 4-bit weights, with 16-bit activations
  elif [[ $model == *[Ff][Pp]8* ]]; then  # quantized already, as RedHatAI's FP8-dynamic copies are: vLLM reads how from it
    precision=() label=fp8-checkpoint kv=auto
  elif [ "$memory" -ge 76294 ]; then  # 80 GB: an 80 GB card reports 81,559 MiB (H100) or 81,920 (A100)
    precision=(--dtype bfloat16) label=bf16 kv=auto
  else
    # vLLM's online FP8: Ada (the L40S) and newer compute in it, and Turing and Ampere keep only the weights in it.
    # The KV cache in FP8 too, as vLLM's Gemma 4 recipe suggests: the weights leave an L40S little room for it.
    precision=(--quantization fp8_per_tensor) label=fp8 kv=fp8
  fi
  # Pass 2's longest request, to compare with the context vLLM takes: max_images images, each with its caption
  # and its share of the track JSON, then the system prompt, the hypothesis and the answer. The JSON is mostly
  # digits, about a token a character, and 20 objects take some 1,700 to 2,600 a timestamp, which the cameras
  # shown share: 1,280 a frame is a guess between one camera and three.
  need=$((max_images * (image_tokens + 1280) + 4096 + max_tokens))
  # --trust-request-mm-kwargs: LocalModel sends each request's image budget, which vLLM 0.32.0 refuses otherwise
  # (HTTP 400 on ICE, 2026-11-04). The only client is our own, on this node.
  # --no-enable-log-requests is vLLM 0.32.0's spelling, which later versions changed: change it with the pin.
  # No --structured-outputs-config with "disable_any_whitespace": don't add it. On 2026-11-04 it stopped Gemma's
  # runaway answers (blank lines between JSON tokens, to the 8,192-token limit), but on Pass 1, 10 windows tried 3
  # times each, 26 of 30 answers came back with no candidates, against 0 of 30 without it. A cut-off is resent on
  # a rerun instead (LocalModel marks it retry), unless [local]'s temperature is 0: a resend would repeat it.
  # --no-enable-prefix-caching: so the same request gets the same answer. At temperature 0 with the cache on, the
  # same Pass 2 or Pass 1 request sent twice came back different every time (0 of 13 pairs the same, and one
  # verdict went REJECTED, then CONFIRMED): a resend finds its prompt cached and computes only the rest, a little
  # differently. With it off, 3 requests sent 3 times each came back identical, at about 15 s a Pass 2 answer
  # (ICE, 2026-11-05). VLLM_BATCH_INVARIANT=1 did that too, but slower (about 22 s). One request at a time keeps
  # batching out of it; Stage 2 sends no other way.
  args=(serve "$model" --host 127.0.0.1 --port "$port" --tensor-parallel-size "$tp" ${precision[@]+"${precision[@]}"}
        --kv-cache-dtype "${KV_CACHE_DTYPE:-$kv}" --max-model-len "${MAX_MODEL_LEN:-auto}"
        --gpu-memory-utilization 0.9 --limit-mm-per-prompt "{\"image\": $max_images}"
        --mm-processor-kwargs "{\"max_soft_tokens\": $image_tokens}" --trust-request-mm-kwargs --no-enable-log-requests
        --disable-uvicorn-access-log --no-enable-prefix-caching)
  if [ "$image_tokens" -gt 280 ]; then  # past Gemma 4's own 280 an image, raise its limit, as vLLM's recipe does
    capacity="\"vision_soft_tokens_per_image\": $image_tokens"
    args+=(--hf-overrides "{$capacity, \"vision_config\": {\"default_output_length\": $image_tokens}}")
  fi
  mkdir -p "$WOD_HOME/logs"
  set -m  # its own process group, so stop reaches the engine and GPU workers it starts too
  nohup vllm "${args[@]}" > "$log" 2>&1 < /dev/null &
  set +m
  echo $! > "$pidfile"
  disown  # this shell doesn't report on it
  echo "vllm: starting $model, $label, $tp GPU(s), max-model-len ${MAX_MODEL_LEN:-auto};" \
       "Pass 2's longest request needs about $need"
}

stop() {  # TERM to the server's process group, then KILL to what's left of it, each given a minute
  if ours; then
    group=-$(cat "$pidfile")
    for signal in TERM KILL; do
      kill -s "$signal" -- "$group" 2> /dev/null || break  # the whole group is gone
      for _ in $(seq 60); do  # it frees the GPU as it exits
        kill -0 -- "$group" 2> /dev/null || break
        sleep 1
      done
    done
    if kill -0 -- "$group" 2> /dev/null; then  # the pid file stays, to stop it again
      echo "vllm: still running" >&2
      exit 1
    fi
  fi
  rm -f "$pidfile"
}

case ${1:-} in
  start)
    if ! ours && healthy; then  # another user's server on this shared node, say: never send it frames
      echo "vllm: port $port answers, but not from a server this script started" >&2
      exit 1
    fi
    if ! ours; then
      launch
    fi
    for _ in $(seq 360); do  # up to an hour: loading the weights and compiling take a while
      # Alive, not ours: until vllm replaces it, the process just started still runs this script.
      if ! kill -0 "$(cat "$pidfile")" 2> /dev/null; then
        rm -f "$pidfile"
        echo "vllm: exited while starting: see $log" >&2
        exit 1
      fi
      if healthy; then
        length=$(served_len)
        echo "vllm: up, max-model-len ${length:-unknown}"
        exit 0
      fi
      sleep 10
    done
    stop
    echo "vllm: not up after an hour, so stopped: see $log" >&2
    exit 1 ;;
  stop)
    stop
    echo "vllm: stopped" ;;
  status)
    if ours && healthy; then
      echo "vllm: up"
      exit 0
    elif ours; then
      echo "vllm: starting, or not answering"
    elif healthy; then
      echo "vllm: port $port answers, but not from a server this script started"
    else
      echo "vllm: down"
    fi
    exit 1 ;;
  *)
    echo "usage: serve_vlm.sh start|stop|status" >&2
    exit 2 ;;
esac
