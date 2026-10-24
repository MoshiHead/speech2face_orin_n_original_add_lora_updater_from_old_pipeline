#!/usr/bin/env bash
# run_x86.sh -- PersonaPlex + IMTalker real-time avatar on an x86_64 NVIDIA GPU
# (RunPod / Lambda / a local workstation). The Jetson counterpart is ./run.sh and keeps the
# tuned Orin settings; this script runs the same configuration with the two Jetson-only
# assumptions removed:
#   * sphn comes from pip, not from the bundled aarch64 .so in src/pylibs
#   * the TensorRT SEANet engine is optional (the PyTorch decoder is used if it is missing)
# and with int4 trunk quantisation driven by what tools/gpu_probe.py found this GPU can do.
#
#   ./run_x86.sh                 defaults (port 8991)
#   PORT=8080 ./run_x86.sh       another port (not 8998: refused by the server)
#   TRY19_INT4_G=0 ./run_x86.sh  bf16 trunk instead of int4 (needs ~26 GB of VRAM)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/src"; W="$HERE/weights"
[ -s "$HERE/.snap_path" ] && [ -s "$HERE/.python_path" ] || { echo "run ./prepare_x86.sh <hf_token> first"; exit 1; }
PY="${PY:-$(cat "$HERE/.python_path")}"   # the Python prepare_x86.sh checked
SNAP="$(cat "$HERE/.snap_path")"
PORT="${PORT:-8991}"   # 8998 is refused by the server (reserved for a legacy production server)
export HF_HOME="${HF_HOME:-$HERE/hf}"

# RunPod (and other) images export HF_HUB_ENABLE_HF_TRANSFER=1; huggingface_hub then refuses to
# download anything at all unless the hf_transfer package is importable. setup_env_x86.sh installs
# it (requirements-x86-extra.txt), but if this Python does not have it, turn the accelerator off
# instead of failing the download.
if [ "${HF_HUB_ENABLE_HF_TRANSFER:-0}" != "0" ] && ! "$PY" -c 'import hf_transfer' 2>/dev/null; then
  echo "    note: HF_HUB_ENABLE_HF_TRANSFER=1 but hf_transfer is not installed -> disabling it"
  export HF_HUB_ENABLE_HF_TRANSFER=0
fi
# compiled-kernel caches next to the package so they survive on the persistent volume
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HERE/.cache/triton}" TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HERE/.cache/inductor}"
export DESIGN_A_DIR="$SRC/niloys" IMTALKER_DIR="$SRC/imtalker"
# NOTE: src/pylibs is deliberately NOT on the path -- the sphn there is an aarch64 build.
export PYTHONPATH="$SRC/niloys:$SRC/moshi_pkg:${PYTHONPATH:-}"
unset TRY19_DEC_CKPT
export NATIVE_PASSTHROUGH=1
export TRY19_STUDENT_GRAPH="${TRY19_STUDENT_GRAPH:-1}" TRY19_RENDER_FP16="${TRY19_RENDER_FP16:-1}" TRY19_PD_REFINE="${TRY19_PD_REFINE:-0}"
export TRY19_PD="$W/pd_quant_onpol_robert5_plaus2.pt"
export TRY19_MIMI_DEC_FT="$W/mimi_dec_gan_robert5.pt"
export TRY19_MIMI_GRAPH="${TRY19_MIMI_GRAPH:-1}" TRY19_MIMI_PARTS="${TRY19_MIMI_PARTS:-seanet_quant}" TRY19_MIMI_CAPTURE_AT_INIT="${TRY19_MIMI_CAPTURE_AT_INIT:-1}"
export TRY19_CODE_GUARD="${TRY19_CODE_GUARD:-1}"

# int4 trunk: ATen's tinygemm kernels are not built for every CUDA architecture, so
# prepare_x86.sh probed them and wrote the verdict. An explicit value in the environment wins.
_int4_user="${TRY19_INT4_G:-}"; _gate_user="${TRY19_DEP_GATING_INT4:-}"
if [ -s "$HERE/.gpu_probe.env" ]; then set -a; . "$HERE/.gpu_probe.env"; set +a; fi
export TRY19_INT4_G="${_int4_user:-${TRY19_INT4_G:-32}}"
export TRY19_DEP_GATING_INT4="${_gate_user:-${TRY19_DEP_GATING_INT4:-1}}"

# TensorRT SEANet decoder: optional here. Without it Mimi's PyTorch SEANet is used.
if [ -s "$W/seanet_w12_fp32.engine" ]; then
  export TRY19_TRT_SEANET="$W/seanet_w12_fp32.engine"
else
  unset TRY19_TRT_SEANET
  echo "note: no weights/seanet_w12_fp32.engine -- using Mimi's PyTorch SEANet decoder"
fi

# ---------------------------------------------------------------------------
# STT + query routing + web search (tool calling) -- opt-in, additive.
# ENABLE_SEARCH=0 (default) reproduces the plain conversational launch command
# with zero new flags appended. ENABLE_RAG is honored as a legacy alias.
#
# When enabled, the moshi reply engine switches to the configuration the
# <lookup>/<ref> LoRA was trained in:
#   * trunk: bnb-4bit (brianmatzelle/personaplex-7b-v1-bnb-4bit) instead of the
#     tinygemm int4 trunk -- PEFT cannot wrap Int4Linear (it is nn.Module, not
#     nn.Linear), and the adapter was trained on the NF4 base anyway
#   * depformer: the stock depformer instead of the Robert_5 one-pass (PD)
#     depformer -- the LoRA wraps depformer attention, which breaks PD's
#     load_state_dict key matching, and the adapter was trained with the stock
#     depformer
#   * Mimi decoder: stock PyTorch SEANet (the GAN fine-tune / TRT engine stay
#     available for the search-off configuration)
# The avatar pipeline (student motion, distilled renderer, blink composite) is
# untouched either way.
ENABLE_SEARCH="${ENABLE_SEARCH:-${ENABLE_RAG:-0}}"
SEARCH_ARGS=()
if [ "$ENABLE_SEARCH" = "1" ]; then
  PERSONAPLEX_BNB4_DIR="${PERSONAPLEX_BNB4_DIR:-$HERE/checkpoints/personaplex_bnb4}"
  REF_LORA_DIR="${REF_LORA_DIR:-${RAG_CHECKPOINT_DIR:-$HERE/checkpoints/rag_lora}}"
  STT_PKG_DIR="${STT_PKG_DIR:-$HERE/checkpoints/stt}"
  CONVERSATION_LOG_DIR="${CONVERSATION_LOG_DIR:-$HERE/conversation_logs}"
  THINKING_SOUND_PATH="${THINKING_SOUND_PATH:-$HERE/assets/ai-thinking-sound.wav}"
  mkdir -p "$CONVERSATION_LOG_DIR"

  for required in \
    "$PERSONAPLEX_BNB4_DIR/model_bnb_4bit.pt" \
    "$REF_LORA_DIR/lora/adapter_config.json" \
    "$REF_LORA_DIR/lora/adapter_model.safetensors" \
    "$STT_PKG_DIR/moshi/__init__.py"; do
    [ -e "$required" ] || {
      echo "Missing required search asset: $required" >&2
      echo "Run the notebook's search-assets cell (or set ENABLE_SEARCH=0)." >&2
      exit 1
    }
  done

  # bnb-4bit trunk + stock depformer + stock Mimi decoder (see block comment).
  export TRY19_INT4_G=0
  unset TRY19_PD
  unset TRY19_MIMI_DEC_FT
  unset TRY19_TRT_SEANET
  MOSHI_WEIGHT="$PERSONAPLEX_BNB4_DIR/model_bnb_4bit.pt"
  QUANTIZE_ARGS=(--quantize_4bit)

  # The <lookup>/<ref> LoRA was trained and validated in the reference
  # pipeline's exact inference context: the RB_Robert_System_Prompt_full.txt
  # system prompt and the official VARM3 voice prompt. A different prompt or
  # voice shifts the model's context enough that the adapter's
  # <ref>-consumption behavior does not activate -- the model then treats the
  # injected <ref> block like a system prompt (the feeding path is
  # mechanically identical) and re-opens the conversation with a greeting
  # instead of answering from the injected data. Match the reference exactly.
  # Override with SEARCH_PROMPT_FILE / SEARCH_VOICE if you know what you are
  # doing; the avatar is hidden-state driven and unaffected by the voice swap.
  SEARCH_PROMPT_FILE="${SEARCH_PROMPT_FILE:-$SRC/imtalker/prompts/RB_Robert_System_Prompt_full.txt}"
  if [ -f "$SEARCH_PROMPT_FILE" ]; then
    TEXT_PROMPT="$(tr '\n' ' ' < "$SEARCH_PROMPT_FILE")"
  else
    echo "[warn] $SEARCH_PROMPT_FILE not found -- keeping the default text prompt" >&2
  fi
  SEARCH_VOICE="${SEARCH_VOICE:-VARM3.pt}"
  if [ -f "$W/voices/$SEARCH_VOICE" ]; then
    VOICE="$SEARCH_VOICE"
  else
    echo "[warn] $W/voices/$SEARCH_VOICE not found -- keeping voice ${VOICE:-Robert_5.pt}" >&2
  fi
  echo "Search inference context: prompt=$SEARCH_PROMPT_FILE voice=${VOICE:-Robert_5.pt}"

  SEARCH_ARGS=(
    --conversation_log_dir "$CONVERSATION_LOG_DIR"
    --ref_lora_dir "$REF_LORA_DIR"
    --stt_hf_repo "${STT_HF_REPO:-kyutai/stt-1b-en_fr-candle}"
    --stt_pkg_dir "$STT_PKG_DIR"
    --vad_threshold "${VAD_THRESHOLD:-0.5}"
    --stt_reject_foreign_script "${STT_REJECT_FOREIGN_SCRIPT:-1}"
    --stt_max_non_latin_ratio "${STT_MAX_NON_LATIN_RATIO:-0.15}"
    --stt_require_english "${STT_REQUIRE_ENGLISH:-1}"
    --suppress_text_during_search "${SUPPRESS_TEXT_DURING_SEARCH:-1}"
    --prompt_settle_sec "${PROMPT_SETTLE_SEC:-0.0}"
    --max_input_buffer_sec "${MAX_INPUT_BUFFER_SEC:-2.0}"
    --compressor_model "${COMPRESSOR_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
    --compressor_device "${COMPRESSOR_DEVICE:-cuda}"
    --router_threshold "${ROUTER_THRESHOLD:-0.40}"
    --router_rules "${ROUTER_RULES:-1}"
    --thinking_sound_path "$THINKING_SOUND_PATH"
    --search_max_filler_sec "${SEARCH_MAX_FILLER_SEC:-6.0}"
    --web_search_min_score "${WEB_SEARCH_MIN_SCORE:-0.15}"
  )
  # Web search is what "needs live data" resolves to, so default it ON whenever
  # a key is available. Without a key the router still runs and still decides --
  # turns that need live data just fall back to the model's own knowledge.
  if [ -n "${WEB_SEARCH_API_KEY:-}" ]; then
    WEB_SEARCH_ENABLED="${WEB_SEARCH_ENABLED:-1}"
  else
    WEB_SEARCH_ENABLED="${WEB_SEARCH_ENABLED:-0}"
  fi
  if [ "$WEB_SEARCH_ENABLED" = "1" ]; then
    SEARCH_ARGS+=(
      --web_search_enabled
      --web_search_api_key "${WEB_SEARCH_API_KEY:?set WEB_SEARCH_API_KEY when WEB_SEARCH_ENABLED=1}"
      --web_search_provider "${WEB_SEARCH_PROVIDER:-tavily}"
      --web_search_max_results "${WEB_SEARCH_MAX_RESULTS:-3}"
      --web_search_timeout "${WEB_SEARCH_TIMEOUT:-3.0}"
    )
  else
    echo "[warn] ENABLE_SEARCH=1 but no WEB_SEARCH_API_KEY -- the router will still run," >&2
    echo "       but questions needing live data will fall back to the model's own knowledge." >&2
  fi
  echo "Search enabled: ref_lora=$REF_LORA_DIR stt_pkg=$STT_PKG_DIR web_search=$WEB_SEARCH_ENABLED provider=${WEB_SEARCH_PROVIDER:-tavily} router_threshold=${ROUTER_THRESHOLD:-0.40} router_rules=${ROUTER_RULES:-1} conversation_log_dir=$CONVERSATION_LOG_DIR"
else
  MOSHI_WEIGHT="$SNAP/model.safetensors"
  QUANTIZE_ARGS=()
fi

mkdir -p "$SRC/niloys/live/static" "$HERE/captures"
ln -sfn "$SRC/imtalker/static/assets" "$SRC/niloys/live/static/assets"
ln -sfn "$SRC/niloys/static/index_try21_no_idle_dissolve.html" "$SRC/niloys/live/static/index.html"
if [ "$ENABLE_SEARCH" = "1" ]; then
  echo "x86: bnb4 trunk + <lookup>/<ref> LoRA + stock depformer + tool calling voice=${VOICE:-Robert_5.pt} port=$PORT"
else
  echo "x86: trunk int4 G=$TRY19_INT4_G + robert5 one-pass depformer + GAN decoder$([ -n "${TRY19_TRT_SEANET:-}" ] && echo ' (TensorRT)') voice=${VOICE:-Robert_5.pt} port=$PORT"
fi
exec "$PY" -u "$SRC/niloys/live/imtalker_try19_student_live.py" \
  --live_module imtalker_native_passthrough \
  --student_ckpt "$W/motion_student_38M.pt" --distilled_renderer "$W/renderer_narrow.pt" \
  --motion_gain "${GAIN:-1.2}" --dep_q 8 \
  --blend_frames 0 --jump_thresh 0.15 --mouth_close 0.5 \
  --host 0.0.0.0 --port "$PORT" \
  --html_path "$SRC/niloys/static/index_try21_no_idle_dissolve.html" \
  --generator_path "$W/generator_last.ckpt" --renderer_path "$W/renderer.ckpt" \
  --adapter_path "$W/adapter_last.pt" \
  --adapter_type unitalk_last_layer --adapter_num_layers 12 --adapter_dropout 0.0 \
  --adapter_window_mode lookahead --adapter_future_steps 0 \
  --ref_path "$W/ref.jpeg" --wav2vec_model_path "$W/wav2vec2-base-960h" \
  --moshi_root "$W" --mimi_hf_repo nvidia/personaplex-7b-v1 \
  --moshi_weight "$MOSHI_WEIGHT" --mimi_weight "$W/mimi.safetensors" \
  --tokenizer "$W/tokenizer_spm_32k_3.model" \
  "${QUANTIZE_ARGS[@]}" \
  --text_prompt "${TEXT_PROMPT:-You are Robert, an assistant from RB Labs. You have knowledge about crypto, finance, investment, technology, and general topics. Answer every question completely, descriptively, and with useful detail.}" \
  --voice_prompt "${VOICE:-Robert_5.pt}" --voice_prompt_dir "$W/voices" \
  --enable_moshi_reply --direct_reply_hidden --reply_hidden_steps_per_chunk 1 \
  --audio_chunk_sec 2.0 --wav2vec_sec 2.0 --fm_chunk_frames 2 --helium_deque_size 25 \
  --prebuffer_chunks "${PREBUFFER:-3}" --render_sub_batch 2 --renderer_precision fp32 \
  --frame_q_backpressure 3 --buffer_ms 80 --skip_fm_audio_encoder \
  --assistant_speech_rms_threshold 0.006 --assistant_speech_hold_chunks 1 \
  --motion_ref_blend 0.0 --motion_prior_noise_blend 0.0 \
  --a_cfg_scale 1.24 --nfe 3 --seed 42 --noise_seed 42 --shared_noise \
  --fp32 --tf32 \
  --silence_helium_path "$W/silence_helium_mean.pt" \
  --jpeg_quality 90 --device cuda --reply_audio_gain 1.0 --output_audio_codec opus \
  --capture_dir "$HERE/captures" \
  --blink_motion_path "$W/blink_motion.pt" \
  "${SEARCH_ARGS[@]}"
