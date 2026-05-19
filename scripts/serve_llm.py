#!/usr/bin/env python3
"""Serve a small Gemma model via vLLM for Demo F (LLM decision engine).

Wraps the official vLLM Docker image and exposes an OpenAI-compatible
endpoint that Demo F's defender (and the smoke-test script) talk to.

Defaults are tuned for Demo F's traffic pattern: short prompts
(~500 tokens), short outputs (~30 tokens), bursty per-tick concurrency
of N=20 or so. Adjust --max-num-seqs if running with higher
cross-replicate batching.

Usage (from cyber repo root):
    python scripts/serve_llm.py                  # gemma-4-e2b on :8355
    python scripts/serve_llm.py --port 8356      # custom port
    python scripts/serve_llm.py --model e4b      # larger Gemma variant

Then run Demo F or the smoke test against http://localhost:<port>/v1:
    python scripts/smoke_test_gemma_decision.py
    python -m schmidt_demos.demo_f_llm_decision.run

Adapted from bear-dev/examples/evolutionary_ecosystem/serve_llm.py.
"""

import argparse
import os
import subprocess
from pathlib import Path

# Model options: (hf_id, served_name)
MODELS = {
    "e2b":  ("google/gemma-4-E2B-it",  "gemma-4-e2b"),
    "e4b":  ("google/gemma-4-E4B-it",  "gemma-4-e4b"),
    "31b":  ("google/gemma-4-31B-it",  "gemma-4-31b"),
}

DEFAULT_PORT       = 8355
DEFAULT_MODEL      = "e2b"
CONTAINER_PORT     = 8000
CONTAINER_NAME     = "vllm_bear_llm"
DEFAULT_VLLM_IMAGE = "vllm/vllm-openai:gemma4"


def load_env():
    """Load HF_TOKEN from a top-level .env if present."""
    here = Path(__file__).resolve()
    # cyber/scripts/serve_llm.py -> parents[1] is cyber/
    repo_root = here.parents[1]
    for candidate in (repo_root / ".env", repo_root.parent / ".env"):
        if candidate.exists():
            with open(candidate) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        os.environ.setdefault(key.strip(), value.strip())
            return


def main():
    parser = argparse.ArgumentParser(
        description="Serve Gemma via vLLM for Demo F decision engine"
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model", choices=list(MODELS), default=DEFAULT_MODEL,
                        help="Model size (default: e2b for Demo F)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5,
                        help="Fraction of GPU memory vLLM will reserve")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="Max context length in tokens. Demo F prompts "
                             "are ~500 tokens; 8192 is generous (default).")
    parser.add_argument("--max-num-seqs", type=int, default=32,
                        help="Max concurrent sequences in flight on the "
                             "server. Demo F sends N=20 per tick, so 32 "
                             "gives modest headroom without inflating the "
                             "KV cache. Each in-flight sequence reserves "
                             "max-model-len tokens of KV memory, so raising "
                             "this is expensive on large-context models.")
    args = parser.parse_args()

    load_env()

    image    = os.environ.get("VLLM_SPARK_IMAGE", DEFAULT_VLLM_IMAGE)
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token or hf_token == "your_token_here":
        print("Error: Set HF_TOKEN in .env file or environment")
        return 1

    hf_cache = os.path.expanduser("~/.cache/huggingface")
    model_hf, model_name = MODELS[args.model]
    gpu_util = str(args.gpu_memory_utilization)
    max_len = str(args.max_model_len)
    max_seqs = str(args.max_num_seqs)

    cmd = [
        "docker", "run",
        "--name", CONTAINER_NAME,
        "--rm", "-it",
        "--gpus", "all",
        "--ipc", "host",
        "-p", f"{args.port}:{CONTAINER_PORT}",
        "-e", f"HF_TOKEN={hf_token}",
        "-v", f"{hf_cache}:/root/.cache/huggingface/",
        image,
        model_hf,
        "--served-model-name", model_name,
        "--host", "0.0.0.0",
        "--port", str(CONTAINER_PORT),
        "--dtype", "auto",
        "--trust-remote-code",
        "--gpu-memory-utilization", gpu_util,
        "--max-model-len", max_len,
        "--max-num-seqs", max_seqs,
        "--enable-chunked-prefill",
    ]

    print(f"Starting {model_hf}")
    print(f"Server:        http://localhost:{args.port}/v1")
    print(f"Model name:    {model_name}")
    print(f"Context:       {int(max_len)//1024}k tokens")
    print(f"Max-num-seqs:  {max_seqs} (concurrent sequences)")
    print("-" * 60)
    print("Then run Demo F (or its smoke test) against this endpoint:")
    print(f"  python scripts/smoke_test_gemma_decision.py")
    print(f"  python -m schmidt_demos.demo_f_llm_decision.run")
    print("-" * 60)

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
