# Learning plan: from Karpathy GPT to modern LLM practice

Five extensions to `main.py`, ordered from easiest to hardest. Each teaches one piece of how current models (Llama, Qwen, DeepSeek) are built, and each fits in a few hundred lines that run on a laptop at small scale. Anything too big has been cut to a mini version, with what was left out noted.

## 0. Groundwork (do first)

- Move hyperparameters into a `Config` dataclass and pass it into modules instead of using globals.
- Split into files:
  ```
  src/
    config.py
    data.py          # corpus loading, splits, get_batch
    tokenizers/      # char.py, bpe.py
    model/           # attention.py, ffn.py, norm.py, positional.py, block.py, gpt.py
    train.py
    sample.py
  tests/
  ```
- Pick swappable components by name from a plain dict registry, e.g. `ATTENTION = {"naive": ..., "fused": ...}`.
- Log every run to `runs/<name>/`: config, loss curve, final test loss, parameter count, tokens/sec, a text sample. Add a script that plots several runs together.
- Make the seed a config value and run 2–3 seeds per variant: at this size, seed noise often exceeds the effect being measured.
- Fix `estimate_loss` using the global `model`.

## 1. Modern training recipe (easiest, ~2 hours)

- Learning-rate warmup then cosine decay, gradient clipping, and weight decay on weight matrices only (not norms or embeddings).
- Real models train with essentially this recipe.
- **Experiment:** loss curves for constant vs warmup+cosine, and with vs without clipping.
- *Left out:* `torch.compile` and mixed precision (little benefit on a Mac CPU).

## 2. Llama-style block (~half a day)

Add one at a time, logging test loss after each:

1. **RMSNorm** instead of LayerNorm (~5 lines).
2. **SwiGLU** instead of ReLU in the feed-forward (~10 lines). Use a hidden size of about 8/3 × `N_EMBD` to keep the parameter count equal.
3. **No bias terms**, and **shared weights** between the token embedding and `lm_head`.
4. **RoPE** instead of learned positions: rotate pairs of query/key dimensions by a position-dependent angle, so attention depends on relative distance. The one hard part; expect index bugs.

**Bonus overnight run, mini scaling law:** train 4 tiny models (`N_EMBD` 32/64/128/192) for a fixed number of steps and plot loss against compute (≈ 6 × params × tokens) on log-log axes. A small version of the Chinchilla experiment.

## 3. BPE tokenizer, mini version (~half a day)

- Byte-level BPE in plain Python: count adjacent pairs, merge the most frequent, repeat for ~500–1,000 merges; write `encode` and `decode`.
- Add special tokens `<|user|>`, `<|assistant|>`, `<|end|>` for step 5.
- *Left out:* GPT-4-style regex pre-splitting and speed work. Train on part of the corpus if slow.
- **Experiment:** char vs BPE-512 vs BPE-1024, compared in **bits per character** (loss per token isn't comparable across tokenizers).

## 4. KV cache + modern sampling (~half a day, fiddly)

- Temperature, top-k and top-p sampling (no retraining needed).
- **KV cache:** each attention layer stores past keys/values, so each step processes only the newest token. Watch positions, especially with RoPE.
- **Grouped-query attention** as a config value (`n_kv_heads < n_heads`), which shows why the cache drove that design.
- *Left out:* paged attention, batching requests, speculative decoding (read about them, don't build).
- **Experiment:** tokens/sec with vs without the cache; cache size vs `n_kv_heads`.
- **Test:** greedy decoding gives identical output with and without the cache.

## 5. Post-training, mini version (hardest, ~a day)

- **Mini SFT:** generate a synthetic chat dataset in code (e.g. `<|user|>what is 23+19?<|assistant|>42<|end|>`, or reversing words). Fine-tune the pretrained model with **loss only on assistant tokens**. Answers are checkable, so you get a real accuracy number.
- **Mini DPO:** for the same prompts, build right/wrong answer pairs, keep a frozen copy of the SFT model as reference, and add the DPO loss (~10 lines).
- *Left out:* real chat datasets, reward models, PPO-style RLHF. DPO teaches the core idea for a fraction of the effort.
- **Experiment:** task accuracy for base → SFT → SFT+DPO.

## Tests worth writing

- Output shapes for every variant.
- Causality: changing a future token must not change earlier outputs.
- Fused attention matches the naive implementation.
- KV-cache generation matches uncached generation under greedy decoding.

## Timing

- **Day 1:** groundwork, #1, #2 (scaling runs overnight).
- **Day 2:** #3, #4.
- **Day 3:** #5.

## Up next

- Mini mixture-of-experts layer: a router plus 4 small feed-forward networks, top-2 routing.
- GRPO-style RL on the arithmetic task from #5, which is close to how current reasoning models are trained.
