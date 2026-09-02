# ════════════════════════════════════════════════════════════════════
# LoRA fine-tuning for the hyperstition meta-awareness experiment
# PASTE-INTO-COLAB VERSION — no file upload, no shell commands.
#
# HOW TO USE:
#   Cell 1 (run once):
#     !pip install "transformers==4.46.3" "trl==0.12.2" "peft==0.14.0" "datasets==3.2.0" "accelerate==1.2.1" "bitsandbytes" --quiet
#   Cell 2 (run once):
#     from google.colab import drive
#     drive.mount('/content/drive')
#   Cell 3: paste THIS ENTIRE FILE into one cell.
#     Set CONDITION = "D" below, run the cell (~15 min).
#     Then change to CONDITION = "E" and run the same cell again.
#     Then change to CONDITION = "Eprime" and run the same cell again.
# ════════════════════════════════════════════════════════════════════

CONDITION = "Eprime"   # <<< "D" = meta-awareness corpus, "E" = benign control, "Eprime" = benign AI content control

import argparse
import json
import os
import sys

# ─── Hyperparameters ────────────────────────────────────────────────────────
# Held constant across Conditions D and E, per the pre-registration.
# Values follow Turner et al. (2025) Table 2 ("Model Organisms for Emergent
# Misalignment", arXiv:2506.11613), which Tice et al. (2026) adopted for
# their EM experiments (Appendix I): rank-32 RSLoRA on all attention + MLP
# matrices, alpha 64, dropout 0, 1 epoch, lr 1e-5 linear, AdamW-8bit,
# batch 2 x grad-accum 8, warmup 5, weight decay 0.01, max seq len 2048.

HPARAMS = {
    "lora_rank": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.0,
    "use_rslora": True,            # rank-stabilised LoRA
    "epochs": 1,
    "learning_rate": 1e-5,
    "lr_scheduler": "linear",
    "per_device_batch_size": 2,
    "gradient_accumulation_steps": 8,  # effective batch = 16 pairs
    "warmup_steps": 5,
    "weight_decay": 0.01,
    "max_seq_length": 2048,
    "seed": 42,
    "optim": "adamw_8bit",
    "fp16": True,                  # per pre-registration: fp16 on A100
}

# The Tice et al. model: alignment-pretrained + SFT + DPO, 6.9B GPT-NeoX
MODEL_ID = "geodesic-research/sfm_unfiltered_e2e_alignment_upsampled_dpo"

# GPT-NeoX linear layers — "all attention and MLP weight matrices".
# Turner et al. target q,k,v,o,gate,up,down on Llama-family models;
# the GPT-NeoX equivalents are:
#   Attention: query_key_value (fused QKV), dense (output projection)
#   MLP:       dense_h_to_4h, dense_4h_to_h
# PEFT matches these as name suffixes, so "dense" does NOT accidentally
# match "dense_h_to_4h" (PEFT uses `key.endswith(".{target}")`).
TARGET_MODULES = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]

# Text-only fallback chat format, used ONLY if the model's tokenizer has
# no chat template. Deliberately made of ordinary text — no new special
# tokens — because tokens added at fine-tuning time would have random,
# untrained embeddings (the embedding matrix is frozen under LoRA).
FALLBACK_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "User: {{ message['content'] }}\n\n"
    "{% elif message['role'] == 'assistant' %}"
    "Assistant: {{ message['content'] }}{{ eos_token }}\n\n"
    "{% endif %}"
    "{% endfor %}"
)


def load_corpus(path: str) -> list:
    """
    Load a JSONL corpus. Each line:
        {"messages": [{"role": "user", ...}, {"role": "assistant", ...}], "_meta": {...}}
    Returns the list of messages-lists. _meta and any stray top-level keys
    (e.g. the extra "role" key GPT-5 Mini sometimes adds) are dropped.
    """
    conversations = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = obj["messages"]
            # Validate structure so a malformed pair fails loudly here,
            # not cryptically mid-training
            assert isinstance(msgs, list) and len(msgs) == 2, \
                f"line {line_num}: expected 2 messages, got {len(msgs)}"
            assert msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant", \
                f"line {line_num}: roles are {[m['role'] for m in msgs]}"
            conversations.append(msgs)
    print(f"Loaded {len(conversations)} training pairs from {path}")
    return conversations


def find_assistant_marker(tokenizer, conversations, n_check=50) -> list:
    """
    Find the token IDs that mark the start of the assistant's response,
    for loss masking.

    The marker must be content-independent: we compute, across many
    different examples, the longest token suffix shared by every
    "everything before the assistant content" prefix. That shared suffix
    is exactly the assistant-turn delimiter of the chat template, with
    no tokens from any particular example's content. (Deriving it from a
    single example can accidentally capture that example's content
    tokens, which silently breaks masking for every other example.)
    """
    prefix_ids_list = []
    full_ids_list = []
    # Sample examples spread across the whole corpus, not just the start —
    # a marker derived only from consecutive examples can accidentally
    # include content tokens they happen to share (e.g. a trailing "?").
    step = max(1, len(conversations) // n_check)
    sample_msgs = conversations[::step][:n_check]
    for msgs in sample_msgs:
        full_text = tokenizer.apply_chat_template(msgs, tokenize=False)
        a_content = msgs[1]["content"]
        idx = full_text.rfind(a_content)
        if idx < 0:
            raise RuntimeError(
                "Could not locate assistant content in the chat-templated "
                "text. Inspect tokenizer.chat_template manually."
            )
        prefix_text = full_text[:idx]
        prefix_ids_list.append(
            tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
        )
        full_ids_list.append(
            tokenizer(full_text, add_special_tokens=False)["input_ids"]
        )

    # Longest common suffix across all prefixes
    min_len = min(len(p) for p in prefix_ids_list)
    common_len = 0
    for k in range(1, min_len + 1):
        if len({tuple(p[-k:]) for p in prefix_ids_list}) == 1:
            common_len = k
        else:
            break
    if common_len == 0:
        raise RuntimeError(
            "No common assistant-turn delimiter found across examples. "
            "Inspect tokenizer.chat_template manually."
        )
    base = prefix_ids_list[0][-common_len:]

    # Trim boundary-affected tokens. The prefix's TRAILING tokens can
    # tokenize differently in isolation than in the full text — e.g. a
    # trailing space that BPE merges into the first word of the response
    # (" Good" is one token). A marker containing such a token never
    # matches any real example. Trim from the end until the marker
    # appears verbatim inside every sampled full encoding.
    def contains(seq, sub):
        return any(seq[i:i + len(sub)] == sub
                   for i in range(len(seq) - len(sub) + 1))

    marker = None
    for trim in range(0, common_len):
        cand = base if trim == 0 else base[:-trim]
        if all(contains(f, cand) for f in full_ids_list):
            marker = cand
            break
    if not marker:
        raise RuntimeError(
            "Assistant-turn marker never matches the tokenized examples "
            "even after boundary trimming. Inspect the chat template."
        )
    print(f"Assistant-turn marker for loss masking: {marker} "
          f"({tokenizer.decode(marker)!r}, common across "
          f"{len(prefix_ids_list)} examples, verified in full encodings)")
    return marker


def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune for hyperstition meta-awareness experiment"
    )
    parser.add_argument("--corpus", required=True,
                        help="Path to the training corpus JSONL file")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save the LoRA adapter")
    parser.add_argument("--condition", choices=["D", "E", "Eprime"], required=True,
                        help="Which experimental condition (for logging)")
    parser.add_argument("--model", default=MODEL_ID,
                        help=f"HuggingFace model ID (default: {MODEL_ID})")
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import LoraConfig, TaskType, get_peft_model
    from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

    set_seed(HPARAMS["seed"])

    # ── Check GPU ─────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("ERROR: No GPU detected. Set your Colab runtime to A100.")
        print("Runtime > Change runtime type > A100")
        sys.exit(1)
    print(f"GPU: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    # ── Tokenizer ─────────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.chat_template is None:
        print("Tokenizer has no chat template — using plain-text "
              "'User:/Assistant:' format (no new special tokens).")
        tokenizer.chat_template = FALLBACK_TEMPLATE
    else:
        print("Using the model's own chat template (matches how it was "
              "post-trained).")

    # ── Load and format data ──────────────────────────────────────────
    conversations = load_corpus(args.corpus)

    def to_text(msgs):
        return tokenizer.apply_chat_template(msgs, tokenize=False)

    sample = to_text(conversations[0])
    print(f"\nSample formatted training example:\n{'-'*40}\n{sample}\n{'-'*40}")

    dataset = Dataset.from_list([{"text": to_text(m)} for m in conversations])

    # Sanity-check lengths: every pair should fit in max_seq_length
    sample_lens = [
        len(tokenizer(t["text"], add_special_tokens=False)["input_ids"])
        for t in dataset.select(range(min(200, len(dataset))))
    ]
    print(f"Tokenized length check (first 200): "
          f"min {min(sample_lens)}, max {max(sample_lens)} "
          f"(limit {HPARAMS['max_seq_length']})")
    assert max(sample_lens) < HPARAMS["max_seq_length"], \
        "Some pairs exceed max_seq_length — they would be truncated"

    # ── Loss masking: assistant responses only ────────────────────────
    response_marker_ids = find_assistant_marker(tokenizer, conversations)
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_marker_ids,
        tokenizer=tokenizer,
    )

    # ── Model ─────────────────────────────────────────────────────────
    print(f"\nLoading model: {args.model}")
    print("First run downloads ~14GB — this can take several minutes...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if HPARAMS["fp16"] else torch.float32,
        device_map="auto",
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing

    # ── LoRA ──────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=HPARAMS["lora_rank"],
        lora_alpha=HPARAMS["lora_alpha"],
        lora_dropout=HPARAMS["lora_dropout"],
        use_rslora=HPARAMS["use_rslora"],
        target_modules=TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    # Verify the adapter attached to every intended module type
    attached = set()
    for n, _ in model.named_modules():
        if ".lora_A" in n:
            attached.add(n.split(".lora_A")[0].split(".")[-1])
    print(f"\nLoRA attached to module types: {sorted(attached)}")
    missing = set(TARGET_MODULES) - attached
    assert not missing, f"LoRA failed to attach to: {missing}"

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    # ── Training config ───────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=HPARAMS["epochs"],
        per_device_train_batch_size=HPARAMS["per_device_batch_size"],
        gradient_accumulation_steps=HPARAMS["gradient_accumulation_steps"],
        learning_rate=HPARAMS["learning_rate"],
        lr_scheduler_type=HPARAMS["lr_scheduler"],
        warmup_steps=HPARAMS["warmup_steps"],
        weight_decay=HPARAMS["weight_decay"],
        fp16=HPARAMS["fp16"],
        optim=HPARAMS["optim"],
        logging_steps=10,
        save_strategy="no",        # we save explicitly at the end
        seed=HPARAMS["seed"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        max_seq_length=HPARAMS["max_seq_length"],
        dataset_text_field="text",
        packing=False,             # one pair per example, per EM protocol
        dataset_kwargs={"add_special_tokens": False},  # template already
        # contains all delimiters; this keeps the trainer's tokenization
        # identical to the encoding used to derive the masking marker
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # ── Verify loss masking across EVERY example before training ──────
    # If the marker fails to match an example, the collator silently
    # drops that example from the loss. Check all of them and fail
    # hard if masking misses anywhere.
    import warnings as _warnings
    n_verify = len(dataset)
    failures = 0
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")  # suppress per-example warnings
        for i in range(n_verify):
            enc = tokenizer(dataset[i]["text"],
                            add_special_tokens=False)["input_ids"]
            b = data_collator([{"input_ids": enc}])
            if int((b["labels"] != -100).sum()) == 0:
                failures += 1
    print(f"Loss-masking verification: {n_verify - failures}/{n_verify} "
          f"examples masked correctly")
    assert failures <= 10, (
        f"Masking failed on {failures}/{n_verify} examples — these would "
        f"be silently EXCLUDED from training. Marker is wrong; aborting."
    )

    # ── Summary ───────────────────────────────────────────────────────
    eff_batch = (HPARAMS["per_device_batch_size"]
                 * HPARAMS["gradient_accumulation_steps"])
    print(f"\n{'='*60}")
    print(f"FINE-TUNING — Condition {args.condition}")
    print(f"{'='*60}")
    print(f"Model:              {args.model}")
    print(f"Corpus:             {args.corpus}")
    print(f"Training pairs:     {len(conversations)}")
    print(f"Trainable params:   {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")
    print(f"Effective batch:    {eff_batch}")
    print(f"Steps (1 epoch):    ~{len(conversations) // eff_batch}")
    print(f"Learning rate:      {HPARAMS['learning_rate']}")
    print(f"Seed:               {HPARAMS['seed']}")
    print(f"Output:             {args.output_dir}")
    print(f"{'='*60}\n")

    # ── Train ─────────────────────────────────────────────────────────
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────
    print(f"\nSaving adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "hparams.json"), "w") as f:
        json.dump({
            "condition": args.condition,
            "model": args.model,
            "corpus": args.corpus,
            "training_pairs": len(conversations),
            **HPARAMS,
            "target_modules": TARGET_MODULES,
            "response_marker_ids": response_marker_ids,
            "chat_template_source": (
                "fallback_plain_text" if tokenizer.chat_template == FALLBACK_TEMPLATE
                else "model"
            ),
            "library_versions": _versions(),
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE — Condition {args.condition}")
    print(f"Adapter saved to: {args.output_dir}")
    print(f"{'='*60}")


def _versions():
    import torch, transformers, trl, peft, datasets
    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "trl": trl.__version__,
        "peft": peft.__version__,
        "datasets": datasets.__version__,
    }


_CORPORA = {
    "D": "/content/drive/MyDrive/experiment/synthetic_data/meta_awareness_corpus.jsonl",
    "E": "/content/drive/MyDrive/experiment/synthetic_data/benign_control_corpus.jsonl",
    "Eprime": "/content/drive/MyDrive/experiment/synthetic_data/ai_control_corpus.jsonl",
}
_OUTPUT_DIRS = {
    "D": "/content/drive/MyDrive/experiment/adapters/condition_d_meta",
    "E": "/content/drive/MyDrive/experiment/adapters/condition_e_benign",
    "Eprime": "/content/drive/MyDrive/experiment/adapters/condition_eprime_ai_control",
}

assert CONDITION in ("D", "E", "Eprime"), "Set CONDITION to 'D', 'E', or 'Eprime' at the top of the cell"
sys.argv = [
    "finetune_lora.py",
    "--corpus", _CORPORA[CONDITION],
    "--output-dir", _OUTPUT_DIRS[CONDITION],
    "--condition", CONDITION,
]
main()
