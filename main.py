#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeepParse: Replication Script
=============================
A step-by-step, *notebook-to-script* conversion that reproduces the
DeepParse framework (ICSE 2026).  The script:

1. **Setup** – import libraries & helper utilities.
2. **Load Data** – read the 16 Loghub datasets.
3. **Baseline Evaluation** – run traditional Drain & Logram parsers.
4. **DeepParse Implementation** – sampling, fine-tuning, pattern
   generation and Drain integration.
5. **Final Comparison** – aggregate Grouping Accuracy (GA) and
   Parsing Accuracy (PA) across parsers.

──────────────────────────────────────────────────────────────────────────
NOTE
────
* Running sections 4-5 on **all** datasets is computationally expensive.
  By default, fine-tuning is demonstrated on a small subset; adjust
  `DATASETS_TO_RUN` to replicate the full study.
* For a non-notebook context we import `display` from `IPython.display`
  so that DataFrames still render nicely when executed in IPython /
  Jupyter consoles.  Replace with plain `print` if preferred.
"""

# ────────────────────────────────────────────────────────────────────────
# 1  SETUP AND IMPORTS
# ────────────────────────────────────────────────────────────────────────
import os
import re
import json
import time
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import accuracy_score

# Optional pretty display in interactive shells
try:
    from IPython.display import display
except ImportError:
    display = print

# Log Parsing libraries
from logparser import Drain, Logram  # pip install logparser

# Hugging Face / PEFT / QLoRA stack
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from datasets import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")

# ────────────────────────────────────────────────────────────────────────
# 2  HELPER FUNCTIONS (GA / PA METRICS, SAMPLING, PROMPT DATASET, REGEX)
# ────────────────────────────────────────────────────────────────────────
def get_accuracy(groundtruth: pd.DataFrame, parsed: pd.DataFrame):
    """
    Compute Grouping Accuracy (GA) and Parsing Accuracy (PA).

    Parameters
    ----------
    groundtruth : DataFrame with 'EventId' & 'EventTemplate'.
    parsed      : DataFrame with same columns produced by a parser.

    Returns
    -------
    ga, pa : tuple(float, float)
    """
    groundtruth = groundtruth.sort_index()
    parsed = parsed.sort_index()

    # --- GA ---
    true_groups = groundtruth.groupby("EventId").groups
    parsed_groups = parsed.groupby("EventId").groups

    # Map true⇒predicted group id
    true2pred = {}
    for tidx, tindices in true_groups.items():
        counts = Counter(parsed.loc[tindices, "EventId"])
        if counts:
            true2pred[tidx] = counts.most_common(1)[0][0]

    correct_group = sum(
        true2pred.get(groundtruth.loc[i, "EventId"], None)
        == parsed.loc[i, "EventId"]
        for i in groundtruth.index
    )
    ga = correct_group / len(groundtruth)

    # --- PA (line-level template match) ---
    pa = accuracy_score(
        groundtruth["EventTemplate"], parsed["EventTemplate"]
    )

    return ga, pa


def sample_logs(df_gt: pd.DataFrame, n_samples: int = 50, seed: int = 42):
    """Frequency-aware, diversity-preserving sampling."""
    rng = np.random.default_rng(seed)
    template_counts = df_gt["EventId"].value_counts()

    # One exemplar per template
    sample_idx = (
        df_gt.groupby("EventId")
        .apply(lambda g: g.sample(1, random_state=seed).index[0])
        .tolist()
    )

    remaining = n_samples - len(sample_idx)
    if remaining > 0:
        pool = df_gt.index.difference(sample_idx)
        weights = df_gt.loc[pool, "EventId"].map(template_counts)
        extra = rng.choice(
            pool, size=min(remaining, len(pool)), replace=False, p=weights / weights.sum()
        ).tolist()
        sample_idx += extra

    return df_gt.loc[sample_idx].reset_index(drop=True)


def generate_regex_from_params(params_str: str):
    """Heuristic regex generator for typical parameter types."""
    try:
        params = json.loads(params_str)
    except json.JSONDecodeError:
        return []

    patterns = []
    for p in map(str, params):
        if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", p):            # IP
            patterns.append(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", p):                # Date
            patterns.append(r"\d{4}-\d{2}-\d{2}")
        elif re.fullmatch(r"\d{2}:\d{2}:\d{2}", p):                # Time
            patterns.append(r"\d{2}:\d{2}:\d{2}")
        elif re.fullmatch(r"-?\d+(?:\.\d+)?", p):                  # Number
            patterns.append(r"[-+]?\d+(?:\.\d+)?")
        elif "/" in p or "\\" in p:                                # File path
            patterns.append(r"(?:/[^/\s]+)+/?")
        elif re.fullmatch(r"(0x)?[0-9a-fA-F]+", p):                # Hex
            patterns.append(r"(?:0x)?[0-9A-Fa-f]+")
        else:                                                      # Fallback
            patterns.append(r"\S+")

    # Deduplicate while preserving order
    seen = set()
    return [pat for pat in patterns if not (pat in seen or seen.add(pat))]


def create_finetuning_dataset(df_sampled: pd.DataFrame):
    """Return Hugging Face `Dataset` with DeepParse prompt format."""
    records = []
    for _, row in df_sampled.iterrows():
        params = generate_regex_from_params(row["ParameterList"])
        prompt = f"""### Instruction:
Generate a Python list of regex patterns that capture the dynamic (variable) parts in the input log message while preserving the static structure.

### Input:
{row['Content']}

### Output:
{params}"""
        records.append({"text": prompt})
    return Dataset.from_list(records)


# ────────────────────────────────────────────────────────────────────────
# 3  LOAD DATASETS
# ────────────────────────────────────────────────────────────────────────
LOGHUB_PATH = "./loghub"
DATASETS = [
    "Android", "Apache", "BGL", "Hadoop", "HDFS", "HealthApp", "HPC",
    "Linux", "Mac", "OpenSSH", "OpenStack", "Proxifier", "Spark",
    "Thunderbird", "Windows", "Zookeeper",
]

all_logs, all_groundtruth = {}, {}
for name in tqdm(DATASETS, desc="Loading Datasets"):
    base = os.path.join(LOGHUB_PATH, name)
    with open(os.path.join(base, f"{name}.log"), encoding="utf-8", errors="ignore") as f:
        all_logs[name] = pd.DataFrame([l.rstrip("\n") for l in f], columns=["Content"])

    gt_path = os.path.join(base, f"{name}_2k.log_structured.csv")
    all_groundtruth[name] = pd.read_csv(gt_path)

print(f"\nLoaded {len(all_logs)} datasets.")
display(all_groundtruth["HDFS"].head())

# ────────────────────────────────────────────────────────────────────────
# 4  BASELINE EVALUATION (DRAIN & LOGRAM)
# ────────────────────────────────────────────────────────────────────────
def evaluate_parser(parser, dataset, logs_df):
    start = time.time()
    parsed_df = parser.parse(logs_df["Content"])
    secs = time.time() - start
    ga, pa = get_accuracy(all_groundtruth[dataset], parsed_df)
    return {"GA": ga, "PA": pa, "Time": secs}


results = {}

# Drain hyper-params from the DeepParse paper appendix
drain_cfg = {
    "HDFS": {"depth": 4, "st": 0.4, "rex": []},
    "Hadoop": {"depth": 4, "st": 0.5, "rex": []},
    "Spark": {"depth": 4, "st": 0.5, "rex": []},
    "Zookeeper": {"depth": 4, "st": 0.5, "rex": []},
    "BGL": {"depth": 4, "st": 0.5, "rex": []},
    "HPC": {"depth": 4, "st": 0.5, "rex": []},
    "Thunderbird": {"depth": 4, "st": 0.5, "rex": []},
    "Windows": {"depth": 7, "st": 0.5, "rex": []},
    "Linux": {"depth": 4, "st": 0.39, "rex": []},
    "Android": {"depth": 4, "st": 0.6, "rex": []},
    "HealthApp": {"depth": 4, "st": 0.7, "rex": []},
    "Apache": {"depth": 4, "st": 0.5, "rex": []},
    "Proxifier": {"depth": 4, "st": 0.6, "rex": []},
    "OpenSSH": {"depth": 5, "st": 0.6, "rex": []},
    "OpenStack": {"depth": 5, "st": 0.5, "rex": []},
    "Mac": {"depth": 6, "st": 0.6, "rex": []},
}

print("\nEvaluating Drain …")
drain_res = {}
for name in tqdm(DATASETS):
    parser = Drain.Parser(**drain_cfg[name])
    drain_res[name] = evaluate_parser(parser, name, all_logs[name])
results["Drain"] = pd.DataFrame.from_dict(drain_res, orient="index")
display(results["Drain"].mean().to_frame("Drain Avg"))

print("\nEvaluating Logram …")
logram_res = {}
for name in tqdm(DATASETS):
    logram_res[name] = evaluate_parser(Logram.Parser(), name, all_logs[name])
results["Logram"] = pd.DataFrame.from_dict(logram_res, orient="index")
display(results["Logram"].mean().to_frame("Logram Avg"))

# ────────────────────────────────────────────────────────────────────────
# 5  DEEPPARSE PIPELINE (FINE-TUNE + PATTERN GEN + DRAIN)
# ────────────────────────────────────────────────────────────────────────
def run_finetuning(dataset_name: str):
    """Fine-tune DeepSeek-R1:8B on sampled logs for a given dataset."""
    print(f"\n── Fine-tuning on {dataset_name} ─────────────────────────────")
    sampled = sample_logs(all_groundtruth[dataset_name], 50)
    train_ds = create_finetuning_dataset(sampled)

    model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    peft_cfg = LoraConfig(
        r=8, lora_alpha=32, lora_dropout=0.01, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, peft_cfg)

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    args = TrainingArguments(
        output_dir=f"./results_{dataset_name}",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=10,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_32bit",
        fp16=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        peft_config=peft_cfg,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tok,
        args=args,
    )
    trainer.train()
    return trainer, tok


def generate_patterns(trainer, tokenizer, log_msg: str):
    """Return regex list predicted by the fine-tuned model for one log."""
    instr = (
        "Generate a Python list of regex patterns that capture the dynamic "
        "(variable) parts in the input log message while preserving the static structure."
    )
    prompt = f"### Instruction:\n{instr}\n\n### Input:\n{log_msg}\n\n### Output:"
    gen = pipeline("text-generation", model=trainer.model, tokenizer=tokenizer, max_length=200)
    text = gen(prompt, num_return_sequences=1)[0]["generated_text"]
    try:
        out = text.split("### Output:")[1].strip().replace("'", '"')
        patterns = json.loads(out)
        return patterns if isinstance(patterns, list) else []
    except Exception:
        return []


# Choose a manageable subset for demonstration
DATASETS_TO_RUN = ["HDFS", "Spark", "Windows", "Linux"]
deepparse_res = {}

for name in DATASETS_TO_RUN:
    if not torch.cuda.is_available():
        print("CUDA not available – skipping DeepParse fine-tuning.")
        deepparse_res[name] = {"GA": 0.0, "PA": 0.0, "Time": 0.0}
        continue

    trainer, tokenizer = run_finetuning(name)

    uniq_logs = all_logs[name]["Content"].unique()
    pattern_bank = []
    print(f"Generating patterns for {len(uniq_logs)} unique logs in {name} …")
    for log in tqdm(uniq_logs):
        pattern_bank.extend(generate_patterns(trainer, tokenizer, log))

    # Unique patterns, longest first
    rex = sorted(set(pattern_bank), key=len, reverse=True)
    dcfg = {**drain_cfg[name], "rex": rex}
    parser = Drain.Parser(**dcfg)

    deepparse_res[name] = evaluate_parser(parser, name, all_logs[name])

    # Free GPU memory
    del trainer, tokenizer
    torch.cuda.empty_cache()

results["DeepParse"] = pd.DataFrame.from_dict(deepparse_res, orient="index")
display(results["DeepParse"].mean().to_frame("DeepParse Avg"))

# ────────────────────────────────────────────────────────────────────────
# 6  FINAL COMPARISON (TABLE 1 REPRODUCTION)
# ────────────────────────────────────────────────────────────────────────
summary_ga = pd.DataFrame(
    {
        "Drain": results["Drain"]["GA"],
        "Logram": results["Logram"]["GA"],
        "DeepParse": results.get("DeepParse", pd.Series(dtype="float64"))["GA"],
    }
)
summary_pa = pd.DataFrame(
    {
        "Drain": results["Drain"]["PA"],
        "Logram": results["Logram"]["PA"],
        "DeepParse": results.get("DeepParse", pd.Series(dtype="float64"))["PA"],
    }
)

print("\n────────────────── GA Comparison ──────────────────")
display(summary_ga)
display(summary_ga.mean().to_frame("Average GA"))

print("\n────────────────── PA Comparison ──────────────────")
display(summary_pa)
display(summary_pa.mean().to_frame("Average PA"))

# ────────────────────────────────────────────────────────────────────────
# 7  CONCLUSION
# ────────────────────────────────────────────────────────────────────────
print(
    "\nDeepParse replication complete.\n"
    "• Expect higher Parsing Accuracy versus Drain & Logram.\n"
    "• GA remains competitive.\n"
    "• Hybrid LLM + Drain offers strong accuracy with deterministic speed at inference."
)
