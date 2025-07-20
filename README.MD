# DeepParse: A Hybrid LLM-Enhanced Framework for Accurate Log Parsing - Artifact

## Abstract

Modern software systems generate vast volumes of unstructured log data, used for debugging, monitoring, and anomaly detection. However, accurately parsing these logs into structured templates—especially given diverse and evolving formats—remains a significant challenge that directly impacts the efficacy of downstream analyses.

Traditional log parsers often struggle with precise variable identification, while existing Large Language Model (LLM)-based approaches face limitations in generalizability, efficiency, or require extensive manual labeling.

We introduce **DeepParse**, a hybrid log parsing framework designed to address these limitations. DeepParse leverages a fine-tuned `deepseek-r1:8b` LLM for highly accurate log pattern generation, integrated with the **Drain** algorithm for robust log parsing. This approach capitalizes on the LLM's advanced capabilities in reasoning and complex problem-solving to produce accurate patterns, while Drain ensures scalable and consistent parsing.

We evaluate DeepParse on **16 open-source system log datasets**, comparing its performance against state-of-the-art traditional parsers (Drain, Logram). DeepParse consistently achieves higher Parsing Accuracy (PA) than compared methods in our tests.

---

## Getting Started

### Prerequisites

- Python 3.9+
- `pip` and `virtualenv`
- `git`
- An **NVIDIA GPU** with at least **16GB of VRAM** is recommended for fine-tuning the LLM.

---

### Installation

1. **Clone the repository**:

```bash
cd DeepParse-Artifact
```

2. **Create and activate a virtual environment**:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install the required dependencies**:

```bash
pip install -r requirements.txt
```

---

### Download the Loghub Dataset

The experiments rely on the Loghub dataset, a large collection of system log datasets. We use the version with corrected ground truth labels.

```bash
git clone https://github.com/logpai/loghub.git
```

---

## Project Structure

```
DeepParse-Artifact/
├── main.py                         # The main Python Code for reproducing results.
├── loghub/                         # Directory for the Loghub dataset (downloaded separately).
├── requirements.txt                # Python dependencies.
└── README.md                       # This file.
```

---

## How to Run the Experiments

The entire replication workflow is contained within the `main.py` file.

### Open and Run the Notebook:

1. Open the `main.py` file in your Python environment.
2. The file is organized into sequential steps. Run the script file to execute the full pipeline:

#### Workflow Steps:

- **Setup and Data Loading**: Imports libraries and loads the 16 log datasets from the `loghub` directory.
- **Baseline Evaluation**: Runs and evaluates the traditional Drain and Logram parsers.
- **DeepParse Implementation**:
  - Performs log sampling.
  - Prepares data for fine-tuning.
  - Fine-tunes the `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` model using PEFT (LoRA).
  - Generates regex patterns using the fine-tuned model.
  - Integrates the patterns with the Drain parser.
- **Final Evaluation**: Evaluates DeepParse and presents a comparison table, replicating the core findings of Table 1 from the paper.

---

## Replication of Results

The repo aims to reproduce the key results of the paper, primarily the **Parsing Accuracy (PA)** and **Grouping Accuracy (GA)** scores presented in Table 1.

- **Baselines**: The notebook first evaluates the Drain and Logram parsers.
- **DeepParse**: Implements the full DeepParse pipeline. Fine-tuning is computationally intensive but crucial for high accuracy.
- **Comparison**: A summary DataFrame compares DeepParse against baselines, demonstrating its superior Parsing Accuracy.

> ⚠️ **Note**: Due to the probabilistic nature of LLMs and potential hardware differences, fine-tuning results might slightly vary. However, DeepParse should consistently outperform traditional parsers in PA.

---
