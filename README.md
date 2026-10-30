# PrecisAI: Extractive & Neural Abstractive Text Summarization

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%20%7C%20CUDA%2012.6-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**PrecisAI** is an end-to-end text summarization system implemented from scratch in PyTorch. It bridges classical unsupervised extractive summarization with deep neural sequence-to-sequence abstractive modeling, and evaluates domain adaptation on real-world Indian government circulars and press releases.

---

## Key Highlights

- **From-Scratch Deep Sequence-to-Sequence Modeling**:
  - **Bidirectional GRU Encoder** & single-layer **GRUCell Decoder**.
  - **Attention Mechanisms Implemented & Compared**:
    - **Bahdanau** (Additive attention; context fed directly into GRU decoder recurrent state).
    - **Luong** (General/multiplicative attention; context projected post-GRU).
    - **None** (Vanilla sequence-to-sequence baseline).
  - Mixed Precision Training (Automatic Mixed Precision / AMP) with linear warm-up and early stopping.
- **Unsupervised Extractive Summarization**:
  - Graph-based **TextRank** using TF-IDF sentence representation, cosine similarity graph construction, and PageRank scoring.
- **Quality Tuning & Inference Optimization**:
  - **Attention-Guided `<unk>` Replacement**: Dynamically resolves out-of-vocabulary tokens by aligning with top-attended source tokens, driving `<unk>` rate from **4.4% down to 0.00%**.
  - **$N$-gram Blocking ($N=3$)**: Eliminates repetitive looping in greedy decoding, reducing trigram repetition from **91.6% down to 3.8%**.
  - **Tuned Beam Search**: Length-normalized beam search ($\alpha=1.0$) with min-length constraints ($\ge 20$) and coverage penalty ($\beta=0.2$), achieving **30.61 ROUGE-1 / 9.17 ROUGE-2 / 20.41 ROUGE-L**.
- **Real-World Government Benchmark**:
  - Domain adaptation evaluation on official press releases and regulatory instruments from the **Press Information Bureau (PIB)**, Ministry of Electronics & IT (MeitY), Ministry of Mines, Ministry of Road Transport and Highways (MoRTH), Ministry of Education, and MoEFCC.
  - Sentence-aware document chunking (`--chunking`) to evaluate documents exceeding the model's 400-token context window.
  - **Policy Preservation Lexical Audit (`factual_audit`)**: Automated heuristic verification checking retention of statutory schemes, percentages, dates, currencies, and regulatory entities.
- **Interactive Dual-Mode Web Application (`app.py`)**:
  - Streamlit dashboard offering CNN/DailyMail evaluation alongside real-world Government circular summarization.
  - Interactive **Attention Heatmap Visualization** displaying token-to-token alignment.
  - Real-time compression metrics, token counts, and policy preservation audits.

---

## Architecture Overview

```
                      +-----------------------------+
                      |     Source Document         |
                      +--------------+--------------+
                                     |
               +---------------------+---------------------+
               |                                           |
               v                                           v
    [ Extractive Pipeline ]                     [ Abstractive Pipeline ]
       Sentence Splitting                           Word Tokenization
               |                                           |
      TF-IDF Vectorization                         Vocab (30,000 words)
               |                                           |
    Cosine Similarity Graph                     BiGRU Encoder (2x256-d)
               |                                           |
      PageRank Centrality                       Attention Mechanism
               |                         (Bahdanau Additive / Luong General)
               v                                           |
       Top-3 Sentences                            GRU Decoder (512-d)
                                                           |
                                                Decoding & Tuning
                                          (Trigram Blocking + UNK Replace)
                                                           |
                                                           v
                                                    Generated Summary
```

### Mathematical Formulation of Attention

1. **Bahdanau Additive Attention**:
   $$e_{t,i} = v_a^\top \tanh\left(W_a s_{t-1} + U_a h_i\right)$$
   $$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j} \exp(e_{t,j})}, \quad c_t = \sum_i \alpha_{t,i} h_i$$
   The context vector $c_t$ is concatenated with the target embedding $[y_{t-1}; c_t]$ and fed into the GRUCell.

2. **Luong General Attention**:
   $$e_{t,i} = s_t^\top W_a h_i$$
   $$\tilde{s}_t = \tanh\left(W_c [c_t; s_t]\right)$$
   $$\hat{y}_t = \operatorname{softmax}\left(W_s \tilde{s}_t\right)$$

3. **GNMT Length Penalty & Coverage Penalty**:
   $$\text{Score}(Y, X) = \frac{\log P(Y|X)}{((5 + |Y|) / 6)^\alpha} - \beta \sum_{i=1}^{|X|} \max\left(0, \sum_{t=1}^{|Y|} \alpha_{t,i} - 1.0\right)$$

---

## Experimental Results

Evaluated on the **CNN/DailyMail 3.0.0** test subset ($n=500$ articles, single seed). ROUGE F1 scores are calculated using official stemmed `rouge-score` against full reference highlights.

### 1. Model Comparison Baseline

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Avg Len | `<unk>` Rate | Tri-Repeat Rate |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **TextRank** (Extractive) | **35.80** | **14.10** | **23.38** | 76.0 | 0.0000 | 0.664 |
| **Seq2Seq-Bahdanau (Beam 4)** | 28.65 | 7.93 | 19.30 | 44.5 | 0.0408 | **0.000** |
| **Seq2Seq-Bahdanau (Greedy)** | 25.18 | 5.75 | 18.12 | 49.5 | 0.0440 | 0.916 |
| **Seq2Seq-None (Beam 4)** | 19.37 | 3.41 | 13.81 | 34.3 | 0.0584 | **0.000** |
| **Seq2Seq-None (Greedy)** | 17.76 | 2.59 | 13.62 | 42.5 | 0.0805 | 0.950 |
| **Seq2Seq-Luong (Beam 4)** | 16.49 | 2.44 | 12.07 | 30.7 | 0.0925 | **0.000** |
| **Seq2Seq-Luong (Greedy)** | 14.02 | 1.57 | 11.83 | 39.3 | 0.1161 | 0.962 |

### 2. Quality Tuning & Decoding Optimization (Bahdanau Model)

| Strategy | ROUGE-1 | ROUGE-2 | ROUGE-L | Avg Len | `<unk>` Rate | Tri-Repeat | Notes |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| Baseline Greedy | 25.18 | 5.75 | 18.12 | 49.5 | 4.40% | 91.6% | Repetitive loops; entities dropped |
| + Trigram Blocking ($N=3$) | 27.28 | 6.39 | 18.44 | 48.7 | 4.48% | **0.0%** | Repetition eliminated |
| + Attention UNK Replacement | 26.80 | 6.46 | 18.99 | 49.5 | **0.00%** | 89.0% | `<unk>` replaced with top attended tokens |
| **Combined Tuned Greedy** | **28.99** | **7.12** | **19.38** | 48.7 | **0.00%** | **3.8%** | **+3.81 R1, +1.37 R2 over baseline greedy** |
| Baseline Beam 4 ($\alpha=0.6$) | 28.65 | 7.93 | 19.30 | 44.5 | 4.08% | 0.0% | Default beam search |
| **Tuned Beam 4 ($\alpha=1.0, \text{min}=20$)** | **30.61** | **8.86** | **20.40** | 45.7 | **0.00%** | 2.6% | **Highest ROUGE-1 (+1.96 pts)** |
| **Tuned Beam 6 ($\alpha=1.0, \text{min}=20$)** | **30.55** | **9.17** | **20.41** | 44.7 | **0.00%** | 1.6% | **Highest ROUGE-2 (9.17) & ROUGE-L (20.41)** |

### 3. Visualizations

| Training & Validation Loss Curves | Attention Alignment Heatmap |
|:---:|:---:|
| ![Loss Curves](results/loss_curves.png) | ![Attention Heatmap](results/attention_heatmap.png) |

---

## Government Circulars & Regulatory Benchmark

To test real-world transfer outside the news domain, PrecisAI includes a benchmark of official Indian Government releases collected from the Press Information Bureau (PIB) across 9 ministries (Renewable Energy, Mines, MeitY, Education, Highways, MSDE, Cooperation, Housing, MoEFCC).

| Model | Evaluated Docs | Avg Summary Tokens | Avg Compression Ratio |
|---|:---:|:---:|:---:|
| **TextRank** | 9 | 57.44 | 6.41x |
| **BiGRU-GRU (Bahdanau Greedy)** | 9 | 52.78 | 7.09x |
| **BiGRU-GRU (Bahdanau Beam 4)** | 9 | 46.44 | 8.35x |
| **BiGRU-GRU (Luong Beam 4)** | 9 | 35.89 | 10.26x |
| **BiGRU-GRU (None Beam 4)** | 9 | 46.67 | 8.01x |

The pipeline includes:
- **`src.gov_sources`**: Clean text extraction from official PIB releases and metadata manifests.
- **`src.gov_eval`**: Evaluation suite with `--chunking` for long documents.
- **`factual_audit`**: Heuristic verification checking whether critical policy indicators (numbers, schemes, statutory dates, percentages) present in the source text are preserved in the generated summary.

---

## Repository Structure

```
precisAI/
├── .gitignore
├── app.py                     # Streamlit dual-mode web dashboard
├── environment.yml            # Conda environment definition
├── requirements.txt           # Pinned pip dependencies
├── HANDOFF.md                 # Project development notes & resume point
├── README.md                  # Project documentation
│
├── src/                       # Core codebase
│   ├── model.py               # BiGRU Encoder, GRU Decoder, Bahdanau & Luong attention
│   ├── vocab.py               # Vocabulary builder with UNK replacement decoder
│   ├── decode.py              # Greedy decoding, tuned beam search, coverage penalties
│   ├── textrank.py            # TF-IDF Cosine Graph + PageRank extractive model
│   ├── preprocess.py          # CNN/DailyMail dataset loader and tensor cache builder
│   ├── train.py               # Mixed-precision training pipeline with early stopping
│   ├── evaluate.py            # ROUGE evaluation & parameter sweep CLI
│   ├── gov_sources.py         # Government document scraper and text cleaner
│   └── gov_eval.py            # Government benchmark evaluation and policy audit
│
├── data/
│   └── circulars/             # Official government documents & benchmark
│       ├── manifest.jsonl     # Provenance and document metadata
│       ├── cleaned/           # Extracted clean UTF-8 document texts
│       ├── samples/           # Sample document extracts
│       └── raw/               # Retrieved official HTML files
│
└── results/                   # Experimental logs and generated metrics
    ├── metrics.json           # Baseline benchmark metrics (JSON)
    ├── metrics.md             # Baseline benchmark table (Markdown)
    ├── tuning.md              # Quality tuning & decoding sweep analysis
    ├── beam_sweep.json        # Hyperparameter grid search logs
    ├── loss_curves.png        # Training/validation loss curves
    ├── attention_heatmap.png  # Decoder-to-encoder attention alignment matrix
    └── government/            # Government document evaluation artifacts
        ├── metrics.md         # Government benchmark summary
        ├── document_metrics.json
        └── summaries.json
```

---

## Quickstart & Installation

### 1. Prerequisites & Environment Setup

```bash
# 1. Clone repository
git clone https://github.com/<your-username>/precisAI.git
cd precisAI

# 2. Create conda environment
conda create -n precisai python=3.11 -y
conda activate precisai

# 3. Install PyTorch with CUDA 12.6
pip install torch --index-url https://download.pytorch.org/whl/cu126

# 4. Install pinned project dependencies
pip install -r requirements.txt
```

### 2. Preprocess Data

Processes the CNN/DailyMail 3.0.0 subset (40k train, 2k val, 500 test), trims articles to 400 tokens and targets to 60 tokens, and constructs a 30,000-token vocabulary:

```bash
python -m src.preprocess

# Or run a fast sanity check on a 500/100/50 subset:
python -m src.preprocess --smoke
```

### 3. Model Training

Train the sequence-to-sequence model with automatic mixed precision (AMP) and early stopping:

```bash
# Train with Bahdanau additive attention (recommended)
python -m src.train --attn bahdanau --epochs 10

# Train Luong or vanilla baseline:
python -m src.train --attn luong --epochs 10
python -m src.train --attn none --epochs 10
```

*Checkpoints are saved to `checkpoints/{attn}.pt`.*

### 4. Evaluation & Quality Tuning

Evaluate checkpoints on the 500-article test set:

```bash
# Standard evaluation
python -m src.evaluate

# Tuned evaluation with attention-guided UNK replacement and beam search
python -m src.evaluate --models bahdanau --unk_replace --beam 4 --alpha 1.0 --min_len 20
```

### 5. Government Document Benchmark

Run domain-adaptation evaluation on the collected government circulars:

```bash
# Extract and update cleaned circular texts from manifest
python -m src.gov_sources

# Run government document evaluation
python -m src.gov_eval

# Run with sentence-aware document chunking
python -m src.gov_eval --chunking
```

### 6. Interactive Web Dashboard

Launch the interactive Streamlit user interface:

```bash
streamlit run app.py
```

The web UI provides:
1. **CNN/DailyMail Mode**: Compare extractive TextRank against trained Seq2Seq models, select decoding methods (greedy vs beam), and render attention heatmaps.
2. **Government Document Mode**: Select an official PIB document from the manifest (or paste your own), inspect token compression ratios, and review real-time policy preservation flags.

---

## Citation & Acknowledgments

- **CNN/DailyMail Dataset**: Hermann et al. (2015), *Teaching Machines to Read and Comprehend*; See et al. (2017), *Get To The Point: Summarization with Pointer-Generator Networks*.
- **TextRank**: Mihalcea and Tarau (2004), *TextRank: Bringing Order into Texts*.
- **Neural Attention Mechanisms**: Bahdanau et al. (2014); Luong et al. (2015).
- **Government Sources**: Press Information Bureau (PIB), Government of India public portals.

---

## License

This project is licensed under the [MIT License](LICENSE).
