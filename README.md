# PoliticHeadlinES_LYS-UDC

Code for **LYS-UDC's** participation in **PoliticHeadlinES-IberLEF 2026**: a zero-shot
LLM-based pipeline for ranking candidate headlines of Spanish political news articles,
in both text-only and multimodal (text + image) settings.

The system formulates headline ranking as a constrained text generation task: given a
news body (and, optionally, its accompanying image), the model is prompted to return an
ordered list of 10 candidate headlines from most to least likely correct, using
deterministic inference (temperature = 0) and regex-based parsing with fallback
heuristics to guarantee a valid prediction.

## Pipeline overview

The pipeline went through two phases:

1. **API-based prototyping** (`tarea_texto.py`, `tarea_multimodal.py`) — rapid
   evaluation of several models through the [Groq API](https://groq.com/) on a fixed
   sample (1,000 articles for text-only, 500 for multimodal).
2. **Local deployment** (`tarea_texto_local_hf.py`, `tarea_multimodal_local_hf.py`) —
   final, official submissions using open-source models loaded locally through
   [Hugging Face Transformers](https://huggingface.co/docs/transformers), with optional
   4-bit/8-bit quantization to enable inference under limited computational resources.

`results_formatter.py` merges the raw text and multimodal prediction files from a local
run into the `task_1` / `task_2` ranking format used for submission and analysis.

## Repository structure

```
.
├── preliminary_tests/
│   ├── generar_muestra.py       # Samples 1,000 articles from the full training set
│   ├── tarea_texto.py           # Text-only ranking via Groq API
│   └── tarea_multimodal.py      # Multimodal ranking via Groq API (vision model)
├── tarea_texto_local_hf.py      # Text-only ranking, local HF model
├── tarea_multimodal_local_hf.py # Multimodal ranking, local HF model (vision-language model)
├── results_formatter.py         # Merges local text + multimodal outputs into results.csv
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

For the local HF scripts with quantization, `accelerate` and `bitsandbytes` are also
required:

```bash
pip install accelerate bitsandbytes
```

### Environment variables

The Groq-based scripts (`tarea_texto.py`, `tarea_multimodal.py`) read a `GROQ_API_KEY`
from a `.env` file in the project root:

```
GROQ_API_KEY=your_api_key_here
```

The local HF scripts (`tarea_texto_local_hf.py`, `tarea_multimodal_local_hf.py`) support
an optional `.env` with:

```
HF_TOKEN=your_hf_token_here       # only needed for gated models
HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
HF_MM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
HF_QUANTIZATION=4bit              # none | 4bit | 8bit
HF_CACHE_DIR=/path/to/cache
```

## Data format

All scripts expect an input CSV with the following columns:

| Column | Description |
|---|---|
| `id` | Unique article identifier |
| `article_body` | Full text of the news article |
| `title_1` … `title_10` | The 10 candidate headlines |
| `y_true` | (Optional) Correct headline index, for computing metrics |
| `image_hash` | (Multimodal only) Filename of the associated image |

Place the input CSV at `train_corpora/train_sample_1000.csv` (Groq scripts) or
`data/test_corpora/test_public.csv` (local HF scripts), with images under an `images/`
subfolder.

## Usage

### API-based (Groq)

```bash
python tarea_texto.py
python tarea_multimodal.py
```

Outputs are written to `results/resultados_texto.csv` and
`results/resultados_multimodal.csv`.

### Local deployment (Hugging Face)

```bash
python tarea_texto_local_hf.py --model meta-llama/Llama-3.1-8B-Instruct --quantization 4bit
python tarea_multimodal_local_hf.py --model Qwen/Qwen2.5-VL-7B-Instruct --quantization 4bit
```

Key arguments (see `--help` for the full list):

- `--model`: local path or Hugging Face Hub ID
- `--quantization`: `none`, `4bit`, or `8bit`
- `--max-new-tokens`: generation length cap
- `--target-column`: ground-truth column name for metrics (default `y_true`)
- `--no-download`: use only locally cached model files

Outputs are written to `results/resultados_texto_local_<model>.csv` and
`results/resultados_multimodal_local_<model>.csv`.

### Merging results for submission/analysis

Once both text and multimodal local outputs exist for a given model, organize them into
a folder (e.g. `results/gemma-gemma/`) and run:

```bash
python results_formatter.py results/gemma-gemma results/mistral-gemma results/phi-gemma
```

This produces a `results.csv` per folder with columns `id`, `task_1` (text-only
ranking), and `task_2` (multimodal ranking), each a space-separated list of 10 headline
tokens (e.g. `t3 t5 t1 ...`).

## Metrics

All scripts compute, when a ground-truth column is available:

- **Accuracy (Top-1)** — fraction of articles where the correct headline is ranked first
- **Mean Reciprocal Rank (MRR)**
- **Mean nDCG**
- **Hit Rate @3**

## License

[GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html)