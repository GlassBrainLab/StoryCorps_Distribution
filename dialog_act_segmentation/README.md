# GBL Dialog Act Segmentation (DASeg)

This folder contains the Glass Brain Lab's implementation of dialogue act segmentation (DASeg)
using the model from Zelasko et al. (2021). The pipeline takes word-level transcript CSV files
produced by the GBL transcription and diarization pipeline and assigns a dialogue act label to
each word.

**Original script author:** Skyler Heininger (Glass Brain Lab)

**Model:** [pzelasko/longformer-swda-nolower](https://huggingface.co/pzelasko/longformer-swda-nolower)
(Zelasko et al., 2021 — no Hugging Face token required)

## Installation

**NOTE:** this is a separate environment from the transcription and diarization pipeline.
The two pipelines have different dependency requirements and should not share an environment.

### 1. Create a Python environment

We recommend [miniforge](https://github.com/conda-forge/miniforge) (which provides the `mamba`
command) to create an isolated environment. The commands below use `mamba`, but `conda` is
interchangeable.

**NOTE:** we have updated this process to reflect the security patching for [the eMudhra security 
incident in October 2026](https://discuss.python.org/t/information-for-python-users-regarding-emudhra-emsign-ca-certificate-security-incident/109576). The versions of `Python` and `pip` below are now for patched versions 
that are closest to the original versions we ran for the research published in PNAS. All other 
module versions are the same. We highly recommend you review documentation for your particular 
systems and consult with your security experts to ensure compliance within your own institution.

Create an environment with Python 3.11.18:

```bash
mamba create -n gbl_daseg_pipeline python=3.11.18
mamba activate gbl_daseg_pipeline
```

### 2. Install packages

Install the packages pinned to the versions used for the published research:

```bash
pip install \
    "transformers==4.51.3" \
    "datasets==3.3.2" \
    "torch==2.6.0" \
    "pandas==2.3.3" \
    "numpy==2.0.1"
```

## Usage

### `daseg_pipeline.py` — full GBL pipeline

This script includes text cleaning, sliding-window chunking, and word-level prediction
alignment on top of the base model.

Process a single transcript file:

```bash
python daseg_pipeline.py \
    --file /path/to/transcripts/0001_wordlevel_transcript.csv \
    --output_dir /path/to/output
```

Process a directory of transcript files:

```bash
python daseg_pipeline.py \
    --directory /path/to/transcripts \
    --output_dir /path/to/output
```

Output files follow the naming pattern `<input_stem>_with_predictions.csv`. The script skips
files already present in the output directory, so it is safe to interrupt and re-run.

See the docstring at the top of `daseg_pipeline.py` for full argument details.

### `daseg_basic_usage.py` — minimal example

This file demonstrates running the Zelasko model directly, without the GBL DASeg pipeline. 
It is intended as a reference for users who want to explore the model's raw output on a text string.

## Notes

- **Device:** by default the pipeline runs on CPU. To use a GPU, change `device=-1` to
  `device=0` in the module-level `pipe` and `tokenizer` initializations at the top of
  `daseg_pipeline.py`.
- **Input format:** `daseg_pipeline.py` expects CSV files with a `text` column containing
  one word per row (one of the output files of `transcribe_diarize_with_timestamps_whisperx.py`).

## Reference

Zelasko, P., Szymański, P., Mizgajski, J., Szymańska, A., Carmiel, Y., & Dehak, N. (2021).
*Discovering the right moment to ask a question: Sentiment, human engagement, and dialogue act
modelling for a human-machine dialogue system*. ICASSP 2021.
