# GBL Transcription and Diarization Pipeline

This README is for the Glass Brain Lab transcription and diarization pipeline, in particular. This 
pipeline is made up of the primary script called `transcribe_diarize_with_timestamps_whisperx.py` 
and is followed by two post-processing scripts: `convert_wordlevel_speaker_to_utterance.py` and 
`make_transcripts_match.py`.

## Installation

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
mamba create -n gbl_xscription_pipeline python=3.11.18
mamba activate gbl_xscription_pipeline
```

### 2. Install packages

Install WhisperX and its dependencies:

```bash
pip install whisperx==3.4.2
```

Then pin the following packages to the versions known to work with this pipeline:

```bash
pip install \
    "pyannote.audio==3.4.0" \
    "huggingface_hub==0.36.0" \
    "transformers==4.57.3" \
    "lightning==2.6.0" \
    "pytorch-lightning==2.5.5" \
    "torchmetrics==1.8.2"
```

These pins are necessary because newer versions of these packages have breaking API changes
that are incompatible with one another. The second `pip install` will downgrade some packages
that were pulled in by the first step.

### 3. Hugging Face token

The speaker diarization step requires a Hugging Face account and access token:

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Generate an access token at huggingface.co/settings/tokens
3. Accept the terms of use for
   [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   on the model page — this is required even with a valid token
4. Pass your token at runtime with `--hf_token YOUR_TOKEN`

## Preprocessing

### Audio Deduplication

A. Identifying Duplicated Audio

TODO: Ardyn to fill in

B. Trimming Duplicated Audio

Once files with repeated audio have been identified (Part A), trimming is a two-step process
using the scripts in the `audio_deduplication/` folder.

First, `make_deduplication_run_table.py` builds the run table that will be used in the actual 
trimming process: it collects all audio file names from your audio directory, filters to include 
only the files identified for de-duplication, and saves a run table in CSV format for tracking 
processing status across batches.

NOTE: this first step also ended up including a bonus functionality of standardizing the de-
duplicated audio file names. We similarly standardized all other file names that were not a 
part of the deduplication effort, separately.

```bash
python make_deduplication_run_table.py \
    --parent_dir /path/to/audio \
    --targets_file /path/to/files_to_deduplicate.csv \
    --all_audio_file /path/to/all_audio_inventory.csv
```

Then, `audio_deduplication.py` does the actual trimming: for each targeted file it removes the
repeated portion by retaining only the last N minutes of audio (where N is the non-duplicate
duration identified in Part A), and overwrites the original file in-place. The script is designed
to run in parallel batches and uses file locking to safely share the run table across jobs.

NOTE: this file was actually called by a python orchestrator script that was written to work with 
our Slurm service on the UVM cluster. As this orchestrator was specific to our systems, it is not 
included here.

```bash
python audio_deduplication.py \
    --targets_file /path/to/batch1_targets.csv \
    --run_table /path/to/deduplication_run_table.csv \
    --batch_id 1 \
    --group YourGroupName
```

See the docstrings at the top of each script for full argument details.

## Transcription and Diarization

To run `transcribe_diarize_with_timestamps_whisperx.py` on a single file, make sure your 
environment has been activated and then run:

```
python transcribe_diarize_with_timestamps_whisperx.py \
    /path/to/audio/parent/folder \
    /path/for/output \
    --hf_token YOUR_HUGGING_FACE_TOKEN \
    --input_file audio_file.mp3
```

More examples showing the different ways to run this script, along with explanations of the run 
arguments, can be found in the docstring at the top of the file.

## Post-processing

TODO: Ardyn to fill in
