'''
The Glass Brain Lab audio transcription and diarization script using WhisperX.

This script processes StoryCorps audio files to produce time-stamped, speaker-labeled transcripts. 
For each audio file, three output files are produced:
    - {audio_name}_diarized_transcript.txt                       (one line per speaker utterance)
    - {audio_name}_wordlevel_transcript.csv   (one row per word with speaker label and timestamp)
    - {audio_name}_speaker_embeddings.csv               (speaker embedding vectors from pyannote)

See the repository README for installation instructions and common setup errors.

~~~ Key run arguments ~~~

    Required positional args:
        input_dir       path to folder containing audio files (or use with --input_file)
        output_dir      path where output files will be saved

    Commonly used optional args:
        --input_file          name of a specific file to process instead of all files in input_dir
        --audio_file_type    extension of audio to process, e.g. .mp3, .wav, .flac (default: .mp3)
        --min_speaker_count              minimum number of speakers expected per file (default: 1)
        --max_speaker_count              maximum number of speakers expected per file (default: 8)
        --hf_token                    Hugging Face token required for pyannote speaker diarization
        --device                           "cuda" for GPU machines, "cpu" otherwise (default: cpu)
        --batch_size                                   16 for GPU machines, 4 for CPU (default: 4)
        --compute_type    "float16", "float32", or "int8"; Macs use "float32" or "int8" (def: int8)
        --replace_output                   if set, re-processes files even if output already exists

    To see all run arguments, go down to line 690.

~~~ Example commands ~~~

    # Processing all files in one folder, on a CPU machine:
    python transcribe_diarize_with_timestamps_whisperx.py \
        /path/to/audio \
        /path/for/output \
        --hf_token YOUR_HUGGING_FACE_TOKEN

    # Same as above, with a GPU Linux machine:
    python transcribe_diarize_with_timestamps_whisperx.py \
        /path/to/audio \
        /path/for/output \
        --hf_token YOUR_HUGGING_FACE_TOKEN \
        --device cuda \
        --batch_size 16 \
        --compute_type float16

    # To process a single, specific file:
    python transcribe_diarize_with_timestamps_whisperx.py \
        /path/to/audio \
        /path/for/output \
        --hf_token YOUR_HUGGING_FACE_TOKEN \
        --input_file audio_file.mp3

    # To process a specific list of files, pass a .txt file containing a
    # comma-separated list of filenames (filenames are relative to input_dir):
    python transcribe_diarize_with_timestamps_whisperx.py \
        /path/to/audio \
        /path/for/output \
        --hf_token YOUR_HUGGING_FACE_TOKEN \
        --input_file files_to_process.txt

    # Alternatively, pass a comma-separated list of filenames directly:
    python transcribe_diarize_with_timestamps_whisperx.py \
        /path/to/audio \
        /path/for/output \
        --hf_token YOUR_HUGGING_FACE_TOKEN \
        --input_file "file1.mp3,file2.mp3,file3.mp3"

If you see "Files to be processed: []", verify that:
    1. The first argument points to the *folder* containing the audio files, not the file itself
    2. The --audio_file_type matches your audio file extension

~~~ Run Notes ~~~

Hugging Face token: A Hugging Face token is required to use the pyannote diarization model.
Create a free account at huggingface.co, generate a token, and pass it via --hf_token.
You must also accept the terms of use for pyannote/speaker-diarization-3.1 on the model page.

Speaker count: For multi-speaker audio, the defaults of min_speaker_count=1 and
max_speaker_count=8 generally produce better results than a tight speaker count range,
even when the true speaker count is known. A recommended follow-up step is to use
the GBL script called map_extra_speakers.py to consolidate any extra hallucinated speakers, as
long as the true total number is known.

Language detection: The --use_language_detection flag is available but has been found
unreliable in practice and is not recommended for general use.

~~~ Attribution ~~~

Author: ardyn.olszko@uvm.edu

TODO: license, statement for how we want this code to be shared
'''

#%% IMPORT PACKAGES

import os
import argparse
import sys
import glob
import re
import json

import whisperx
from whisperx.diarize import DiarizationPipeline
import whisperx.audio

import time
import pandas as pd

# find_storycorps_file_location is a helper script, used only with --use_nested_dirs.
# External users who are not using the Glass Brain Lab StoryCorps database do not use 
# that option and so do not need this file.
try:
    from find_storycorps_file_location import get_path_to_audio

except ImportError:
    get_path_to_audio = None

#%% SET DEFAULTS (DO NOT CHANGE)
# Changing these boolean defaults affects syntax of argument parsing

do_replace_output = False # if true, replace any existing diarized transcripts in output directory
use_speaker_count_files = False # if true, uses speaker count extracted from filenames
use_filename_in_prompt = False # if true, add filename to initial prompt
use_language_threshold = False # if true, check for english langauge above threshold
use_nested_dirs = False # if true, uses alphabetical nested directory structure when saving out transcripts
use_file_list_full_path = False # if true, uses absolute path for every file as given in the input file instead of relative to the input dir
use_speaker_count_metadata = False # if true, use speaker count extracted from metadata dictionary (specific format)
save_diagnostics = False # if true, saves out additional "diagnostic" files for every audio file processed
condition_on_previous_text=False

#%% SET USER DEFAULTS (CAN BE CHANGED)

input_file = None # None or name of file to be processed 

# if use_speaker_count_files == True
speaker_count_file = None  # required when --use_speaker_count_files is set
speaker_count_tol_min = 1 # tolerance for min speaker count (if equal to 1, then transcript can have 1 less speaker than expected)
speaker_count_tol_max = 1 # tolerance for max speaker count (if equal to 1, then transcript can have 1 more speaker than expected

# if use speaker_count_metadata == True
speaker_count_metadata = None  # required when --use_speaker_count_metadata is set

# if use_speaker_count_files == False
max_speaker_count = 8
min_speaker_count = 1

initial_prompt = None
# initial_prompt = 'This is a recorded conversation that includes people with the following names: '

use_word_timestamps = True # if true, returns start/end timestamps at the word level (instead of phrase level)

language_threshold = 0.95  # minimum proportion of content that must be in the target language
language_type = 'en'

whisper_model_type = "medium.en"
lang_detect_model_type = "medium"
diarization_pipeline_type = "pyannote/speaker-diarization-3.1"
# diarization_pipeline_type = "pyannote/speaker-diarization" # older pyannote (runs slower)
hf_token = None  # provide your Hugging Face token via --hf_token (see Run Notes in module docstring)
audio_file_type = '.mp3'

# Note: these ASR options are passed at model load time (not transcribe time);
# passing them directly to transcribe() does not appear to have any effect.
no_speech_threshold=0.6
hallucination_silence_threshold=None # time in seconds

min_file_size = 10 # file size in KB

#%% SET DEVICE DEFAULTS

# CPU defaults (suitable for most machines)
device = 'cpu'
batch_size = 4
compute_type = 'int8'

# GPU settings (for Linux/Windows machines with a CUDA-capable GPU):
# device = 'cuda'
# batch_size = 16
# compute_type = 'float16'

#%% FUNCTIONS  
    
def _worker(audio, mp3_name, output_dir,
            use_nested_dirs=False,
            fill_nearest=False,
            return_char_alignments=False,
            language='en'):
    """
    Run transcription, alignment, and diarization for a single audio file.

    Parameters
    ----------
    audio : np.ndarray
        Audio loaded by whisperx.load_audio.
    mp3_name : str
        Filename stem (no extension) used to construct output file names.
    output_dir : str
        Directory where output files are saved.
    use_nested_dirs : bool
        If True, appends a lab-specific alphabetical subdirectory path under output_dir.
        Requires find_audio_file_location.py (not included in this repository).
    fill_nearest : bool
        Passed to whisperx.assign_word_speakers; if True, fills unassigned words with
        the nearest speaker label by time.
    return_char_alignments : bool
        Passed to whisperx.align; if True, returns character-level alignments in addition
        to word-level.
    language : str
        Language code for transcription (default: 'en').
    """
    if use_nested_dirs:
        if get_path_to_audio is None:
            raise ImportError(
                "--use_nested_dirs requires find_audio_file_location.py, which is a "
                "lab-specific helper not included in this repository."
            )
        
        target_path = get_path_to_audio(mp3_name + '.mp3')
        output_dir = os.path.join(output_dir,target_path)

    output_path = os.path.join(output_dir, mp3_name + '_diarized_transcript.txt')  
    
    # Get speaker count info
    speaker_count_source = 'min_speaker_count, max_speaker_count'
    if use_speaker_count_files | use_speaker_count_metadata:
        try:
            expected_number_of_speakers = num_spkrs_df.loc[mp3_name,'NumberOfSpeakers'].max()
            print(f'... {expected_number_of_speakers} speakers expected, tolerance -{speaker_count_tol_min},+{speaker_count_tol_max}')
            min_speakers = max([0,expected_number_of_speakers - speaker_count_tol_min])
            max_speakers = expected_number_of_speakers + speaker_count_tol_max

            if use_speaker_count_files:
                speaker_count_source = 'speaker_count_files'

            else:
                speaker_count_source = 'speaker_count_metadata'

        except KeyError:
            # speaker count not available; use defaults
            min_speakers=min_speaker_count
            max_speakers=max_speaker_count
            pass

    else:
        min_speakers=min_speaker_count
        max_speakers=max_speaker_count
           
    # Prepare prompt for transcription
    if use_filename_in_prompt:
        initial_prompt_str = get_initial_prompt(initial_prompt,mp3_name.replace('_',' '))
    else:
        initial_prompt_str = get_initial_prompt(initial_prompt)
    
    # # Prepare transcription   
    transcription_result = model.transcribe(audio, language=language, batch_size=batch_size)

    if save_diagnostics:
        with open(output_path.replace('_diarized_transcript.txt','_dx_transcript.txt'),
                  "w", encoding="utf-8") as text_file:
            text_file.write( str(transcription_result) )

    print('... transcription complete')
    
    # Prepare alignment
    transcription_result = whisperx.align(
        transcription_result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=return_char_alignments
    )

    # print(transcription_result)
    if save_diagnostics:
        with open(output_path.replace('_diarized_transcript.txt','_dx_transcript_aligned.txt'),
                  "w", encoding="utf-8") as text_file:
            text_file.write( str(transcription_result) ) 
                
    # Prepare diarization
    # NOTE: request Pyannote embeddings be returned along w/ diarization results
    # this version of WhisperX allows return of speaker embeddings as:
    # speaker_embeddings = {speaker: embeddings[s].tolist() for s, speaker in enumerate(diarization.labels())}
    diarization_result, speaker_embeddings = pipeline(
        audio,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        return_embeddings=True
    )

    # save embeddings to file
    print('...saving embeddings to file')
    df_data = []
    for speaker, embedding in speaker_embeddings.items():
        df_data.append({'speaker': speaker, 'embedding': embedding})

    df = pd.DataFrame(df_data)
    df.to_csv(output_path.replace('_diarized_transcript.txt','_speaker_embeddings.csv'))

    if save_diagnostics:
        diarization_result.to_csv(output_path.replace('_diarized_transcript.txt','_dx_diarization.csv'))

    print('... diarization complete')
    
    # Combine transcription and diarization into final result
    final_result = whisperx.assign_word_speakers(
        diarization_result,
        transcription_result, 
        fill_nearest=fill_nearest
    )

    if save_diagnostics:
        with open(output_path.replace('_diarized_transcript.txt','_dx_combined.txt'),
                  "w", encoding="utf-8") as text_file:
            text_file.write( str(final_result) )
    
    # Format outputs
    final_result_str = ""
    for seg in final_result['segments']:
        try:
            line = f"{seg['start']:.2f} {seg['end']:.2f} {seg['speaker']} {seg['text']}\n"

        except:
            line = f"{seg['start']:.2f} {seg['end']:.2f} None {seg['text']}\n"

        final_result_str += line

    with open(f"{output_path}", "w", encoding="utf-8") as text_file:
        text_file.write(final_result_str)
        
    # Save out word-level transcript 
    if use_word_timestamps:
        df = pd.DataFrame(final_result['word_segments'])            
        df.to_csv(output_path.replace('_diarized_transcript.txt','_wordlevel_transcript.csv'), 
                  index = False, header = ['text','start','end','score','speaker'])  
    
    if save_diagnostics:
        log_dict = {}
        log_dict['language'] = language
        log_dict['batch_size'] = batch_size
        log_dict['device'] = device 
        log_dict['return_char_alignments'] = return_char_alignments
        log_dict['speaker_count_source'] = speaker_count_source
        log_dict['min_speakers'] = min_speakers
        log_dict['max_speakers'] = max_speakers 
        log_dict['fill_nearest'] = fill_nearest
        with open(output_path.replace('_diarized_transcript.txt','_dx_log.txt'),
                  "w", encoding="utf-8") as text_file:
            text_file.write( str(log_dict) )
                
    print(f'Finished {output_path}')
    

def transcribe_mp3s(audio, mp3_name, output_dir, use_nested_dirs=False):
    """Entry point for processing a single audio file. Calls _worker if a filename is provided."""
    if mp3_name:
        print(f"Starting {mp3_name}")
        _worker(audio, mp3_name, output_dir,use_nested_dirs=use_nested_dirs)
     
def update_globals(args):
    """Apply parsed command-line arguments to global configuration variables.
    Only updates a variable if the corresponding argument was explicitly provided.

    The global declarations are required because this script stores its configuration
    as module-level variables that are read directly by worker functions at runtime."""
    # 1. Declare globals
    # input and file handling
    global input_file, use_nested_dirs, use_file_list_full_path, audio_file_type, min_file_size

    # output and run flags
    global do_replace_output, save_diagnostics

    # speaker count
    global use_speaker_count_files, speaker_count_file
    global use_speaker_count_metadata, speaker_count_metadata
    global speaker_count_tol_min, speaker_count_tol_max, min_speaker_count, max_speaker_count

    # transcription model
    global whisper_model_type, initial_prompt, use_filename_in_prompt
    global condition_on_previous_text, no_speech_threshold, hallucination_silence_threshold

    # language detection
    global use_language_threshold, lang_detect_model_type, language_type, language_threshold

    # diarization
    global hf_token, diarization_pipeline_type

    # device and compute
    global device, batch_size, compute_type

    # 2. Overwrite defaults with user-defined run args, if available
    # input and file handling
    if args.input_file:
        input_file = args.input_file

    if args.use_nested_dirs:
        use_nested_dirs = args.use_nested_dirs

    if args.use_file_list_full_path:
        use_file_list_full_path = args.use_file_list_full_path

    if args.audio_file_type:
        audio_file_type = args.audio_file_type

    if args.min_file_size:
        min_file_size = args.min_file_size

    # output and run flags
    if args.replace_output:
        do_replace_output = args.replace_output

    if args.save_diagnostics:
        save_diagnostics = args.save_diagnostics

    # speaker count
    if args.use_speaker_count_files:
        use_speaker_count_files = args.use_speaker_count_files

    if args.speaker_count_file:
        speaker_count_file = args.speaker_count_file

    if args.use_speaker_count_metadata:
        use_speaker_count_metadata = args.use_speaker_count_metadata

    if args.speaker_count_metadata:
        speaker_count_metadata = args.speaker_count_metadata

    if args.speaker_count_tolerance is not None:
        speaker_count_tol_str = args.speaker_count_tolerance
        speaker_count_tol_list = speaker_count_tol_str.split(',')
        if len(speaker_count_tol_list)==1:
            speaker_count = int(re.sub(r'\D','',speaker_count_tol_list[0]))
            speaker_count_tol_min = speaker_count
            speaker_count_tol_max = speaker_count

        elif len(speaker_count_tol_list)==2:
            speaker_count_tol_min = int(re.sub(r'\D','',speaker_count_tol_list[0]))
            speaker_count_tol_max = int(re.sub(r'\D','',speaker_count_tol_list[1]))

    if args.min_speaker_count is not None:
        min_speaker_count = args.min_speaker_count

    if args.max_speaker_count is not None:
        max_speaker_count = args.max_speaker_count

    # the transcription model
    if args.whisper_model_type:
        whisper_model_type = args.whisper_model_type

    if args.initial_prompt:
        initial_prompt = args.initial_prompt

    if args.use_filename_in_prompt:
        use_filename_in_prompt = args.use_filename_in_prompt

    if args.condition_on_previous_text:
        condition_on_previous_text = args.condition_on_previous_text

    if args.no_speech_threshold:
        no_speech_threshold = args.no_speech_threshold

    if args.hallucination_silence_threshold is not None:
        hallucination_silence_threshold = args.hallucination_silence_threshold

    # L
    # language detection
    if args.use_language_detection:
        use_language_threshold = args.use_language_detection

    if args.lang_detect_model_type:
        lang_detect_model_type = args.lang_detect_model_type

    if args.lang_type:
        language_type = args.lang_type

    if args.lang_threshold is not None:
        language_threshold = args.lang_threshold

    # D
    # diarization
    if args.hf_token:
        hf_token = args.hf_token

    if args.diarization_pipeline:
        diarization_pipeline_type = args.diarization_pipeline

    # device and compute
    if args.device:
        device = args.device

    if args.batch_size:
        batch_size = args.batch_size

    if args.compute_type:
        compute_type = args.compute_type

def print_globals():
    """Print and return all current global configuration values as a list of strings."""
    globals_list=[]
    globals_list.append("Updated global values:")

    # input and file handling
    globals_list.append(f'input_file = {input_file}')
    globals_list.append(f'use_nested_dirs = {use_nested_dirs}')
    globals_list.append(f'use_file_list_full_path = {use_file_list_full_path}')
    globals_list.append(f"audio_file_type = {audio_file_type}")
    globals_list.append(f"min_file_size = {min_file_size}")

    # output and run flags
    globals_list.append(f"do_replace_output = {do_replace_output}")
    globals_list.append(f"save_diagnostics = {save_diagnostics}")

    # speaker count
    globals_list.append(f"use_speaker_count_files = {use_speaker_count_files}")
    globals_list.append(f"speaker_count_file = {speaker_count_file}")
    globals_list.append(f"use_speaker_count_metadata = {use_speaker_count_metadata}")
    globals_list.append(f"speaker_count_metadata = {speaker_count_metadata}")
    globals_list.append(f"speaker_count_tol_min = {speaker_count_tol_min}")
    globals_list.append(f"speaker_count_tol_max = {speaker_count_tol_max}")
    globals_list.append(f"min_speaker_count = {min_speaker_count}")
    globals_list.append(f"max_speaker_count = {max_speaker_count}")

    # the transcription model
    globals_list.append(f"whisper_model_type = {whisper_model_type}")
    globals_list.append(f"initial_prompt = {initial_prompt}")
    globals_list.append(f"use_filename_in_prompt = {use_filename_in_prompt}")
    globals_list.append(f"condition_on_previous_text = {condition_on_previous_text}")
    globals_list.append(f"no_speech_threshold = {no_speech_threshold}")
    globals_list.append(f"hallucination_silence_threshold = {hallucination_silence_threshold}")

    # language detection
    globals_list.append(f"use_language_threshold = {use_language_threshold}")
    globals_list.append(f"lang_detect_model_type = {lang_detect_model_type}")
    globals_list.append(f"language_type = {language_type}")
    globals_list.append(f"language_threshold = {language_threshold}")

    # diarization
    globals_list.append(f"hf_token = REDACTED")
    globals_list.append(f"diarization_pipeline_type = {diarization_pipeline_type}")

    # device and compute
    globals_list.append(f"device = {device}")
    globals_list.append(f"batch_size = {batch_size}")
    globals_list.append(f"compute_type = {compute_type}")

    for line in globals_list:
        print(line)

    return globals_list
    
def detect_language_threshold(audio, model, threshold=0.6, threshold_language='en'):
    """
    Method detect_language_threshold will detect the language of the inputted file, and write
    the maximum probability to the output csv. This will also return the max probability.
    The model needs to be passed in as it takes a while to load the model. Threshold defaults to 0.6 for language
    english, change this if desired.

    :param infile: Input mp3, wav, m4a, etc file
    :param model: Whisper model to use for language detection
    :param threshold: Threshold for the probability to be above. Default to 0.6 to threshold for accents.
    :param threshold_language: Specific language being thresholded across.
    :param save: whether the results should be saved or not, defaults to False
    :param outfile: result .csv file
    :return: max language, max probability (max is most likely, i.e. highest probability). This will only return
    this if it is above the threshold, which if it isn't it will return false. As a result, the output can be used
    as a boolean, since any output will be considered "truthy".
    
    MODIFIED FROM language_detection_methods.py
    """

    # Load audio into correct form for detect_language
    # See https://lablab.ai/t/whisper-tutorial for this
    audio = whisperx.audio.pad_or_trim(audio)
    mel = whisperx.audio.log_mel_spectrogram(audio,n_mels=80,padding=0).to(model.device)

    # Detect the language
    _, probs = model.detect_language(mel)

    # This is looking at the max language and the probability for the output
    max_lang = max(probs, key=probs.get)
    max_val = probs[max_lang]

    # iterate through the dictionary
    max2 = 0
    for k in probs.keys():
        v = probs[k]
        if max2 < v < max_val:
            max2 = v

    # Now look for the thresholding language
    if probs[threshold_language] < threshold or probs[threshold_language] < max_val:
        # for implementing this, you can use the following:
        # function call: language, value = detect_language_threshold(infile, model,...etc)
        # implementation: if language and value:
        # Since this is returning either false booleans or floats, this works with the truthiness of the
        # floats to pass if they have a value. If for some reason this isn't working (different versions of python)
        # You can instead use: if language != False and value != False:       
        return False, False
    
    else:
        return max_lang, max_val

def get_initial_prompt(initial_prompt,additional_string = None):
    """
    Return the initial prompt string, optionally appending an additional string.
    Returns None if initial_prompt is None.
    """
    if initial_prompt is None:
        initial_prompt_str = None

    else:
        initial_prompt_str = initial_prompt
        if additional_string is not None:
            initial_prompt_str = str(initial_prompt_str + additional_string)    
    
    return initial_prompt_str

def return_metadata_speaker_count(metadata_file):
    """
    Parse a metadata JSON file and return a DataFrame of expected speaker counts per interview.

    The returned DataFrame has columns: InterviewID, StoryName, NumberOfSpeakers, and several
    sub-counts (Participants, Filename, Additional) used to derive the final expected count.
    """
    # load metadata dictionary
    with open(metadata_file,'r',encoding='utf-8') as f:
        metadata_dict = json.load(f)
    
    # process into dict of number of speakers expected
    speaker_counts_list = []
    for interview_key,details in metadata_dict.items():
        interview_title = details['title_gbl']
        filename_split = re.split('__and_|_and_|__',interview_title)
        filename_speaker_count = len(filename_split)
        
        additional_speakers = details['interview']['additional_participants'] # list of speakers (first_name, last_name)
        additional_speaker_count = len(details['interview']['additional_participants'])
        total_speakers = additional_speaker_count
        
        if 'participants' in details:
            participants_data = details['participants'] # list of speakers (first_name, last_name)
            participants_speaker_count = len(participants_data)
   
            total_speakers = participants_speaker_count + additional_speaker_count
            
            repeat_names_list=[]
            for participant_data in participants_data:
                first_name = participant_data['first_name']
                last_name = participant_data['last_name']

                for add_speaker_dict in additional_speakers:
                    add_first_name = add_speaker_dict['first_name']
                    add_last_name = add_speaker_dict['last_name']

                    if first_name == add_first_name and last_name == add_last_name:
                        total_speakers -= 1
                        repeat_names_list.append((first_name,last_name))

                        continue

            if total_speakers > additional_speaker_count and total_speakers > participants_speaker_count:
                print(f'\n{interview_title} has more speakers than might be expected')
                print('Names that repeated: ')
                print(repeat_names_list)
             
            speaker_count = max([additional_speaker_count,participants_speaker_count,total_speakers])
            
            interview_dict = {}
            interview_dict['InterviewID'] = interview_key
            interview_dict['StoryName'] = interview_title
            interview_dict['NumberOfSpeakers'] = speaker_count
            interview_dict['NumberOfSpeakers_Participants'] = participants_speaker_count
            interview_dict['NumberOfSpeakers_Filename'] = filename_speaker_count
            interview_dict['NumberOfSpeakers_Additional'] = additional_speaker_count
            speaker_counts_list.append(interview_dict)
        
    df = pd.DataFrame(speaker_counts_list)
    
    return df

#%% Main script

if __name__ == '__main__':
    
    start = time.time()

    # Define the run arguments
    parser = argparse.ArgumentParser(description='Save output of MP3 filenames to text files')

    # input and file handling
    parser.add_argument('input_dir', type=str, help='directory path containing MP3 files, or list of directories, or list of files')
    parser.add_argument('output_dir', type=str, help='directory path where transcribed files should be saved')
    parser.add_argument('--input_file', type=str, help='name(s) of file(s) to be processed; otherwise, process all files in input_dir')
    parser.add_argument('--use_nested_dirs', action='store_true', help='save transcripts using alphabetical nested filename structure')
    parser.add_argument('--use_file_list_full_path', action='store_true', help='use full path of file as given in file list instead of relative to input_dir')
    parser.add_argument('--audio_file_type', type=str, help='type of audio file to process')
    parser.add_argument('--min_file_size', type=float, help='minimum size that audio file must be to process (in KB)')

    # output and run flags
    parser.add_argument('--replace_output', action='store_true', help='replace any existing diarized transcript')
    parser.add_argument('--save_diagnostics', action='store_true', help='save out log and separate transcript and diarization results')

    # speaker count
    parser.add_argument('--use_speaker_count_files', action='store_true', help='use pre-processed speaker count files')
    parser.add_argument('--speaker_count_file', type=str, help='full path to speaker count file')
    parser.add_argument('--use_speaker_count_metadata', action='store_true', help='use participant count from metadata dict')
    parser.add_argument('--speaker_count_metadata', type=str, help='full path to metadata dict')
    parser.add_argument('--speaker_count_tolerance', type=str, help='minus,plus or single value tolerance to use with pre-processed speaker count')
    parser.add_argument('--min_speaker_count', type=int, help='min expected number of speakers')
    parser.add_argument('--max_speaker_count', type=int, help='max expected number of speakers')

    # the transcription model
    parser.add_argument('--whisper_model_type', type=str, help='type of model to use for whisper transcription')
    parser.add_argument('--initial_prompt', type=str, help='initial prompt that provides context to whisper transcription')
    parser.add_argument('--use_filename_in_prompt', action='store_true', help='include audio filename in the initial prompt')
    parser.add_argument('--condition_on_previous_text', action='store_true', help='condition transcription on previous text')
    parser.add_argument('--no_speech_threshold', type=float, help='silent if no_speech probability exceeds threshold')
    parser.add_argument('--hallucination_silence_threshold', type=float, help='skip silent periods longer than this threshold (in seconds)')

    # language detection
    parser.add_argument('--use_language_detection', action='store_true', help='turn on language detection')
    parser.add_argument('--lang_detect_model_type', type=str, help='type of model to use for language detection')
    parser.add_argument('--lang_type', type=str, help='language to detect')
    parser.add_argument('--lang_threshold', type=float, help='proportion threshold to use for language detection')

    # diarization
    parser.add_argument('--hf_token', type=str, help='hugging face token, specific to speaker diarization pipeline')
    parser.add_argument('--diarization_pipeline', type=str, help='pyannote speaker diarization pipeline')

    # device and compute
    parser.add_argument('--device', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--compute_type', type=str)      
    
    args = parser.parse_args()
    
    # Parse aguments into variables
    jobdir = args.input_dir
    output_dir = args.output_dir

    # Update global variables based on parsed arguments
    update_globals(args)
    settings = print_globals()
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load speaker count
    if use_speaker_count_files:
        num_spkrs_df = pd.read_csv(speaker_count_file)
        num_spkrs_df.set_index('StoryName',inplace=True)

    elif use_speaker_count_metadata:
        num_spkrs_df = return_metadata_speaker_count(speaker_count_metadata)
        num_spkrs_df.set_index('StoryName',inplace=True) # StoryName is the title_gbl of the metadata file: should match audio filename
        
    # Load diarization pipeline
    try:
        pipeline = DiarizationPipeline(use_auth_token=hf_token, device=device)

    except FileExistsError:
        pass


    initial_prompt_str = get_initial_prompt(initial_prompt)
        
    # Load transcription model
    # model = whisper.load_model(whisper_model_type)
    # ASR options are passed at model load time; passing them to transcribe() does not take effect.
    model = whisperx.load_model(
        whisper_model_type,
        device,
        compute_type=compute_type,
        asr_options = {
            'initial_prompt':initial_prompt, 
            'condition_on_previous_text':condition_on_previous_text,
            'no_speech_threshold':no_speech_threshold, 
            'hallucination_silence_threshold':hallucination_silence_threshold
            }
        )
    print(model.options)
    
    # Load align model
    model_a, metadata = whisperx.load_align_model(language_code='en', device=device)
    
    # Load language detection model
    if use_language_threshold:
        # lang_model = whisper.load_model(lang_detect_model_type)
        lang_model = whisperx.load_model(lang_detect_model_type, device, compute_type=compute_type)
    
    # Process input_file if given    
    if input_file is None: # no file: process all files in directory
        mp3s = glob.glob(os.path.join(jobdir,'*'+audio_file_type))

    elif ('.txt' in input_file):
        input_file_string = open(input_file, 'r', encoding='utf-8').read() 
        if (',' not in input_file_string): # input file contains single mp3 file
            if not use_file_list_full_path:
                mp3s = [os.path.join(jobdir,input_file_string)]

            else:
                mp3s = [input_file_string]

        else: # input file contains comma-separated list of mp3 files
            input_file_list = input_file_string.split(',')
            if not use_file_list_full_path:
                mp3s = [os.path.join(jobdir,file) for file in input_file_list] 

            else:
                mp3s = input_file_list

    elif (',' not in input_file) and ('.txt' not in input_file): # input given as single mp3 file
        if not use_file_list_full_path:
            mp3s = [os.path.join(jobdir,input_file)]

        else:
            mp3s = [input_file]

    else: # input given as list of mp3 files
        input_file_list = input_file.split(',')
        if not use_file_list_full_path:
            mp3s = [os.path.join(jobdir,file) for file in input_file_list]

        else:
            mp3s = input_file_list
                   
    print(f'Source: {jobdir}')
    print(f'Transcripts: {output_dir}')

    print(f'Files to be proccessed: {mp3s}')
    print("Starting our loop of files")
    for mp3 in mp3s:
        file_start = time.time()
        
        # Check if audio file is already processed
        # mp3_name = os.path.splitext(mp3)[0]
        mp3_name = os.path.basename(mp3).replace(audio_file_type,'')
        is_processed = mp3_name + '_diarized_transcript.txt' in os.listdir(output_dir)
        if (is_processed and not do_replace_output):
            continue
        
        # Check file size
        mp3_size = os.path.getsize(mp3) # file size in bytes
        if mp3_size*0.001 < min_file_size:
            print(f'Skipping processing due to file size (<{min_file_size}KB): {mp3}')
            with open(os.path.join(output_dir,"transcribe_diarize_failed_file_size.csv"), 'a', encoding='utf-8') as f:
                f.write(mp3_name + f"{audio_file_type}\n")

            continue
        
        # Load as audio
        try:
            audio = whisperx.load_audio(mp3)

        except:
            print(f'Error loading: {mp3}')
            with open(os.path.join(output_dir,"transcribe_diarize_failed_to_load.csv"), 'a', encoding='utf-8') as f:
                f.write(mp3_name + f"{audio_file_type}\n")

            continue
        
        print(f'Processing: {mp3}')
        
        # Check for language above threshold
        if use_language_threshold:
            
            try:
                language, probability = detect_language_threshold(
                    audio,
                    lang_model,
                    threshold=language_threshold,
                    threshold_language=language_type
                )

            except Exception as e:
                # write detection errors filenames to csv
                with open(os.path.join(output_dir,"transcribe_diarize_language_detection_errors.csv"), 'a', encoding='utf-8') as f:
                    language = False
                    probability = False
                    f.write(mp3_name + f"{audio_file_type}\n")

                continue

            if not language and not probability:
                # write nonenglish or below threshold filenames to csv
                with open(os.path.join(output_dir,"transcribe_diarize_language_detection_nonenglish.csv"), 'a', encoding='utf-8') as f:
                    f.write(mp3_name + f"{audio_file_type}\n")
                    
                continue
        
        # Transcribe/diarize
        transcribe_mp3s(audio, mp3_name, output_dir, use_nested_dirs=use_nested_dirs)
        
        file_time = (time.time()-file_start) # seconds
        
        with open(os.path.join(output_dir,"transcribe_diarize_file_processing_time.csv"), 'a', encoding='utf-8') as f:
            file_settings = (' | ').join(settings)
            f.write(mp3_name + f'{audio_file_type} | ' +\
                    str(file_time) + " | " +\
                        file_settings +\
                        "\n")
            
    print(f'time to diarize {len(mp3s)} files: ', (time.time()-start)/60)
    print("Done processing files...")
    sys.exit(0)