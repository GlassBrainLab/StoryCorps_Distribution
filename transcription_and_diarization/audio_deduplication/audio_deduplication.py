
"""
GBL audio de-duplication script: trims StoryCorps audio files that have been identified as
containing repeated audio at the end. The duplicate portion is removed and the original file
is replaced with the trimmed version in-place.

This script is designed to run in parallel batches (e.g. via SLURM array jobs). Each batch
processes a subset of target files and updates a shared run table with progress. File locking
handles concurrent access across parallel batches safely.

A targets CSV must first be prepared using make_deduplication_run_table.py.

~~~ Key run arguments ~~~

    Required:
        --targets_file                               path to CSV of files targeted for trimming
        --run_table           path to shared run table CSV for tracking progress across batches
        --batch_id                                        integer identifier for this batch run
        --group            label for the collection being processed (used in progress messages)

    Optional:
        --update_interval  number of files to process before syncing progress to disk (default: 10)

~~~ Example command ~~~

    python audio_deduplication.py \\
        --targets_file /path/to/batch1_targets.csv \\
        --run_table /path/to/deduplication_run_table.csv \\
        --batch_id 1 \\
        --group YourGroupName

~~~ Run Notes ~~~

Targets CSV must have columns: new_source_path, best_dur_minutes, file_id, run_table_row_idx, status.
Run table must have a column named stage_2_process_status.

NOTE: **trimmed audio overwrites the original file in-place!** We created a backup archive of the 
audio files for deduplication and so this in-place trimming is as-designed.

~~~ Attribution ~~~

Authors: ardyn.olszko and kirsten.bonson@uvm.edu
"""
import os
import csv
import sys
import time
import fcntl
import random
import librosa # cannot write mp3
import argparse
import numpy as np
import pandas as pd

from mutagen.mp3 import MP3
from pydub import AudioSegment
from unidecode import unidecode


parser = argparse.ArgumentParser(
    description='args for mock deduplication process script with concurrent handling',
)
parser.add_argument('--targets_file', type=str, required=True,
                   help='CSV holding file names + paths to back up')
parser.add_argument('--run_table', type=str, required=True,
                   help='CSV for de-duplicating pipeline process orchestration')
parser.add_argument('--batch_id', type=int, required=True,
                   help="Batch ID for this process")
parser.add_argument('--group', type=str, required=True,
                   help="Group name (e.g. MemoryLoss)")
parser.add_argument('--update_interval', type=int, default=10,
                   help="Number of files to process before updating reference tables (default: 10)")

def get_file_id(given_string, name_only=False):
    """
    Extract the unique file ID from a filename or path. The ID is the last 
    underscore-separated component of the filename, without extension.
    """
    if name_only:
        fname = given_string

    else:
        path, fname = os.path.split(given_string)

    name, ext = os.path.splitext(fname)
    file_id = name.split('_')[-1]

    return file_id

def is_size_identical(source_name_path, backup_name_path):
    """
    Return True if two files have identical sizes in bytes.
    """
    return os.path.getsize(source_name_path) == os.path.getsize(backup_name_path)

def find_src_path(source_path, target_id):
    """
    Search the directory of source_path for a file whose ID matches target_id.
    Returns the full path if found, or None.
    """
    path, fname = os.path.split(source_path)
    contents = os.listdir(path)
    
    for file in contents:
        file_id = get_file_id(file)
        
        if file_id == target_id:
            new_src_path = os.path.join(path, file)
            print(f'      ...file ID match found at: {new_src_path}')
            print(f'      (if these looks the same, then it is probably an invisible or non-ascii character difference)')

            return new_src_path

    return None

def clean_fname(file_name):
    """
    Remove non-ASCII characters and strip carriage returns from a filename.
    """
    cleaned_name = unidecode(file_name)

    if "\n" in cleaned_name:
        cleaned_name = cleaned_name.replace("\n", "")

    return cleaned_name

def get_mp3_duration(mp3_file):
    """
    Get the duration of the MP3 file in seconds

    NOTE: this uses the mutagen library and reads the MP3
    file header data to determine duration. Alternatively
    we could use 
        librosa.get_duration(filename=audio_file_path)
    which actually reads in the full audio into memory to
    determine duration -- better if you distrust the 
    metadata for any reason (corruption) but slower.
    """
    audio = MP3(mp3_file)

    return audio.info.length, audio.info.sample_rate

def load_audio(file_path, start_time=None, use_mono=False):
    """
    Loading only the amount of audio we want, now updated to
    do the last X minutes rather than a specific duration

    NOTE: before we were using duration and trimming from the front,
    now we're using a start time and trimming from the rear
    """
    return librosa.load(file_path, sr=None, mono=use_mono, offset=start_time)

def load_and_trim_audio(file_path, end_duration_m):
    """
    This function does the actual trimming of the audio.
    """
    audio_duration, fs = get_mp3_duration(file_path)
    print(f'   ...duration of original audio: {round(audio_duration, 2)} seconds')
    
    # load only the last end_duration_m minutes
    start_time_s = max(0, audio_duration - (end_duration_m * 60))
    audio, fs = load_audio(file_path, start_time=start_time_s)

    print(f'   ...start time now at {round(start_time_s / 60, 2)} minutes into original')
    print(f'   ...new audio dtype and shape: {audio.dtype}, {audio.shape}')

    return audio, fs

def save_audio(data, fs, filename):
    """
    Save our newly-trimmed audio to file.
    """
    # scale float32 data to int16 range for compatibility with MP3 export
    if data.dtype == np.float32:
        # scale to 16-bit PCM range
        data = (data * 32767).astype(np.int16)

    elif data.dtype != np.int16:
        raise ValueError("Audio data must be either float32 or int16 for saving.")

    # check if audio is mono or stereo and create AudioSegment
    if data.ndim == 1:
        # mono audio
        audio_segment = AudioSegment(
            data.tobytes(),
            frame_rate=fs,
            sample_width=2,  # 2 bytes for int16
            channels=1
        )

    else:
        # stereo audio (assuming 2D array with channels in second dimension)
        audio_segment = AudioSegment(
            np.transpose(data).tobytes(),
            frame_rate=fs,
            sample_width=2,  # 2 bytes for int16
            channels=2
        )

    # save as MP3
    audio_segment.export(filename, format="mp3")

def has_been_trimmed(file_path, goal_duration_min, buffer_min=1):
    """
    Check if a file has been trimmed or not.
    """
    current_duration_s, sr = get_mp3_duration(file_path)
    current_duration_min = current_duration_s / 60

    is_trimmed = (current_duration_min <= (goal_duration_min + buffer_min)) and \
        (current_duration_min >= (goal_duration_min - buffer_min))
    
    if is_trimmed:
        print(f'   ...file {file_path} has duration {round((current_duration_s / 60), 2)} min,')
        print(f'which is w/in best_dur_minutes ({goal_duration_min}) +- 10 sec;')
        print(f'this file has been modified since repeats were detected, skipping!')
    
    return is_trimmed

def trim_file(row_data):
    """
    Perform the trimming of the audio file.

    NOTE: this function calls load_and_trim_audio() which does the actual editing.
    """
    try:
        name_path = row_data['new_source_path']

        # make sure we can find this file, if not with exact path then with ID
        if not os.path.exists(name_path):
            print(f'WARNING: file at {name_path} not found, searching path with just file ID')
            target_id = get_file_id(name_path)
            new_src_path = find_src_path(name_path, target_id)
            
            if new_src_path is None:
                print(f'   ...no file found with ID {target_id}, skipping this file!')
                return 'FAILED'
            
            print(f'   ...file found with ID {target_id}, trimming this file')
            name_path = new_src_path

        # check if duration is already as expected; skip if so
        end_duration_to_save_min = row_data['best_dur_minutes']
        if has_been_trimmed(name_path, end_duration_to_save_min):
            return 'SKIPPED'
        
        # get just the last X minutes requested
        print(f'   ...processing {os.path.basename(name_path)} to save only the last {round(end_duration_to_save_min, 2)} minutes...')
        try:
            audio, fs = load_and_trim_audio(name_path, end_duration_to_save_min)
            print(f'   ...file successfully trimmed!')
            
        except Exception as error:
            print(f'ERROR: could not trim file {name_path}, skipping! | {error}')
            return 'FAILED'
        
        # save the new version
        save_audio(audio, fs, name_path)
        print(f'   ...saved processed audio to {name_path}')

        return 'SUCCEEDED'

    except Exception as error:
        print(f'ERROR: failed to trim and save file | {error}')
        print(f'      ...skipping this target!')

        return 'FAILED'

def safe_csv_update(csv_path, updates_dict, batch_id, table_type, max_retries=3):
    """
    Safely update specific rows in CSV with file locking to handle concurrent access.
    This reads the current state from disk, applies updates, then writes back.
    
    Parameters:
        csv_path (str) - path to CSV file
        updates_dict (dict) - dictionary mapping row indices to dictionaries of column updates
                              (e.g. {row_idx: {'column_name': 'new_value', ...}, ...})
        batch_id (int) - batch ID for logging
        table_type (str) - type of table for logging
        max_retries (int) - maximum number of retry attempts
    """
    for attempt in range(max_retries):
        try:
            # add small random delay to reduce collision probability
            time.sleep(random.uniform(0.1, 0.5))
            
            with open(csv_path, 'r+') as f:
                # acquire exclusive lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                
                # read current state from disk
                f.seek(0)
                current_df = pd.read_csv(f)
                
                # apply updates
                for row_idx, column_updates in updates_dict.items():
                    for column_name, new_value in column_updates.items():
                        current_df.loc[row_idx, column_name] = new_value
                
                # write back to file
                f.seek(0)
                f.truncate()
                current_df.to_csv(f, index=False)
                
                # lock is automatically released when file closes
                print(f"   Successfully updated {table_type} table to reflect Batch {batch_id} progress")
                return True
                
        except Exception as error:
            if hasattr(error, 'strerror') and error.strerror == 'No such file or directory':
                print(f'{table_type} table {csv_path} not found! Please check and try again')
                raise Exception(error.strerror)
            
            print(f"WARNING: {table_type} table sync attempt {attempt + 1} failed: {error}")
            if attempt < max_retries - 1:
                # exponential backoff with jitter
                delay = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)

            else:
                print(f"ERROR: Failed to update {table_type} CSV after {max_retries} attempts")
                return False
    
    return False

def sync_table_files(
        processed_files,
        targets_df,
        process_status_column,
        targets_file,
        run_table_file,
        batch_id
    ):
    """
    Update both reference tables with the results of processed files.
    This function now reads the current state from disk before applying updates.
    
    Parameters:
        processed_files (list) - list of tuples (target_id, backup_path, targets_row_idx, run_table_row_idx)
        targets_df (pd.DataFrame) - in-memory targets dataframe (for targets file updates)
        process_status_column (str) - column name for tracking process status
        targets_file (str) - path to targets CSV file
        run_table_file (str) - path to run table CSV file
        batch_id (int) - current batch ID
    """
    # build update dictionaries for both reference tables -- these contain only the *changes*
    targets_updates = {}
    run_table_updates = {}
    for target_id, result, targets_row_idx, run_table_row_idx in processed_files:
        # prepare updates for targets table
        targets_updates[targets_row_idx] = {'status': result}
        
        # prepare updates for run table
        status_value = 'DONE' if result != 'FAILED' else 'FAILED'
        run_table_updates[run_table_row_idx] = {process_status_column: status_value}
    
    print(f'\nUpdating reference tables for processed collection of {len(processed_files)} files')
    for targets_row_idx, column_updates in targets_updates.items():
        for column_name, new_value in column_updates.items():
            targets_df.loc[targets_row_idx, column_name] = new_value
    
    targets_success = safe_csv_update(targets_file, targets_updates, batch_id, 'targets')
    if not targets_success:
        print(f'WARNING: failed to update targets CSV for this batch')
        return False
    
    # update run table by reading current state from disk first
    run_success = safe_csv_update(run_table_file, run_table_updates, batch_id, 'run')
    if not run_success:
        print(f'WARNING: failed to update run table CSV for this batch')
        return False
    
    return True

def safe_read(file_path, max_retries=3):
    """
    Read a CSV with file locking and retry logic to handle concurrent access across parallel batches.
    """
    for attempt in range(max_retries):
        try:
            # 1. Create and acquire file lock
            # NOTE: by default, this will *block and wait* until the file becomes available,
            # so if another job has this locked, this job will pause until the lock
            # has been lifted before locking it itself and continuing.
            with open(file_path, 'r') as fin:
                fcntl.flock(fin.fileno(), fcntl.LOCK_EX)
                
                # 2. Read in the latest version of run table from disk
                disk_df = pd.read_csv(file_path)
                return disk_df
            
        except Exception as error:
            print(f'WARNING: CSV read attempt {attempt + 1} failed with error: {error}, will retry w/ backoff delay')
            if attempt < max_retries - 1:
                # exponential backoff with jitter
                delay = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)

            else:
                raise Exception(f'Failed to read CSV {file_path} after {max_retries} attempts')

        raise Exception(f'Failed to read CSV {file_path} after {max_retries} attempts')

def save_failed_files(stage_no, batch_id, failed_files):
    """
    Save a list of failed file paths to a CSV named after the stage and batch.
    """
    with open(f's{stage_no}_b{batch_id}_failed_files.csv', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows([[x] for x in failed_files])


def main(
        stage_no,
        group,
        targets_file,
        run_table,
        batch_id,
        process_status_column, 
        update_interval=10
        ):
    """
    Run the de-duplication process.

    Parameters:
        stage_no (int) - pipeline stage number, used for naming output files and status columns
        group (str) - label for the collection being processed, used in progress messages
        targets_file (str) - path to CSV of files targeted for trimming
        run_table (str) - path to shared run table CSV that tracks progress across batches
        batch_id (int) - integer identifier for this batch run
        process_status_column (str) - name of the run table column to update with processing status
        update_interval (int) - number of files to process before syncing progress to disk (default: 10)
    """
    print(f'\nBeginning trimming process for audio files in {group} group: Batch {batch_id}')
    print(f'Reference tables will be updated every {update_interval} files')

    print(f'\nReading in target file and run table references')
    targets_df = pd.read_csv(targets_file)
    print(f'Found {len(targets_df)} total records in target file {targets_file}')
    
    # first, filter to only process files that have not succeeded
    # NOTE: we check in trim_file() if the target file is about the expected trim length,
    # so we should not trim an already-trimmed file, but better to be efficient about it
    initial_count = len(targets_df)
    targets_df = targets_df[targets_df['status'] != 'SUCCEEDED']
    print(f'Filtered to {len(targets_df)} files that need trimming (status != "SUCCEEDED")')
    print(f'Skipping {initial_count - len(targets_df)} files that already succeeded')
    
    if len(targets_df) == 0:
        print('No files need processing - all files have already succeeded!')
        return
    
    target_run_idxs = targets_df.run_table_row_idx.values.tolist()

    # read the run table to get the initial state (other jobs may have written to it!)
    run_table_df = safe_read(run_table)
    
    print(f'\nNow trimming {len(targets_df)} {group} audio files')
    try:
        # note that we're beginning the process for these target files
        # (if this job gets interrupted, this batch will stay as PROCESSING as an indicator)
        run_table_df.loc[target_run_idxs, process_status_column] = 'PROCESSING'
        
        # write the initial PROCESSING status to disk
        initial_updates = {idx: {process_status_column: 'PROCESSING'} for idx in target_run_idxs}
        safe_csv_update(run_table, initial_updates, batch_id, 'run')

        # process files and update table files periodically
        failed_files = []
        total_processed = 0
        processed_files = []
        for target_row_idx, target_row in targets_df.iterrows():
            print(f'\nProcessing file {total_processed + 1} of {len(targets_df)}')
            
            # *** HERE we're trimming the audio ***
            result = trim_file(target_row)

            # if it failed, add it to failed list for this batch
            if result == 'FAILED':
                failed_files.append(target_row.new_source_path)
            
            # regardless of result, add to our collection of processed files
            processed_files.append((
                target_row.file_id,
                result,
                target_row_idx,
                target_row.run_table_row_idx
            ))
            total_processed += 1
            
            # update tables if we've hit our update interval or if this is the last file
            if len(processed_files) >= update_interval or total_processed == len(targets_df):
                success = sync_table_files(
                    processed_files,
                    targets_df,
                    process_status_column,
                    targets_file,
                    run_table,
                    batch_id
                )
                
                if not success:
                    print(f'WARNING: Failed to update reference tables for collection of {update_interval} files ending at file number {total_processed} in targets list')
                
                processed_files = []
                print(f'Progress: {total_processed}/{len(targets_df)} files processed')

    except Exception as error:
        print(f'ERROR: failure during trimming process for group {group}, Batch {batch_id} | {error}')
        
        # try to update tables with whatever we've processed so far
        if processed_files:
            print(f'Attempting to save progress for {len(processed_files)} processed files before exiting')
            sync_table_files(
                processed_files,
                targets_df,
                process_status_column,
                targets_file,
                run_table,
                batch_id
            )

        # and save our currently-noted failed files before quitting
        if len(failed_files) > 0:
            save_failed_files(stage_no, batch_id, failed_files)

        sys.exit()

    print(f'\nCompleted backup process for {total_processed} files in {group} group: Batch {batch_id}')

    if len(failed_files) > 0:
        print(f'\n{len(failed_files)} files failed to be trimmed, saving these to CSV')
        save_failed_files(stage_no, batch_id, failed_files)

if __name__ == '__main__':
    args = parser.parse_args()
    targets_file = args.targets_file
    run_table = args.run_table
    group = args.group
    batch_id = args.batch_id

    update_interval = args.update_interval
    stage_no = 2
    process_status_column = f'stage_{stage_no}_process_status'

    print(f'\nStarting Stage {stage_no} of de-dupe process for batch {batch_id}: trimming files w/ duplicate audio')

    try:
        main(stage_no, group, targets_file, run_table, batch_id, process_status_column, update_interval)
        sys.exit(0)

    except Exception as error:
        print(f'ERROR: Stage {stage_no} Batch {batch_id} process failed | {error} | Exiting code 1', file=sys.stderr)
        sys.exit(1)