"""
Prepares the run table for the audio de-duplication pipeline. This is a two-step script:

Step 1: Recursively collects all MP3 file names and paths from a given directory and saves
    the inventory to a dated CSV. File names are cleaned to remove non-ASCII characters and
    carriage returns. The cleaned name may differ from the original filename, but all matching
    throughout the pipeline is done by file ID (the last underscore-separated component of the
    filename). NOTE: we cleaned all other file names, that were not part of this de-duplication 
    effort, separately.

Step 2: Filters the inventory to only the files identified for de-duplication (from a provided
    targets CSV), adds pipeline management columns, and saves the result as the run table used
    by audio_deduplication.py.

NOTE: the update_path() function updates paths written using a local mount syntax to those
on our remote drive. This is specific to the Glass Brain Lab and would need to be changed 
or removed depending on your specific setup.

~~~ Key run arguments ~~~

    Required:
        --parent_dir                         path to the top-level directory containing audio files
        --targets_file                             path to CSV of files targeted for de-duplication
        --all_audio_file         path to inventory CSV of all audio files (filename, path columns);
                           can be the output of Step 1 from a prior run, or a pre-existing inventory

~~~ Example command ~~~

    python make_deduplication_run_table.py \\
        --parent_dir /path/to/audio \\
        --targets_file /path/to/files_to_deduplicate.csv \\
        --all_audio_file /path/to/all_audio_inventory.csv

~~~ Run Notes ~~~

Output files are saved to the current working directory:
    - sc_all_audio_paths_{date}.csv                       (inventory from Step 1)
    - sc_audio_deduplication_run_table_{date}.csv         (run table from Step 2)

The targets CSV must have a column named filepath containing paths to the targeted audio files.

~~~ Attribution ~~~

Author: kirsten.bonson@uvm.edu
"""
import os
import csv
import argparse
import pandas as pd

from datetime import datetime
from unidecode import unidecode

parser = argparse.ArgumentParser(
    "Given a path to a directory containing a nested folder structure, collect all MP3 files within"
)
parser.add_argument('-p', '--parent_dir', type=str, required=True,
                   help='Path to the top-level directory containing audio files')
parser.add_argument('--targets_file', type=str, required=True,
                   help='Path to CSV of files identified for de-duplication (must have a filepath column)')
parser.add_argument('--all_audio_file', type=str, required=True,
                   help='Path to inventory CSV of all audio files (filename, path columns); can be the output of Step 1 from a prior run')

def clean_fname(file_name):
    """
    Remove non-ASCII characters and strip carriage returns from a filename.
    """
    cleaned_name = unidecode(file_name)

    if "\n" in cleaned_name:
        cleaned_name = cleaned_name.replace("\n", "")

    return cleaned_name


def collect_all_files(top_level_dir):
    """
    Walk top_level_dir recursively and return a list of (cleaned_filename, directory_path) 
    tuples for all MP3 files found.
    """
    print(f'\nCollecting all file names and paths in {top_level_dir}...')
    
    contents = os.listdir(top_level_dir)
    folders = [x for x in contents if '.' not in x]
    folders.sort()  # no real need for this, I just prefer to watch the process in order
    
    # collect top-level files with their paths
    all_file_data = []
    top_level_files = [x for x in contents if '.mp3' in x]
    for filename in top_level_files:
        cleaned_name = clean_fname(filename)
        all_file_data.append((cleaned_name, top_level_dir))
    
    print(f'\nA total of {len(top_level_files)} file names found in {top_level_dir} at top-level')
    
    # we could do this all in a single call, but it takes a while with no terminal output,
    # so iterate at first alphabetical level and print progress instead
    for letter_folder in folders:
        top_letter_folder = f'{top_level_dir}/{letter_folder}'
        print(f'\nNow collecting file names recursively in {top_letter_folder}...')
        print(f' ...this might take a bit...')
        
        # get both filenames and their directory paths
        folder_file_data = []
        for root, dirs, filenames in os.walk(top_letter_folder):
            for f in filenames:
                if os.path.splitext(f)[1] == '.mp3':
                    cleaned_name = clean_fname(f)
                    folder_file_data.append((cleaned_name, root))
        
        print(f' ...{len(folder_file_data)} file names found in {top_letter_folder}, adding to the consolidated list')
        all_file_data += folder_file_data
        print(f' ...{len(all_file_data)} total file names collected overall so far')
    
    print(f'\nDone! {len(all_file_data)} file names found recursively in {top_level_dir}')
    
    return all_file_data

def get_file_id(given_string, name_only=False):
    """
    Extract the unique file ID from a filename or path.
    The ID is the last underscore-separated component of the filename, without extension.
    """
    if name_only:
        fname = given_string

    else:
        path, fname = os.path.split(given_string)

    name, ext = os.path.splitext(fname)
    file_id = name.split('_')[-1]

    return file_id

def update_path(file_path):
    """
    Clean the filename and update the directory path.
    
    NOTE: this path transformation ("Volumes" to "netfiles") is specific to the GBL setup
    and would need to be adapted for other environments.
    """
    path, fname = os.path.split(file_path)
    vacc_path = path.replace('Volumes', 'netfiles')
    cleaned_name = clean_fname(fname)

    return os.path.join(vacc_path, cleaned_name)


if __name__ == "__main__":
    args = parser.parse_args()
    parent_dir = args.parent_dir
    targets_file = args.targets_file
    all_audio_file = args.all_audio_file

    # Step 1: get all audio file names with paths
    all_file_data = collect_all_files(parent_dir)

    now = datetime.now()
    year = now.year
    month = now.month
    day = now.day

    with open(f'sc_all_audio_paths_{month}{day}{year}.csv', 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(['filename', 'path'])
        writer.writerows(all_file_data)

    # Step 2: filter to only files identified for de-duplication
    targets_df = pd.read_csv(targets_file)
    all_audio_df = pd.read_csv(all_audio_file)
    meta_df = all_audio_df.copy()

    # add ID only as a new targets column
    targets_df['file_id'] = targets_df.filepath.apply(get_file_id)
    target_ids = targets_df.file_id.tolist()
    meta_df['file_id'] = meta_df.filename.apply(get_file_id, args=(True,))

    # filter the metadata to include only rows with matching IDs
    rows_to_keep = meta_df[meta_df['file_id'].isin(target_ids)]
    rows_to_keep = rows_to_keep.reset_index(drop=True)

    # update paths to include file names
    rows_to_keep.path = rows_to_keep.apply(lambda row: os.path.join(row.path, row.filename), axis=1)

    # change Volumes to netfiles and scrub non-ascii characters
    rows_to_keep.path = rows_to_keep.path.apply(update_path)
    rows_to_keep = rows_to_keep[['file_id', 'filename', 'path']]

    # add columns we'll need for processing management
    new_columns = [
        'stage_1_batch_number',
        'stage_1_job_id',
        'stage_1_job_status',
        'stage_1_process_status',
        'backup_path',
        'stage_2_batch_number',
        'stage_2_job_id',
        'stage_2_job_status',
        'stage_2_process_status'
    ]
    rows_to_keep = rows_to_keep.assign(**dict([(new_column, None) for new_column in new_columns]))

    # save table
    rows_to_keep.to_csv(f'sc_audio_deduplication_run_table_{month}{day}{year}.csv', index=False)