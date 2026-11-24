"""
Performs Dialogue Act Segmentation (DASeg) on word-level transcript files produced by the
GBL transcription and diarization pipeline. This script applies Zelasko et al.'s (2021) 
longformer-based model to assign a DASeg class to each word, then writes a CSV containing
the original transcript columns plus the DASeg labels and confidence scores.

~~~ Key run arguments ~~~

    Required (use one or the other, not both):
        --directory         path to a directory of word-level transcript CSV files
        --file                     path to a single word-level transcript CSV file

    Optional:
        --output_dir        directory where output files will be saved (default: current directory)

~~~ Example commands ~~~

    Single file:
        python daseg_pipeline.py \\
            --file /path/to/transcripts/0001_wordlevel_transcript.csv \\
            --output_dir /path/to/output

    Directory of files:
        python daseg_pipeline.py \\
            --directory /path/to/transcripts \\
            --output_dir /path/to/output

~~~ Run notes ~~~

Output filenames follow the pattern: <input_stem>_with_predictions.csv

The script checks the output directory for already-processed files and skips them,
so it is safe to interrupt and re-run.

By default, the model runs on CPU. To use a GPU, change device=-1 to device=0 in the
module-level pipeline and tokenizer initializations near the top of this file.

~~~ Attribution ~~~

Original author: Skyler Heininger
Modifications:   kirsten.bonson@uvm.edu (March 2026)
    - replaced file matching by substring with StoryCorps file ID matching
    - added keep_default_na=False to pandas read statements
    - added progress logging

Model: Zelasko et al. (2021). "Discovering the right moment to ask a question."
       pzelasko/longformer-swda-nolower on Hugging Face (no token required).
"""

import sys
import os
import argparse

import pandas as pd
from transformers import pipeline, AutoTokenizer
from datasets import Dataset
import numpy as np
import re

# Initialize the pipeline for token classification.
# Default is CPU (device=-1). To use a GPU, change device=-1 to device=0 in both lines below.
pipe = pipeline("token-classification", model="pzelasko/longformer-swda-nolower", device=-1)
tokenizer = AutoTokenizer.from_pretrained("pzelasko/longformer-swda-nolower", device=-1)

# When False (recommended), transcripts are split into 512-character windows before being passed
# to the model. Setting this to True passes the full transcript at once, which can cause the
# pipeline to silently stop producing output on long inputs.
IGNORE_MAX_LENGTH = False


def concatenate_words_to_length(words, max_length=512):
    """
    Concatenate words until reaching the maximum number of characters of `max_length`. 
    This was done somewhat informally, using the number of characters, because it just 
    seemed to work. Consider changing this in the future when comparing to ground-truthed labels.

    Option to ignore maximum length and concatenate all words together, using 
    IGNORE_MAX_LENGTH global constant.

    :param words: List of strings
    "param max_length: Maximum number of characters in sequence, NOT number of tokens or words.
    """
    windows = []
    current_window = []
    current_length = 0

    if IGNORE_MAX_LENGTH:
        all_words = " ".join(map(str, words))
        return [all_words]

    for word in words:
        word = str(word)
        if current_length + len(word) + 1 > max_length:  # +1 for space or punctuation
            windows.append(" ".join(current_window))
            current_window = [word]
            current_length = len(word)

        else:
            current_window.append(word)
            current_length += len(word) + 1

    if current_window:
        windows.append(" ".join(current_window))

    return windows


def clean_prediction(prediction, chars_to_remove=None):
    """
    Used to clean the word level of a prediction to ensure there are no prediction
    'artifacts' left over (default chars_to_remove has the general artifacts). This is because
    daseg will for some reason output weird characters with predictions.

    :param prediction: dictionary output of daseg, for a single token
    :param chars_to_remove: Keep as None to remove common prediction artifacts.
    """
    if chars_to_remove is None:
        chars_to_remove = ['Ä', 'Ġ', ' ']

    original_word = prediction['word']
    original_word = str(original_word).replace("â€”", '')  # Remove another problem sequence

    # Remove unwanted characters but keep spaces
    cleaned_word = ''.join(char for char in original_word if char not in chars_to_remove)
    prediction['word'] = cleaned_word

    return prediction, cleaned_word


def scrub_word(word):
    """
    Scrub the word by removing non-ASCII characters and other unwanted symbols.
    Daseg struggles with non-ASCII characters and will sometimes crash, avoid using those.
    """
    word = re.sub(r'[^\x00-\x7F]+', '', word)
    return word


def align_predictions_with_words_using_word_endings(words, predictions):
    """
    Aligns predictions with words using the ends of words. This generally works well, 
    but again an approach using a tokenizer would work better (the issue is the tokenizer 
    is not consistent with the pipeline object). So, using endings of words allows us to tell 
    what token matches the ending of a word.

    :param words: List of Strings
    :param predictions: list of dictionaries, each dictionary is an output from daseg for a token
    """
    # Note after the fact: I am unsure as to why I made a complete_predictions list, I must have made it for a different
    # version of the pipeline
    complete_predictions = []
    for predictions_list in predictions:
        complete_predictions.extend(predictions_list)

    aligned_predictions = []

    prediction_index = 0
    num_predictions = len(complete_predictions)

    for word in words:
        word = str(word)
        word_predictions = []
        current_concat = ""

        cleaned_word = scrub_word(word)

        # Continue processing while there are still predictions to check
        while prediction_index < num_predictions:
            # Remove unwanted characters from prediction
            prediction = complete_predictions[prediction_index]
            prediction, cleaned_pred = clean_prediction(prediction)

            # Word has nothing in it, skip
            if not cleaned_pred:
                prediction_index += 1
                continue

            if cleaned_word.endswith(cleaned_pred):
                # If the prediction matches the end of the current word segment
                word_predictions.append(prediction)
                current_concat += cleaned_pred
                prediction_index += 1
                break

            word_predictions.append(prediction)
            prediction_index += 1

        # If there are no matches, append None or an empty list
        if not word_predictions:
            aligned_predictions.append(None)

        else:
            aligned_predictions.append(word_predictions)

    return aligned_predictions


def pad_list_to_dataframe_length(data, lst):
    """
    Method for ensuring dataframe is the same length as a list
    :param data: dataframe
    :param lst: List
    """
    df_length = len(data)

    # Pad with empty strings if shorter
    if len(lst) < df_length:
        lst.extend([""] * (df_length - len(lst)))

    # Truncate if it's longer
    elif len(lst) > df_length:
        lst = lst[:df_length]

    return lst


def process_file(
        file_path,
        output_dir,
        all_words_split_by_window=None,
        pre_loaded=False,
        filename_if_preloaded=""
        ):
    """
    Opens file, loads words, uses daseg, aligns token level predictions with words, 
    determines how to replace I- entities with first non-I- label, and saves file to output_dir.
    
    I- replacement in this is done using the first non-I- label, rather than last label of a DA. 
    One recommended change would be to modify this to match the last label of a DA, rather than 
    first non-I- label. (In practice there seems to be no difference).

    Also, only the prediction for the first token of each word is used. This mirrors Zelasko's 
    usage. It is strongly recommended to not use the last token of each word, this will give highly unpredictable and incorrect predictions.

    :param file_path: String, filepath
    :param output_dir: String, will create output directory if it does not exist
    :param all_words_split_by_window: List of Strings, each string is a chunk of text
        (currently getting this using a sliding window, within process_file)
    :param pre_loaded: Boolean, If the data has already been loaded (If so, input data using file_path)
    :param filename_if_preloaded: String, If pre-loaded data, This is the output filename
    """
    if all_words_split_by_window is None:
        all_words_split_by_window = []

    # Including this already split variable in case data loading pipeline has already split the words
    # using a sliding window for instance
    already_split = False

    # Load the file (assuming it's a CSV or Excel file)
    print(f'Reading input transcript')
    if pre_loaded:
        data = file_path
        # already_split = True

    elif file_path.startswith('='):
        # Wrong type of CSV file
        print("Issue with csv file", flush=True)
        return
    
    elif file_path.endswith('.csv'):
        data = pd.read_csv(file_path, keep_default_na=False)

    elif file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path, keep_default_na=False)

    else:
        raise ValueError("Unsupported file format. Use CSV or Excel files.")

    # Remove rows with problem sequence (some of this parsing is likely redundant)
    # NOTE: dashes are currently dropped rather than replaced with spaces
    print(f'Removing rows with problem sequences')
    total_words_before_cleaning = len(data)
    data = data[~((data['text'] == 'â€”') | (data['text'] == '—') | (data['text'] == '–') | (data['text'] == '-'))]
    total_words_after_cleaning = len(data)

    if total_words_before_cleaning != total_words_after_cleaning:
        print(f'WARNING: {total_words_before_cleaning - total_words_after_cleaning} dashes or corrupt words dropped')

    words = data['text'].tolist()
    # And remove problem sequence from words list directly
    words = [str(word).replace('â€”', '').replace('—', '') for word in words]

    words = [scrub_word(word) for word in words]

    # Concatenate words using a sliding window
    print('Concatenating words using sliding window')
    if not already_split:
        all_words_split_by_window = concatenate_words_to_length(words)

    if not all_words_split_by_window:
        return None

    # Create a dataset from the sliding window list
    dataset = Dataset.from_dict({'text': all_words_split_by_window})

    # Apply the pipeline to the dataset in batches
    # Adjust batch size as needed (most likely would need to reduce batch size if not enough resources)
    print('Passing dataset through DASeg pipeline')
    predictions = pipe(dataset['text'], batch_size=16)

    # Map the predictions back to individual words
    print('Aligning predictions back to words')
    word_predictions = align_predictions_with_words_using_word_endings(words, predictions)
    print("Aligned")
    entities = []
    raw_entities = []
    scores = []
    words_pred = []
    chunks = []
    raw_scores = []
    specific_raw_counts = []

    chunk_index = 0
    previous_raw_label = "I-"
    score = 0

    temp_predictions = []

    for idx, pred_list in enumerate(word_predictions):
        if pred_list is None:
            chunk_index += 1
            entities.append("")
            raw_entities.append("")
            raw_scores.append("")
            scores.append("")
            chunks.append("")
            words_pred.append("")
            continue

        pred = pred_list[0]  # Modified from -1
        entity = pred['entity']
        score = pred['score']

        temp_word = ''
        for temp_pred in pred_list:
            temp_word += temp_pred['word']

        temp_predictions.append({
            'entity': entity,
            'raw_entity': entity,
            'score': score,
            'raw_score': score,
            'word': temp_word
        })

        if entity != previous_raw_label and previous_raw_label != "I-":
            chunk_index += 1

        if entity != 'I-':
            # This is the non-I- label, so we fill in all the previous I- labels with this one
            for temp_pred in temp_predictions:
                if temp_pred['entity'] == 'I-':
                    temp_pred['entity'] = entity
                    temp_pred['score'] = score

            # After filling, push all the filled predictions into the final results
            for temp_pred in temp_predictions:
                entities.append(temp_pred['entity'])
                raw_entities.append(temp_pred['raw_entity'])
                raw_scores.append(temp_pred['raw_score'])
                scores.append(temp_pred['score'])
                chunks.append(chunk_index)
                words_pred.append(temp_pred['word'])

            # Reset the temp_predictions list and increment the chunk index
            temp_predictions = []

        previous_raw_label = entity

    # Handle remaining entries by just adding them
    if temp_predictions:
        for temp_pred in temp_predictions:
            entities.append(temp_pred['entity'])
            raw_entities.append(temp_pred['raw_entity'])
            raw_scores.append(temp_pred['raw_score'])
            scores.append(temp_pred['score'])
            chunks.append(chunk_index)
            words_pred.append(temp_pred['word'])

    data['Raw DA Class'] = pad_list_to_dataframe_length(data, raw_entities)
    data['Raw Score'] = pad_list_to_dataframe_length(data, raw_scores)
    data['Proc DA Class'] = pad_list_to_dataframe_length(data, entities)
    data['Score'] = pad_list_to_dataframe_length(data, scores)
    data['Words Prediction'] = pad_list_to_dataframe_length(data, words_pred)
    data['Chunk'] = pad_list_to_dataframe_length(data, chunks)

    # Create a new filename for the output
    print('Creating ouput')
    if not pre_loaded:
        file_name, file_extension = os.path.splitext(file_path)

    else:
        file_name, file_extension = os.path.splitext(filename_if_preloaded)

    file_name = file_name.split("/")[-1]
    new_file_path = f"{file_name}_with_predictions.csv"
    save_file_path = f"{file_name}_count_plot"

    # Output to given dir
    if output_dir is not None:
        new_file_path = os.path.join(output_dir, new_file_path)
        save_file_path = os.path.join(output_dir, save_file_path)

    # Save the updated dataframe to a new CSV file
    data.to_csv(new_file_path, index=False)
    print(f"Predictions saved to {new_file_path}", flush=True)

    return specific_raw_counts


def get_id_from_fname(fname, transcript_tag, lowercase=False):
    """
    Extract the StoryCorps file ID from a transcript filename by stripping the transcript_tag
    suffix and taking the last underscore-separated component.

    :param fname: filename string
    :param transcript_tag: suffix to strip before extracting the ID (e.g. '_wordlevel_transcript.csv')
    :param lowercase: if True, return the ID in lowercase
    """
    fname_stem = fname.replace(transcript_tag, "")
    sc_id = fname_stem.split("_")[-1]

    if lowercase:
        return sc_id.lower()
    
    return sc_id

def main():
    """
    Loads transcript files from the given directory or single file path, skips any that have
    already been processed (by checking the output directory), and calls process_file on each
    remaining file.
    """
    parser = argparse.ArgumentParser(description="Token classification with Longformer pipeline.")

    parser.add_argument('--directory', type=str, help="Directory containing CSV or Excel transcript files")
    parser.add_argument('--file', type=str, help="Specific transcript file to process")
    parser.add_argument('--output_dir', type=str, default=None, help="Where output files will be stored")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.directory and not args.file:
        diarized = [os.path.join(args.directory, f) for f in os.listdir(args.directory)
                    if f.endswith('.csv') or f.endswith('.xlsx')]
        
        print(f"\n{len(diarized)} files were found within the given directory.")
        
    elif args.file and not args.directory:
        diarized = [args.file]
        print(f"\nFile given to be processed: {args.file}")

    else:
        print('\nPlease specify either --directory or --file, but not both')
        sys.exit()

    already_daseg = [os.path.join(args.output_dir, f) for f in os.listdir(args.output_dir)
                     if f.endswith('.csv') or f.endswith('.xlsx')]

    files_to_daseg = {}
    files_daseg = {}


    # Store filenames in dictionary
    for file in diarized:
        filename = os.path.basename(file)  # Get filename without path

        # NOTE: now we have StoryCorps file IDs in the file names that we can use!
        sc_file_id = get_id_from_fname(filename, transcript_tag='_wordlevel_transcript.csv')

        files_to_daseg[sc_file_id] = file

    for file in already_daseg:
        filename = os.path.basename(file)
        
        # NOTE: now we have StoryCorps file IDs in the file names that we can use!
        sc_file_id = get_id_from_fname(filename, transcript_tag='_wordlevel_transcript_with_predictions.csv')

        # Add the audio file to the dictionary where the key is the part
        files_daseg[sc_file_id] = file

    print(f"{len(files_daseg)} files were already in output directory.\nBeginning processing now.", flush=True)
    for key in files_to_daseg.keys():
        # Check that the file has not already been processed
        if key not in files_daseg.keys():
            try:
                # Process the file
                print('Processing file')
                process_file(files_to_daseg[key], args.output_dir)

            except Exception as e:
                print(e)

        else:
            print(f'File {key} has already been processed -- skipping!')
            continue



if __name__ == "__main__":
    main()