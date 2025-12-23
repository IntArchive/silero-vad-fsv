#!/usr/bin/env python3
"""
Prepare data.feather file from JSON input with audio_path and speech_ts columns.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd

COMMAND_TO_RUN = \
"""
# Basic usage
python prepare_data.py ./datasets/vad1/data.json ./datasets/vad1/data.feather \
    --parent-path /home/haidm/Documents/My_git/silero-vad-fsv/datasets/vad1/files/vfva

# Skip missing files instead of erroring
python prepare_data.py ./datasets/vad1/data.json ./datasets/vad1/data.feather \
    --parent-path /path/to/audio/files --skip-missing

# Disable file checking (faster, but no validation)
python prepare_data.py ./datasets/vad1/data.json ./datasets/vad1/data.feather \
    --parent-path /path/to/audio/files --no-check-files
    
"""

def modify_audio_path(row: pd.Series, parent_path: str) -> pd.Series:
    """
    Modify audio path by prepending parent_path and extracting filename.
    
    Args:
        row: DataFrame row containing 'audio' column
        parent_path: Base path to prepend to audio filename
        
    Returns:
        Modified row with updated 'audio' path
    """
    # Extract filename from original path
    filename = row['audio'].split('/')[-1]
    # Construct new path
    row['audio'] = os.path.join(parent_path, filename)
    return row


def extract_speech_timestamps(label: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """
    Extract speech timestamps from label format.
    
    Args:
        label: List of dictionaries with 'start' and 'end' keys
        
    Returns:
        List of dictionaries with only 'start' and 'end' keys
    """
    return [{'start': item['start'], 'end': item['end']} for item in label]


def check_audio_files_exist(df: pd.DataFrame, audio_path_col: str = 'audio_path') -> Tuple[bool, List[int]]:
    """
    Check if all audio files in the dataframe exist.
    
    Args:
        df: DataFrame containing audio paths
        audio_path_col: Name of the column containing audio paths
        
    Returns:
        Tuple of (all_exist: bool, missing_indices: List[int])
    """
    missing_indices = []
    for idx, audio_path in enumerate(df[audio_path_col]):
        if not os.path.exists(audio_path):
            missing_indices.append(idx)
    
    return len(missing_indices) == 0, missing_indices


def prepare_data(
    input_json: str,
    output_feather: str,
    parent_path: str,
    check_files: bool = True,
    skip_missing: bool = False
) -> None:
    """
    Prepare data.feather file from JSON input.
    
    Args:
        input_json: Path to input JSON file
        output_feather: Path to output feather file
        parent_path: Base path for audio files
        check_files: Whether to check if audio files exist
        skip_missing: Whether to skip rows with missing audio files
    """
    # Read JSON file
    print(f"Reading JSON file: {input_json}")
    try:
        df = pd.read_json(input_json)
    except Exception as e:
        print(f"Error reading JSON file: {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loaded {len(df)} rows from JSON file")
    
    # Modify audio paths
    print(f"Modifying audio paths with parent path: {parent_path}")
    df = df.apply(lambda row: modify_audio_path(row, parent_path), axis=1)
    
    # Extract speech timestamps (keep only start and end)
    print("Extracting speech timestamps...")
    
    try:
        df['label'] = df['label'].apply(extract_speech_timestamps)
    except Exception as e:
        print(df['label'].head())
    # Rename columns
    df = df.rename(columns={'audio': 'audio_path', 'label': 'speech_ts'})
    
    # Select only required columns
    df = df[['audio_path', 'speech_ts']]
    
    # Check file existence if requested
    if check_files:
        print("Checking if audio files exist...")
        all_exist, missing_indices = check_audio_files_exist(df)
        
        if not all_exist:
            missing_count = len(missing_indices)
            print(f"Warning: {missing_count} audio file(s) are missing:")
            
            # Show first 10 missing files
            for idx in missing_indices[:10]:
                print(f"  Row {idx}: {df.at[idx, 'audio_path']}")
            if missing_count > 10:
                print(f"  ... and {missing_count - 10} more")
            
            if skip_missing:
                print(f"Removing {missing_count} rows with missing audio files...")
                df = df.drop(missing_indices).reset_index(drop=True)
                print(f"Remaining rows: {len(df)}")
            else:
                print("Error: Some audio files are missing. Use --skip-missing to remove them.", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"✓ All {len(df)} audio files exist")
    
    # Save to feather
    print(f"Saving to feather file: {output_feather}")
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_feather)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        df.to_feather(output_feather)
        print(f"✓ Successfully saved {len(df)} rows to {output_feather}")
    except Exception as e:
        print(f"Error saving feather file: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Prepare data.feather file from JSON input with audio_path and speech_ts columns.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python prepare_data.py input.json output.feather --parent-path /path/to/audio/files
  
  # Skip missing files
  python prepare_data.py input.json output.feather --parent-path /path/to/audio/files --skip-missing
  
  # Disable file existence check
  python prepare_data.py input.json output.feather --parent-path /path/to/audio/files --no-check-files
        """
    )
    
    parser.add_argument(
        'input_json',
        type=str,
        help='Path to input JSON file'
    )
    
    parser.add_argument(
        'output_feather',
        type=str,
        help='Path to output feather file'
    )
    
    parser.add_argument(
        '--parent-path',
        type=str,
        required=True,
        help='Base path for audio files (will be prepended to filenames from JSON)'
    )
    
    parser.add_argument(
        '--check-files',
        action='store_true',
        default=True,
        help='Check if all audio files exist (default: True)'
    )
    
    parser.add_argument(
        '--no-check-files',
        dest='check_files',
        action='store_false',
        help='Disable file existence checking'
    )
    
    parser.add_argument(
        '--skip-missing',
        action='store_true',
        help='Skip rows with missing audio files instead of exiting with error'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input_json):
        print(f"Error: Input JSON file not found: {args.input_json}", file=sys.stderr)
        sys.exit(1)
    
    prepare_data(
        input_json=args.input_json,
        output_feather=args.output_feather,
        parent_path=args.parent_path,
        check_files=args.check_files,
        skip_missing=args.skip_missing
    )


if __name__ == '__main__':
    main()

