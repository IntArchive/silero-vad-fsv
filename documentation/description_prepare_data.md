# Description of `prepare_data.py`

This document describes the implementation details of `prepare_data.py`, a script that converts JSON annotation files into Feather format with standardized columns for audio path and speech timestamps.

## Overview

The script processes JSON files containing audio annotations and converts them into a Feather format suitable for training and validation. It handles path modifications, timestamp extraction, file validation, and error handling.

---

## 1. Audio Path Modification

The `modify_audio_path` function transforms audio file paths by extracting the filename from the original path and prepending a new parent directory path. This allows relocating audio files while maintaining references in the dataset.

```python fold title:modify_audio_path
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
```

**Key Features:**
- Extracts the filename from the original path using string splitting
- Uses `os.path.join()` for cross-platform path construction
- Modifies the row in-place and returns it

---

## 2. Speech Timestamp Extraction

The `extract_speech_timestamps` function processes label annotations to extract only the essential timing information (start and end times), removing metadata like channel and label types that are not needed for the final dataset.

```python fold title:extract_speech_timestamps
def extract_speech_timestamps(label: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """
    Extract speech timestamps from label format.
    
    Args:
        label: List of dictionaries with 'start' and 'end' keys
        
    Returns:
        List of dictionaries with only 'start' and 'end' keys
    """
    return [{'start': item['start'], 'end': item['end']} for item in label]
```

**Key Features:**
- Filters label dictionaries to keep only `start` and `end` keys
- Returns a clean list of timestamp dictionaries
- Preserves the original list structure

---

## 3. File Existence Validation

The `check_audio_files_exist` function validates that all audio files referenced in the dataset actually exist on the filesystem. This prevents errors during training by catching missing files early.

```python fold title:check_audio_files_exist
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
```

**Key Features:**
- Iterates through all audio paths in the dataframe
- Uses `os.path.exists()` to check file existence
- Returns both a boolean status and a list of missing file indices
- Allows for selective handling of missing files

---

## 4. Main Data Preparation Pipeline

The `prepare_data` function orchestrates the entire data preparation workflow, from reading the JSON file to saving the final Feather file. It includes comprehensive error handling and user feedback.

```python fold title:prepare_data
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
    df['label'] = df['label'].apply(extract_speech_timestamps)
    
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
        
        df.to_feather(output_feather, index=False)
        print(f"✓ Successfully saved {len(df)} rows to {output_feather}")
    except Exception as e:
        print(f"Error saving feather file: {e}", file=sys.stderr)
        sys.exit(1)
```

**Key Features:**
- **JSON Reading**: Loads the input JSON file with error handling
- **Path Modification**: Updates all audio paths using the parent path
- **Timestamp Extraction**: Cleans label data to keep only start/end times
- **Column Standardization**: Renames columns to `audio_path` and `speech_ts`
- **File Validation**: Optionally checks file existence with detailed reporting
- **Missing File Handling**: Can skip missing files or exit with error
- **Output Directory Creation**: Automatically creates output directories if needed
- **Progress Feedback**: Provides informative messages throughout the process

---

## 5. Command-Line Interface with argparse

The `main` function implements a comprehensive command-line interface using `argparse`, making the script reusable with different configurations.

```python fold title:main
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
```

**Key Features:**
- **Positional Arguments**: `input_json` and `output_feather` for required inputs
- **Required Optional Argument**: `--parent-path` must be specified
- **Mutually Exclusive Flags**: `--check-files` and `--no-check-files` control file validation
- **Flexible Options**: `--skip-missing` allows graceful handling of missing files
- **Input Validation**: Checks if input JSON file exists before processing
- **Help Documentation**: Includes examples in the epilog for user guidance
- **Error Handling**: Exits with appropriate error codes and messages

---

## Summary

The `prepare_data.py` script provides a robust, reusable solution for converting JSON annotation files to Feather format. It includes:

1. **Path Management**: Flexible audio path modification for different directory structures
2. **Data Cleaning**: Extracts only necessary timestamp information
3. **Validation**: Comprehensive file existence checking with detailed reporting
4. **Error Handling**: Graceful error handling with informative messages
5. **CLI Interface**: User-friendly command-line interface with argparse
6. **Flexibility**: Multiple options for handling edge cases (missing files, validation)

The script is designed to be production-ready, with proper error handling, progress feedback, and documentation.

