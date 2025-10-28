# Curator Parquet Converter

This script converts Curator parquet files with their metadata into a combined format. It supports both local filesystem and S3 storage.

For more details, see the [design doc](https://docs.google.com/document/d/1ORk_9Dw6XtYkYto9ai2OrwcAg1mUp_9_6qZfnCDk5SA/edit?usp=sharing).

## Usage

Run the script from the workspace root directory:

```bash
# For local files
PYTHONPATH=$PYTHONPATH:. python3 src/visual_search/v1/apis/utils/curator_parquet_converter.py <base_dir> <output_path>

# For S3 files
PYTHONPATH=$PYTHONPATH:. python3 src/visual_search/v1/apis/utils/curator_parquet_converter.py <s3_uri> <output_path> --source-type s3
```

### Arguments

- `base_dir`: Base directory containing all data (local path or S3 URI)
- `output_path`: Where to save the output parquet file
- `--source-type`: Either "local" or "s3" (default: "local")

### Examples

```bash
# Process local files
PYTHONPATH=$PYTHONPATH:. python3 src/visual_search/v1/apis/utils/curator_parquet_converter.py /path/to/data output.parquet

# Process S3 files
PYTHONPATH=$PYTHONPATH:. python3 src/visual_search/v1/apis/utils/curator_parquet_converter.py s3://bucket/path output.parquet --source-type s3
```

## Requirements

- Python 3.x
- pandas
- s3fs (for S3 support)
