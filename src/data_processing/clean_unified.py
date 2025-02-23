import pandas as pd
import json
from tqdm import tqdm
import numpy as np
import os

def process_chunk(chunk):
    """Process a single chunk of the dataframe."""
    # Group by HMDB ID and name within the chunk
    results = []
    for _, group in chunk.groupby(['hmdb_id', 'name']):
        # Filter and combine Transport_direction
        transport_dirs = set(group['Transport_direction'].dropna())
        transport_dirs.discard('unknown')
        transport_dirs.discard('')
        combined_transport = ' | '.join(sorted(transport_dirs)) if transport_dirs else 'unknown'
        
        # Filter and combine Direction
        directions = set(group['Direction'].dropna())
        directions.discard('unknown')
        directions.discard('')
        combined_direction = ' | '.join(sorted(directions)) if directions else 'unknown'
        
        # Take the first row as base and update transport info
        result = group.iloc[0].copy()
        result['Transport_direction'] = combined_transport
        result['Direction'] = combined_direction
        
        results.append(result)
    
    return pd.DataFrame(results)

def main():
    input_file = "data/cleaned/unified_dataset.csv"
    output_file = "data/cleaned/unified_dataset_cleaned.csv"
    chunk_size = 8000  # Adjust this based on your RAM
    
    print("Processing large file in chunks...")
    
    # Get file size for progress estimation
    file_size = os.path.getsize(input_file)
    print(f"File size: {file_size / (1024*1024*1024):.2f} GB")
    
    # Write header first
    first_chunk = next(pd.read_csv(input_file, chunksize=1, encoding='utf-8'))
    first_chunk.head(0).to_csv(output_file, index=False, encoding='utf-8')
    
    chunks_processed = 0
    rows_processed = 0
    
    # Process chunks with progress bar
    with tqdm(total=file_size, unit='B', unit_scale=True, desc="Processing file") as pbar:
        for chunk in pd.read_csv(input_file, chunksize=chunk_size, encoding='utf-8'):
            # Update progress based on chunk size
            chunk_rows = len(chunk)
            rows_processed += chunk_rows
            
            # Process the chunk
            processed_chunk = process_chunk(chunk)
            
            # Filter out records where both transport fields are unknown/empty
            processed_chunk = processed_chunk[
                ~((processed_chunk['Transport_direction'].isin(['unknown', ''])) & 
                  (processed_chunk['Direction'].isin(['unknown', ''])))
            ]
            
            # Append to file
            processed_chunk.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8')
            
            # Update progress
            chunks_processed += 1
            pbar.update(file_size / (rows_processed / chunk_rows))
            
            if chunks_processed % 10 == 0:
                print(f"\nProcessed {chunks_processed} chunks ({rows_processed:,} rows)")
    
    print("\nProcessing complete!")
    print(f"Results saved to: {output_file}")
    print(f"Total chunks processed: {chunks_processed}")
    print(f"Total rows processed: {rows_processed:,}")
    
    # Print some statistics from the final file
    print("\nReading sample from processed file for statistics...")
    sample_df = pd.read_csv(output_file, nrows=1000, encoding='utf-8')
    
    print("\nSample of merged transport directions:")
    print(sample_df[['name', 'Transport_direction', 'Direction']].head())
    
    print("\nUnique transport direction combinations (from sample):")
    print(sample_df['Transport_direction'].value_counts().head())
    
    print("\nUnique direction combinations (from sample):")
    print(sample_df['Direction'].value_counts().head())

if __name__ == "__main__":
    main()