import pandas as pd
import requests
import xml.etree.ElementTree as ET

def get_hmdb_producers(hmdb_id):
    """
    Fetch microbial producers for a given HMDB ID
    """
    # Remove 'HMDB' prefix if present
    hmdb_id = hmdb_id.replace('HMDB', '')
    
    # HMDB API URL
    url = f"http://www.hmdb.ca/metabolites/HMDB{hmdb_id}.xml"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            
            # Extract biological sources
            sources = []
            for source in root.findall(".//biological_properties/synthesis_sources"):
                sources.extend([s.text for s in source.findall(".//source")])
            
            # Extract pathways that might indicate microbial involvement
            pathways = []
            for pathway in root.findall(".//biological_properties/pathways/pathway"):
                pathway_name = pathway.find("name").text
                if any(term in pathway_name.lower() for term in ["bacterial", "microbial", "fermentation"]):
                    pathways.append(pathway_name)
            
            return {
                'hmdb_id': f"HMDB{hmdb_id}",
                'biological_sources': sources,
                'microbial_pathways': pathways
            }
            
    except Exception as e:
        print(f"Error processing HMDB{hmdb_id}: {str(e)}")
        return None

def process_hmdb_list(hmdb_ids):
    """
    Process a list of HMDB IDs and create a mapping DataFrame
    """
    results = []
    for hmdb_id in hmdb_ids:
        result = get_hmdb_producers(hmdb_id)
        if result:
            results.append(result)
    
    return pd.DataFrame(results)

# Function to merge with AGORA2 data if available
def merge_with_agora(hmdb_df, agora_file):
    """
    Merge HMDB data with AGORA2 metabolic model data
    """
    agora_data = pd.read_csv(agora_file)
    # Perform merge based on metabolite IDs
    merged_df = pd.merge(hmdb_df, agora_data, on='hmdb_id', how='left')
    return merged_df

# Your list of HMDB IDs
hmdb_ids = ['HMDB0000002', 'HMDB0000011']  # Your 3000 IDs here

# Process in batches of 100
batch_size = 100
all_results = []

for i in range(0, len(hmdb_ids), batch_size):
    batch = hmdb_ids[i:i + batch_size]
    batch_results = process_hmdb_list(batch)
    all_results.append(batch_results)

# Combine all results
final_df = pd.concat(all_results)

# Save to CSV
final_df.to_csv('metabolite_producers_mapping.csv', index=False)