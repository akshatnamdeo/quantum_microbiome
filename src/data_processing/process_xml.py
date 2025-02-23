import xml.etree.ElementTree as ET
import csv
import os
import time

def convert_metabolites_to_csv(xml_file, csv_file, batch_size=100):
    """Convert HMDB metabolites XML to CSV focusing on neurotransmitter/hormone/receptor/gene data"""
    
    # Define namespace
    ns = {'hmdb': 'http://www.hmdb.ca'}
    
    headers = [
        # ID and Basic Info
        'hmdb_id',           
        'name',
        'status',
        
        # Classification
        'compound_type',     
        'is_neurotransmitter', 
        'is_hormone',        
        
        # Gene/Protein Data
        'gene_names',        # All associated genes
        'protein_types',     # Types of proteins (enzyme, receptor, etc)
        'uniprot_ids',      # For protein cross-referencing
        
        # Biological Activity
        'brain_tissues',     
        'cellular_locations',
        'pathways',         
        
        # Search Terms (for classification)
        'description',      
        'taxonomy_description'
    ]
    
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        
        batch = []
        metabolites_processed = 0
        
        context = ET.iterparse(xml_file, events=('end',))
        
        for event, elem in context:
            if elem.tag.split('}')[-1] == 'metabolite':
                data = {}
                
                # Basic Info with namespace
                for field in ['accession', 'name', 'status']:
                    node = elem.find(f'.//hmdb:{field}', ns)
                    data['hmdb_id' if field == 'accession' else field] = (
                        node.text.strip() if node is not None and node.text else ''
                    )
                
                # Get descriptions
                for field in ['description', 'taxonomy']:
                    desc_path = f'.//hmdb:{field}/hmdb:description' if field == 'taxonomy' else f'.//hmdb:description'
                    node = elem.find(desc_path, ns)
                    data[f'{"taxonomy_" if field == "taxonomy" else ""}description'] = (
                        node.text.strip() if node is not None and node.text else ''
                    )
                
                # Determine compound type and classification
                desc_lower = data['description'].lower()
                tax_lower = data.get('taxonomy_description', '').lower()
                
                data['compound_type'] = ''
                
                # Look for neurotransmitter indicators
                neurotransmitter_terms = ['neurotransmitter', 'gaba', 'glutamate', 'serotonin', 
                                        'dopamine', 'acetylcholine', 'glycine', 'norepinephrine']
                data['is_neurotransmitter'] = 'Yes' if any(term in desc_lower or term in tax_lower 
                                                         for term in neurotransmitter_terms) else 'No'
                
                # Look for hormone indicators
                hormone_terms = ['hormone', 'steroid', 'peptide hormone', 'endocrine']
                data['is_hormone'] = 'Yes' if any(term in desc_lower or term in tax_lower 
                                               for term in hormone_terms) else 'No'
                
                if data['is_neurotransmitter'] == 'Yes':
                    data['compound_type'] = 'neurotransmitter'
                if data['is_hormone'] == 'Yes':
                    data['compound_type'] = data['compound_type'] + '|hormone' if data['compound_type'] else 'hormone'
                
                # Extract protein/gene data
                genes = []
                protein_types = []
                uniprot_ids = []
                
                for protein in elem.findall('.//hmdb:protein_associations/hmdb:protein', ns):
                    for field, lst in [('gene_name', genes), ('protein_type', protein_types), 
                                     ('uniprot_id', uniprot_ids)]:
                        node = protein.find(f'./hmdb:{field}', ns)
                        if node is not None and node.text:
                            lst.append(node.text.strip())
                
                data['gene_names'] = '|'.join(genes)
                data['protein_types'] = '|'.join(protein_types)
                data['uniprot_ids'] = '|'.join(uniprot_ids)
                
                # Get locations
                locations = elem.findall('.//hmdb:biological_properties/hmdb:cellular_locations/hmdb:cellular', ns)
                data['cellular_locations'] = '|'.join(loc.text.strip() for loc in locations if loc.text)
                
                # Get brain-specific tissues
                tissues = elem.findall('.//hmdb:biological_properties/hmdb:tissue_locations/hmdb:tissue', ns)
                brain_tissues = []
                for tissue in tissues:
                    if tissue.text and any(term in tissue.text.lower() for term in 
                                         ['brain', 'cortex', 'hippocampus', 'neuron', 'glial']):
                        brain_tissues.append(tissue.text.strip())
                data['brain_tissues'] = '|'.join(brain_tissues)
                
                # Get pathways
                pathways = elem.findall('.//hmdb:biological_properties/hmdb:pathways/hmdb:pathway/hmdb:name', ns)
                data['pathways'] = '|'.join(p.text.strip() for p in pathways if p.text)
                
                batch.append(data)
                metabolites_processed += 1
                
                if len(batch) >= batch_size:
                    writer.writerows(batch)
                    batch = []
                    print(f"Processed {metabolites_processed} metabolites")
                
                elem.clear()
        
        if batch:
            writer.writerows(batch)
        
        print(f"\nConversion complete! Processed {metabolites_processed} metabolites")

# File paths
xml_file = "data/hmdb_metabolites.xml"
csv_file = "data/processed/hmdb_neuro_hormone_gene_data.csv"

# Run conversion
convert_metabolites_to_csv(xml_file, csv_file)