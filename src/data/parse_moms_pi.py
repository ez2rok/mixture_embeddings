import pickle
import pandas as pd
import argparse

from util.data_handling.data_loader import load_dataset


def load(metadata_path, manifest_path):
    metadata = pd.read_csv(metadata_path, sep='\t')
    manifest = pd.read_csv(manifest_path, sep='\t')
    return metadata, manifest


def get_sample_data_to_sample_id(metadata, manifest):
    """map data about the sample (subject_id, visit_number, sample_body_site) to
    sample_id. This allows us to map the mixture embeddings to its metadata."""
    
    sample_data_to_sample_id = {}
    for _, row in metadata.iterrows():
        sample = row['subject_id'] + '_' + str(row['visit_number']).zfill(2) + '_' + row['sample_body_site'].replace(' ', '_')
        sample_data_to_sample_id[sample] = row['sample_id']
    return sample_data_to_sample_id


def save(sample_data_to_sample_id, path):
    with open(path, 'wb') as f:
        pickle.dump(sample_data_to_sample_id, f)
    return path


def main(metadata_path, manifest_path, out_path):
    """Map the sample data (subject_id, visit_number, sample_body_site) to the sample id."""
    
    metadata, manifest = load(metadata_path, manifest_path)
    sample_data_to_sample_id = get_sample_data_to_sample_id(metadata, manifest)
    save(sample_data_to_sample_id, out_path)
    return sample_data_to_sample_id
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_path', type=str, default='data/interim/moms_pi/16s_metadata.tsv')
    parser.add_argument('--manifest_path', type=str, default='data/interim/moms_pi/16s_manifest.tsv')
    parser.add_argument('--out_path', type=str, default='data/interim/moms_pi/sample_data_to_sample_id.pickle')
    args = parser.parse_args()
    
    main(args.metadata_path, args.manifest_path, args.out_path)