import pickle
import pandas as pd
import argparse

from util.data_handling.data_loader import load_dataset


def load(auxillary_data_path, otu_tables_path, metadata_path, manifest_path):
    auxillary_data = load_dataset(auxillary_data_path)
    id_to_str_seq = auxillary_data[0]
    otu_tables = load_dataset(otu_tables_path)
    metadata = pd.read_csv(metadata_path, sep='\t')
    manifest = pd.read_csv(manifest_path, sep='\t')
    return otu_tables, metadata, manifest, id_to_str_seq


def get_sample_data_to_sample_id(metadata, manifest):
    """map data about the sample (subject_id, visit_number, sample_body_site) to
    sample_id. This allows us to map the mixture embeddings to its metadata."""
    
    sample_data_to_sample_id = {}
    for _, row in metadata.iterrows():
        sample = row['subject_id'] + '_' + str(row['visit_number']).zfill(2) + '_' + row['sample_body_site'].replace(' ', '_')
        sample_data_to_sample_id[sample] = row['sample_id']
    return sample_data_to_sample_id


def normalize_otu_table(otu_tables):
    return {otu_type: otu_table.norm() for otu_type, otu_table in otu_tables.items()}


def drop_missing_ids(otu_tables, id_to_str_seq, verbose=False):
    "Remove ids from the OTU matrix that are not found in the greengenes dataset."
    
    otu_tables_cleaned = {}
    for otu_type, otu_table in otu_tables.items():
        
        ids = otu_table.ids(axis='observation')
        valid_ids = [id_ for id_ in ids if int(id_) in id_to_str_seq]
        otu_table_valid = otu_table.filter(valid_ids, axis='observation', inplace=False)
        otu_tables_cleaned[otu_type] = otu_table_valid
        
        if verbose:
            valid_ratio = len(valid_ids) / len(ids)
            print("{}: drop {:.2%} of OTUs (because their IDs aren't in greengenes)".format(otu_type, 1 - valid_ratio))
    return otu_tables_cleaned


def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    return path


def main(auxillary_data_path, otu_tables_path, metadata_path, manifest_path, out_dir):
    """ Normalize the OTU tables and remove ids that are not found in the greengenes dataset.
    Map the sample data (subject_id, visit_number, sample_body_site) to the sample id."""
    
    results = load(auxillary_data_path, otu_tables_path, metadata_path, manifest_path)
    otu_tables, metadata, manifest, id_to_str_seq = results
    
    otu_tables_normed = normalize_otu_table(otu_tables)
    otu_tables_normed_and_cleaned = drop_missing_ids(otu_tables_normed, id_to_str_seq, verbose=True)    
    sample_data_to_sample_id = get_sample_data_to_sample_id(metadata, manifest)
    
    save(otu_tables_normed_and_cleaned, out_dir + 'otu_tables_normed_and_cleaned.pickle')
    save(sample_data_to_sample_id, out_dir + 'sample_data_to_sample_id.pickle')
    return otu_tables_normed_and_cleaned, sample_data_to_sample_id
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--auxillary_data_path', type=str, default='data/interim/greengenes/auxillary_data.pickle')
    parser.add_argument('--otu_tables_path', type=str, default='data/interim/moms_pi/16s_tables.pkl')
    parser.add_argument('--metadata_path', type=str, default='data/interim/moms_pi/16s_metadata.tsv')
    parser.add_argument('--manifest_path', type=str, default='data/interim/moms_pi/16s_manifest.tsv')
    parser.add_argument('--out_dir', type=str, default='data/interim/moms_pi/')
    args = parser.parse_args()
    
    main(args.auxillary_data_path, args.otu_tables_path, args.metadata_path, args.manifest_path, args.out_dir)