import pandas as pd
from src.util.data_handling.data_loader import save_as_pickle 


def process_ibd(ibd_data_path, ibd_metadata_path):
    
    good_metadata = [
        'sample',
        'Participant ID',
        'Project',
        'External ID',
        'date_of_receipt',
        'ProjectSpecificID',
        'visit_num',
        'site_name',
        'consent_age', 
        'diagnosis', # UC = ulcerative colitis, CD = Crohn disease; https://emedicine.medscape.com/article/179037-overview
        'hbi', # Harvey-Bradshaw Index; https://globalrph.com/medcalcs/harvey-bradshaw-index-measuring-crohns-disease/
        'sex',
        'race',
        'fecalcal', # Fecal Calprotectin Test; https://www.verywellhealth.com/how-the-fecal-calprotectin-test-is-used-in-ibd-4140079
        'sccai', # Simple clinical colitis activity index; https://en.wikipedia.org/wiki/Simple_clinical_colitis_activity_index
        ]
    
    # load data
    ibd_data = pd.read_csv(ibd_data_path)
    ibd_metadata = pd.read_csv(ibd_metadata_path)
    
    # remove participants who did not complete the study
    mask = ibd_metadata['Did the subject withdraw from the study?'] == 'No'
    ibd_data = ibd_data[mask]
    ibd_metadata = ibd_metadata[mask]
    
    # only track interesting or good metadata
    ibd_metadata = ibd_metadata[good_metadata]
    ibd_metadata = ibd_metadata.fillna(0)

    # rename columns
    ibd_data.rename(columns={'sample': 'sample id'}, inplace=True) 
    ibd_metadata.rename(columns={'sample': 'sample id'}, inplace=True)
    
    # make sample id the index
    ibd_data = ibd_data.set_index('sample id')
    ibd_metadata = ibd_metadata.set_index('sample id')
    
    return ibd_data, ibd_metadata


def process_t2d(t2d_data_path, t2d_metadata_path):

    # read files
    t2d_data = pd.read_csv(t2d_data_path)
    t2d_metadata = pd.read_csv(t2d_metadata_path)

    # rename columns
    t2d_data.rename(columns={'VisitID': 'sample id'}, inplace=True) 
    t2d_metadata.rename(columns={'VisitID': 'sample id'}, inplace=True) 
    
    # make sample id the index
    t2d_data = t2d_data.set_index('sample id')
    t2d_metadata = t2d_metadata.set_index('sample id')
    
    return t2d_data, t2d_metadata


def process_moms(moms_data_path, moms_metadata_path):
    
    good_metadata = [
        'sample id',
        'sample_body_site',
        'subject_id',
        'visit_number',
        'sample'
        ]

    # load files
    moms_data = pd.read_csv(moms_data_path)
    moms_metadata = pd.read_csv(moms_metadata_path)

    # select good metadata
    moms_data = moms_data.drop(columns=['site', 'patient', 'visit'])
    moms_data.insert(loc=0, column='sample id', value=moms_metadata['sample_id'])

    sample_id = moms_metadata['sample_id']
    moms_metadata = moms_metadata.drop(columns=['sample_id'])
    moms_metadata.insert(0, column='sample id', value=sample_id)
    moms_metadata = moms_metadata[good_metadata]
    
    # make sample id the index
    moms_data = moms_data.set_index('sample id')
    moms_metadata = moms_metadata.set_index('sample id')
    
    return moms_data, moms_metadata


def main(ihmp_dir, outdir):
    
    # get file paths
    ibd_data_path = ihmp_dir + '/ibd_data.csv'
    ibd_metadata_path = ihmp_dir + '/ibd_metadata.csv'
    t2d_data_path = ihmp_dir + '/t2d_data.csv'
    t2d_metadata_path = ihmp_dir + '/t2d_metadata.csv'
    moms_data_path = ihmp_dir + '/moms_data.csv'
    moms_metadata_path = ihmp_dir + '/moms_metadata.csv'
    
    # process the data
    print('Processing ibd data...', end='\t\t')
    ibd_data, ibd_metadata = process_ibd(ibd_data_path, ibd_metadata_path)
    print('Done.')
    
    print('Processing t2d data...', end='\t\t')
    t2d_data, t2d_metadata = process_t2d(t2d_data_path, t2d_metadata_path)
    print('Done.')
    
    print('Processing moms data...', end='\t\t')
    moms_data, moms_metadata = process_moms(moms_data_path, moms_metadata_path)
    print('Done.')
    
    # save the data
    name_to_object = {
        'ibd_data': ibd_data, 
        'ibd_metadata': ibd_metadata,
        't2d_data': t2d_data, 
        't2d_metadata': t2d_metadata,
        'moms_data': moms_data, 
        'moms_metadata': moms_metadata
    }
    for name, obj in name_to_object.items():
        save_as_pickle(obj, '{}/{}.pickle'.format(outdir, name))
        
    return name_to_object