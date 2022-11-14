import pickle
import os


def save_as_pickle(obj, filename='../../data/processed/greengenes/'):
    "save obj as a pickle file"
    
    directory = os.path.dirname(filename)
    if directory != '' and not os.path.exists(directory):
        os.makedirs(directory)

    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    print("\tSaved", filename)
            
    return filename