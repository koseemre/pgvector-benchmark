from util.data_util import data_util
import settings

def insert_data(dataset_names, dataset_paths):
    # writing to db
    for dataset_name in dataset_names:
        print("writing operation is being started")
        data_util.write_to_db(dataset_name, dataset_paths[dataset_name], convert=True, is_truncate=True)
        
    
if __name__ == '__main__':
    
    dataset_paths = settings.dataset_paths
    dataset_names = settings.dataset_names
    insert_data(dataset_names, dataset_paths)