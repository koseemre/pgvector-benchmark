from service.pgvector_service import pgvector_service
from service.test_service import metrics
from util.data_util import data_util
import settings

import numpy as np
import pandas as pd
import json, time, os

def process_experiment(dataset_name, create_index, dataset_path, vector_type, index_type, partial_initial_indexing, indexing_param):
    
    # load dataset
    print("dataset_path:", dataset_path, " vector_type:", vector_type, " index_type:", index_type)
    dataset, dimension = data_util.get_dataset(dataset_path)
                
    start = time.time()
    indexing_stat = {}
    
    if create_index:
        print("Creating index for dataset:", dataset_name)
        print(dataset_name, "dataset training length:", len(dataset['train']), " with dimension:", dimension) 
        indexing_params = pgvector_service.create_index(dataset_name,
                                dataset_length = len(dataset['train']),
                                vector_type = vector_type,
                                index_type = index_type,
                                partial_initial_indexing = partial_initial_indexing,
                                indexing_param = indexing_param)
    
        indexing_stat["indexing_duration"] = time.time() - start
        indexing_stat["index_type"] = index_type
        indexing_stat["vector_type"] = vector_type
        indexing_stat["indexing_params"] = indexing_params
        
        if partial_initial_indexing[0]:
            print("partial_initial_indexing...")
            partial_str = str(partial_initial_indexing[1]).replace(".", "_")

    print("Test dataset length:", len(dataset['test']))
    
    query_params = settings.query_params[index_type]
    
    if index_type == "IVFFLAT":
        new_query_params = []
        for query_param in query_params:
            # will be scaled ?
            if query_param[1]:
                # calculates n_probe number for IVFFLAT index by multiplying ideal_nprobe with scale_factor
                new_query_params.append(int((indexing_stat["indexing_params"]["list_number"]**0.5) * query_param[0]))
            else:
                new_query_params.append(query_param[0])
        query_params = new_query_params
        
    recall_stats = []
    for query_param in query_params:
                
        start_time = time.time()
        run_distance_df = pgvector_service.query(dataset_name, dataset['test'], vector_type, index_type=index_type, query_param=query_param)
        end_time = time.time()
        duration = end_time - start_time
        qps = len(dataset['test']) / duration
        print("query duration:", duration)
        
        run_distance_df['Distance'] = run_distance_df['Score'].apply(lambda x: 1 - x)
        run_distances = np.array(run_distance_df.groupby('SourceId')['Distance'].apply(lambda x: x.values))
            
        recall_result = metrics.recall(dataset_name, dataset_path, run_distances, count=100, epsilon=1e-3)
        recall_stat = {"query_param": query_param, "recall_mean": recall_result[0], "recall_std": recall_result[1], "duration": duration, "qps": qps, "index_type": index_type, "vector_type": vector_type, "dataset_name": dataset_name}
        recall_stats.append(recall_stat)
        
    return indexing_stat, recall_stats

if __name__ == '__main__':

    indexing = True
    index_types = settings.index_types
    
    if indexing == False:
        index_types = index_types[-1:]
    
    for index_type in index_types: 
        
        for indexing_param in settings.indexing_parameters[index_type]:
            
            for dataset_name in settings.dataset_names:
                print("dataset_name:", dataset_name)
                index_stats = []
                for partial_initial_indexing in settings.partial_initial_indexing_sets:
                    single_stat = {}
                    vector_type = settings.dataset_vector_type_map[dataset_name]
                    dataset_path = os.path.join(settings.main_dataset_path, settings.dataset_paths[dataset_name])
                    
                    indexing_stat, recall_stats = process_experiment(dataset_name=dataset_name,
                                    create_index=indexing,
                                    dataset_path=dataset_path,
                                    vector_type = vector_type,
                                    index_type = index_type,
                                    partial_initial_indexing = partial_initial_indexing,
                                    indexing_param = indexing_param)

                    single_stat["indexing_stat"] = indexing_stat
                    single_stat["recall_stats"] = recall_stats
                    index_stats.append(single_stat)
            
                indexing_param_str = ""
                if indexing_param is not None:
                    indexing_param_str = str(indexing_param[0]) + "_" + str(indexing_param[1])
                stat_file_name = os.path.join(settings.OUTPUT_PATH, index_type + '_' + dataset_name + '_' + indexing_param_str + '_stat.json')
                
                with open(stat_file_name, 'w') as fp:
                    stat = {'stats': index_stats, 'index_type': index_type, 'dataset_name': dataset_name}
                    json.dump(index_stats, fp, indent=4, sort_keys=True)