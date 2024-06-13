from entity.mnist import engine as mnist_engine, MNIST
from entity.glove import engine as glove_engine, GLOVE
from entity.sift import engine as sift_engine, SIFT
from entity.deep1b import engine as deep1b_engine, DEEP1B
from entity.gist import engine as gist_engine, GIST
from queries import postgre_query as queries
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import Session
from sqlalchemy import insert
from sqlalchemy.sql import text as sa_text
import numpy as np
import settings
import typing
import h5py
import os

class DataUtil:
    
    def __init__(self) -> None:
        self.table_connection_map: typing.Dict[str, tuple[Engine,] ] = { 
            "MNIST": (mnist_engine, MNIST),
            "DEEP1B": (deep1b_engine, DEEP1B),
            "GIST": (gist_engine, GIST),
            "GLOVE": (glove_engine, GLOVE),
            "SIFT": (sift_engine, SIFT)
            }
        self.table_name_map = settings.table_name_map
        
    def get_dataset(self, hdf5_filename):
        
        hdf5_file = h5py.File(hdf5_filename, "r")
        # here for backward compatibility, to ensure old datasets can still be used with newer versions
        # cast to integer because the json parser (later on) cannot interpret numpy integers
        dimension = int(hdf5_file.attrs["dimension"]) if "dimension" in hdf5_file.attrs else len(hdf5_file["train"][0])
        return hdf5_file, dimension
    
    @staticmethod
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]
            
    def truncate_and_drop_indexes(self, dataset_name):
        
        with self.table_connection_map[dataset_name][0].connect() as conn:
            print("Truncating ", dataset_name)
            conn.execute(sa_text('''TRUNCATE TABLE {}'''.format(self.table_name_map[dataset_name][0])).execution_options(autocommit=True))
            print("Dropping indexes of ", dataset_name)
            conn.execute(sa_text('''DROP INDEX IF EXISTS {}'''.format(self.table_name_map[dataset_name][1])).execution_options(autocommit=True))
            conn.commit()
                            
    def insert_to_table(self, dataset_name, embedding_list:list, convert = False, is_truncate = False, start_index = 0):
        
        print("len embedding_list:", len(embedding_list))
        
        if convert:
            if dataset_name in ["MNIST", "DEEP1B", "GIST", "GLOVE", "SIFT"]:
                embedding_list = [{ "id": start_index + idx,  "embedding": embedding } for idx, embedding in enumerate(embedding_list)]
        session = Session(self.table_connection_map[dataset_name][0])
        
        try:
            
            # truncate table
            if is_truncate:
                print("truncating table")
                self.truncate_and_drop_indexes(dataset_name)
                            
            batch_count = 0
            print("inserting started..")
            for batch_ in DataUtil.batch(embedding_list, 10000):
                res = session.execute(insert(self.table_connection_map[dataset_name][1]), batch_)
                session.commit()
                batch_count+= 1
            print("inserting done.")
        except Exception as e:
            print(e)
            raise e
        finally:
            session.close()
                
    def write_to_db(self, dataset_name, dataset_file_name, convert = True, is_truncate = False, insert_data_interval=None): # example: dataset_name: "MNIST", dataset_file_name: mnist/mnist-784-euclidean.hdf5
        
        data_path = os.path.join(settings.BASE_DATA_PATH, dataset_file_name)
        dataset, dimension = self.get_dataset(data_path)
        data_arr = np.array( dataset['train'])

        print("data_path:", data_path)
        print("dimension:", dimension)
        print("data len:", len(data_arr))

        if insert_data_interval is not None:
            print("insert_data_interval:", insert_data_interval)
            data_arr = data_arr[insert_data_interval[0]:insert_data_interval[1]]
            print("insert_data_size:", insert_data_interval[1] - insert_data_interval[0])

        print("Inserting data to ", dataset_name, " table..")
        self.insert_to_table(dataset_name, data_arr.tolist(), convert = convert, is_truncate = is_truncate, start_index = insert_data_interval[0] if insert_data_interval is not None else 0)
        print("Data inserted to ", dataset_name, " table")
        
    def get_table_size_information(self, dataset_name):
        with self.table_connection_map[dataset_name][0].connect() as conn:
            table_size_query = sa_text(queries.TABLE_SIZE_QUERY.format(table_name = self.table_name_map[dataset_name][0]))
            table_information = conn.execute(table_size_query)
            table_information = table_information.fetchall()
            print("Table size information for ", dataset_name, ":", table_information)
            return table_information
        
data_util = DataUtil()