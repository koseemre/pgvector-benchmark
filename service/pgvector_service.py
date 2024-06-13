
from entity.mnist import engine as mnist_engine, MNIST
from entity.glove import engine as glove_engine, GLOVE
from entity.sift import engine as sift_engine, SIFT
from entity.deep1b import engine as deep1b_engine, DEEP1B
from entity.gist import engine as gist_engine, GIST

from sqlalchemy.sql import text as sa_text
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import Session
import pandas as pd
import psycopg2
import typing

from queries import postgre_query
from util.data_util import data_util
import settings

class PGVectorService:

    def __init__(self) -> None:
        self.table_connection_map: typing.Dict[str, tuple[Engine,] ] = { 
            "MNIST": (mnist_engine, MNIST),
            "DEEP1B": (deep1b_engine, DEEP1B),
            "GIST": (gist_engine, GIST),
            "GLOVE": (glove_engine, GLOVE),
            "SIFT": (sift_engine, SIFT)
            }
        
        # tuple like (table_name, embedding_index, embedding_column_name)
        self.table_name_map = settings.table_name_map
        
        self.indexing_config_map = {
            # memory, query
            "IVFFLAT": (3, postgre_query.IVFFLAT_INDEX_QUERY),
            "HNSW": (10, postgre_query.HNSW_INDEX_QUERY)
        }
        
        self.similarity_query = {
            
            'vector_l2_ops':  """SELECT id, 
                (embedding <-> '{embedding}' ) as simScore 
                from {table_name}
                order by (embedding <-> '{embedding}')
                LIMIT({limit});""",
            'vector_cosine_ops': """SELECT id, 
                1-(embedding <=> '{embedding}' ) as simScore 
                from {table_name}
                order by (embedding <=> '{embedding}')
                LIMIT({limit});"""
        }
    
    @staticmethod
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]
    
    def get_embeddings(self, dataset_name, size):
        
        session = Session(self.table_connection_map[dataset_name][0])
        try:
            search_list = session.query(self.table_connection_map[dataset_name][1]).filter().limit(size).all()
            return search_list      
        except Exception as e:
            print(e)
            raise e
        finally:
            session.close()        
        

    def query(self, dataset_name, input_vector_list:list, vector_type:str, limit=100, index_type = "IVFFLAT", query_param = None):
        
        res_list = []
        table_name = self.table_name_map[dataset_name][0]
        conn = psycopg2.connect("dbname={0} user={1} host={2} password={3}".format(settings.PGVECTOR_DATABASE, settings.PGVECTOR_USER, settings.PGVECTOR_HOST, settings.PGVECTOR_PASSWORD))
        cur = conn.cursor()
        try:
            cur.execute("BEGIN;")
            if index_type == 'HNSW':
                cur.execute("SET hnsw.ef_search = {};".format(query_param))
            elif index_type == 'IVFFLAT':
                cur.execute("SET ivfflat.probes = {};".format(query_param))
            cur.execute("COMMIT;")
                   
            for idx, input_data in enumerate(input_vector_list):
                s = self.similarity_query[vector_type].format( embedding = list(input_data),
                                                    table_name=table_name,
                                                    limit = limit)
                cur.execute("BEGIN;")
                cur.execute(s)
                neighs = cur.fetchall()
                cur.execute("COMMIT;")
                # source_test_index, similar_id, similarity_score
                sim_res = [(idx, r[0], round(r[1], 4)) for r in neighs]
                res_list += sim_res
                
            sim_res_df = pd.DataFrame(res_list, columns=["SourceId", "SimilarId", "Score"])
            return sim_res_df
        except Exception as e:
            print(e)
            raise e
        finally:
            try:
                conn.close()
                cur.close()
            except Exception as e:
                pass
            
    def truncate_and_drop_indexes(self, dataset_name):
        
        with self.table_connection_map[dataset_name][0].connect() as conn:
            print("Truncating ", dataset_name)
            conn.execute(sa_text('''TRUNCATE TABLE {}'''.format(self.table_name_map[dataset_name][0])).execution_options(autocommit=True))
            print("Dropping indexes of ", dataset_name)
            conn.execute(sa_text('''DROP INDEX IF EXISTS {}'''.format(self.table_name_map[dataset_name][1])).execution_options(autocommit=True))
            conn.commit()
            
        #self.table_connection_map[dataset_name][0].execute(sa_text('''TRUNCATE TABLE {}'''.format(self.table_name_map[dataset_name][1])).execution_options(autocommit=True))
        #self.table_connection_map[dataset_name][0].execute(sa_text('''DROP INDEX {}'''.format(self.table_name_map[dataset_name][1])).execution_options(autocommit=True))

    def create_index(self, dataset_name, dataset_length, vector_type = "vector_cosine_ops", index_type = "IVFFLAT", partial_initial_indexing = (False, 0.5), indexing_param = None):
        
        indexing_params = {}
        initial_insert_data_size = None
        indexing_params["dataset_length"] = dataset_length
        indexing_params["partial_indexing"] = False
        indexing_params["indexing_param"] = indexing_param
        
        if partial_initial_indexing[0]:
            print("Drop indexes and truncate table")
            self.truncate_and_drop_indexes(dataset_name)
            print("Inserting partial data")
            initial_insert_data_size = int(dataset_length * partial_initial_indexing[1])
            indexing_params["partial_indexing"] = True
            indexing_params["initial_insert_data_size"] = initial_insert_data_size
            data_util.write_to_db(dataset_name, settings.dataset_paths[dataset_name], convert=True, is_truncate=True, insert_data_interval=(0, initial_insert_data_size))
            
        if index_type in ['IVFFLAT', "HNSW"]:
            
            print(index_type, " indexing starts..")
            index_query:str = self.indexing_config_map[index_type][1]
            
            if index_type == 'IVFFLAT':
                list_number = int(dataset_length ** 0.5)
                if partial_initial_indexing[0]:
                    list_number = int(initial_insert_data_size ** 0.5)
                index_query = index_query.format(index_name = self.table_name_map[dataset_name][1],
                                                table_name = self.table_name_map[dataset_name][0],
                                                embedding_column_name = self.table_name_map[dataset_name][2],
                                                vector_type = vector_type,
                                                list_number = list_number
                                                )
                indexing_params["list_number"] = list_number
            elif index_type == 'HNSW':
                m = indexing_param[0]
                ef_construction = indexing_param[1]
                index_query = index_query.format(index_name = self.table_name_map[dataset_name][1],
                                                table_name = self.table_name_map[dataset_name][0],
                                                embedding_column_name = self.table_name_map[dataset_name][2],
                                                vector_type = vector_type,
                                                m = m,
                                                ef_construction = ef_construction
                                                )
                indexing_params["m"] = m
                indexing_params["ef_construction"] = ef_construction
            else:
                raise Exception("Unknown index type", index_type)
              
            indexing_params["index_query"] = index_query
            indexing_params["index_type"] = index_type
            print("index_query:", index_query)
            
            with self.table_connection_map[dataset_name][0].connect() as conn:
                print("setting maintanence memory ")
                conn.execute(sa_text('''SET maintenance_work_mem = '{} GB';'''.format(self.indexing_config_map[index_type][0]))) 
                print("droping index for ", dataset_name)
                conn.execute(sa_text('''DROP INDEX IF EXISTS {}'''.format(self.table_name_map[dataset_name][1])))
                print("creating index for ", dataset_name)
                conn.execute(sa_text(index_query) )
                conn.commit()
                print("index has created for ", dataset_name)
                
            # append index size information
            table_size_information = data_util.get_table_size_information(dataset_name)
            index_size = table_size_information[0][2]
            indexing_params["initial_index_size"] = index_size
            
            # add remaining data    
            if partial_initial_indexing[0]:
                print("Insert remaining data")
                if initial_insert_data_size is None:
                    raise Exception("initial_insert_data_size is None")
                
                remaining_insert_data_size = dataset_length - initial_insert_data_size
                data_util.write_to_db(dataset_name, settings.dataset_paths[dataset_name], convert=True, is_truncate=False, insert_data_interval=(initial_insert_data_size, dataset_length))
                # append index size information at last
                table_size_information = data_util.get_table_size_information(dataset_name)
                index_size = table_size_information[0][2]
                indexing_params["final_index_size"] = index_size
                indexing_params["remaining_insert_data_size"] = remaining_insert_data_size
                
                                        
            return indexing_params
        else:
            raise Exception("Unknown index type", index_type)
        
pgvector_service = PGVectorService()