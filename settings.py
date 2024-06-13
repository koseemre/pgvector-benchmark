import os
from dotenv import load_dotenv

load_dotenv('.env')

PGVECTOR_HOST = os.getenv("PGVECTOR_HOST")
PGVECTOR_USER = os.getenv("PGVECTOR_USER")
PGVECTOR_PASSWORD = os.getenv("PGVECTOR_PASSWORD")
PGVECTOR_DATABASE = os.getenv("PGVECTOR_DATABASE")

# current file path
BASE_PATH = os.path.dirname(os.path.realpath(__file__))
BASE_DATA_PATH = os.path.join(BASE_PATH, "datasets")
OUTPUT_PATH = os.path.join(BASE_PATH, "stats")

dataset_paths = {"MNIST": "mnist/mnist-784-euclidean.hdf5",
                    "DEEP1B": "Deep1B/deep-image-96-angular.hdf5",
                    "GIST": "gist/gist-960-euclidean.hdf5",
                    "GLOVE": "glove/glove-200-angular.hdf5",
                    "SIFT": "sift/sift-128-euclidean.hdf5" }

table_name_map = { 
            # table_name, embedding_index, embedding_column_name
            "MNIST": ("mnist", "mnist_embedding_index", "embedding"),
            "DEEP1B": ("deep1b", "deep1b_embedding_index", "embedding"),
            "GIST": ("gist", "gist_embedding_index", "embedding"),
            "GLOVE": ("glove", "glove_embedding_index", "embedding"),
            "SIFT": ("sift", "sift_embedding_index", "embedding")
            }


dataset_names = ["GLOVE", "MNIST", "GIST", "SIFT"]
partial_initial_indexing_sets = [(False, None), (True, 0.5), (True, 0.4), (True, 0.3), (True, 0.2), (True, 0.1)]

# m and ef_construction parameters for HNSW
indexing_parameters = { "HNSW": [(16, 64)],
                        "IVFFLAT": [None] }

hnsw_ef_search_params = [20, 40, 60, 80, 100, 200, 300, 400]
# second parameter is whether to use sqrt of the list number
n_probe_numbers = [(1, True), (1,False), (5, False), (10, False)]

query_params = { "IVFFLAT": n_probe_numbers,
                 "HNSW": hnsw_ef_search_params } 

index_types = ["IVFFLAT", "HNSW"]

print("index_types:", index_types, " dataset_names:", dataset_names)
    
dataset_paths = { "MNIST": "mnist/mnist-784-euclidean.hdf5",
                    "DEEP1B": "Deep1B/deep-image-96-angular.hdf5",
                    "GIST": "gist/gist-960-euclidean.hdf5",
                    "GLOVE": "glove/glove-200-angular.hdf5",
                    "SIFT": "sift/sift-128-euclidean.hdf5" }

main_dataset_path = os.path.join(BASE_PATH, "datasets")

# euclidean -> L2, angular -> cosine
dataset_vector_type_map = { "MNIST": "vector_l2_ops",
                            "DEEP1B": "vector_cosine_ops",
                            "GIST": "vector_l2_ops",
                            "GLOVE": "vector_cosine_ops",
                            "SIFT": "vector_l2_ops" }