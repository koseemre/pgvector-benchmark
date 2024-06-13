import os
from urllib.request import urlretrieve

file_dir = os.path.dirname(os.path.abspath(__file__))

source_urls = [
    "http://ann-benchmarks.com/gist-960-euclidean.hdf5",
    "http://ann-benchmarks.com/glove-200-angular.hdf5",
    "http://ann-benchmarks.com/mnist-784-euclidean.hdf5",
    "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
    "http://ann-benchmarks.com/deep-image-96-angular.hdf5"    
]

destination_paths = [
    os.path.join (file_dir, "datasets/gist/gist-960-euclidean.hdf5"),
    os.path.join (file_dir, "datasets/glove/glove-200-angular.hdf5"),
    os.path.join (file_dir, "datasets/mnist/mnist-784-euclidean.hdf5"),
    os.path.join (file_dir, "datasets/sift/sift-128-euclidean.hdf5"),
    os.path.join (file_dir, "datasets/Deep1B/deep-image-96-angular.hdf5")
]

for source_url, destination_path in zip(source_urls, destination_paths):
    print("source_url:", source_url)
    if not os.path.isdir(destination_path):
        # create directory if not exists
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        
    urlretrieve(source_url, destination_path)