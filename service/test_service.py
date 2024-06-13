from util.data_util import data_util
import numpy as np

class Metrics:
    
    def __init__(self) -> None:
        self.distance_functions = { "EUCLEDIAN": Metrics.eucledian_distance,
                                    "ANGULAR": Metrics.angular_distance}
    
    # knn_threshold, epsilon_threshold and get_recall_values are from ann-benchmarks/ann_benchmarks/plotting/metrics.py
    @staticmethod
    def knn_threshold(data, count, epsilon):
        return data[count - 1] + epsilon

    @staticmethod
    def epsilon_threshold(data, count, epsilon):
        return data[count - 1] * (1 + epsilon)
        
    def get_recall_values(self, dataset_distances, run_distances, count=100, threshold=knn_threshold, epsilon=1e-3):
        recalls = np.zeros(len(run_distances))
        for i in range(len(run_distances)):
            t = threshold(dataset_distances[i], count, epsilon)
            actual = 0
            for d in run_distances[i][:count]:
                if d <= t:
                    actual += 1
            recalls[i] = actual
        return (np.mean(recalls) / float(count), np.std(recalls) / float(count), recalls)
        
    # recall_result = test_service.recall(dataset_name, datasets[dataset_name][0]['test'])
    def recall(self, dataset_name:str, dataset_path:str, run_distances, count:int=100, epsilon:float=1e-3):

        print(dataset_name, " starting to testing..")
        print("dataset_path:", dataset_path)
        dataset, dimension = data_util.get_dataset(dataset_path)
        mean, std, recalls = self.get_recall_values(dataset['distances'], run_distances, count, Metrics.knn_threshold, epsilon)
        print(dataset_name, " recall mean: ", mean, " recall std: ", std)
        return mean, std, recalls
            
    def get_distance(self, metric:str, a:np.array, b:np.array):
        return self.distance_functions[metric](a,b)
    
    @staticmethod
    def norm(a):
        return np.sqrt(np.sum(a**2))
        
    @staticmethod
    def angular_distance(a, b):
        distance = 1 - np.dot(a, b) / (Metrics.norm(a) * Metrics.norm(b))    
        return distance
    
    def eucledian_distance(a,b):
        sum_square = np.sum(np.square(a - b))
        return np.sqrt(sum_square)
        
metrics = Metrics()