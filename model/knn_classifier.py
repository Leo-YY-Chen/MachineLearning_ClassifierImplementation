import sys
sys.path.append('..')
import numpy as np
from classifier import Classifier






class KNNClassifier(Classifier):
    def __init__(self, n_neighbor = 3, p_norm = 2):
        super().__init__()
        self.information.type = 'KNN'
        self.attributes["hyper_parameters"] = {'p_norm': p_norm, 'n_neighbor': n_neighbor}

    '''def train(self, data):
        self.update_weights(data)
        self.accuracy = 1 # This is equvelent to self.calculate_and_set_accuracy(data)
        return None

    def test(self, data): 
        if self.weights == None:
            raise NameError('Train KNN before testing!')
        self.set_accuracy(data)
        return None

    def update_parameters(self, data):
        self.weights = {"features":data.features, "labels":data.labels}
        return None'''
    
    def update_parameters(self, features, labels):
        self.attributes['parameters'] = {"train_features":features, "train_labels":labels}


    '''def predict_labels(self, data):
        labels = [self.predict_a_label(data.features[id,:]) for id in range(data.features.shape[0])]
        return labels'''
    
    def get_predictions(self, features):
        return [self.predict_a_label(features[instance,:]) for instance in range(features.shape[0])]
    





    def predict_a_label(self, a_feature):
        distances = self.calcualte_distances(a_feature)
        indices = self.get_neighbors_indices(distances)
        label = self.get_major_label_of_neighbors(indices)
        return label
    
    def get_neighbors_indices(self, distances_to_neighbors):
        indices = np.argsort(distances_to_neighbors)[:self.attributes['hyper_parameters']['n_neighbor']]
        return indices

    def get_major_label_of_neighbors(self, indices_of_neighbor):
        major_label = 1 if np.sum(self.attributes['parameters']['train_labels'][indices_of_neighbor]) >= 0 else -1
        return major_label
    
    def calcualte_distances(self, the_feature):
        features = self.attributes['parameters']['train_features']
        distance = np.linalg.norm(features - the_feature,
                                  ord = self.attributes['hyper_parameters']['p_norm'], 
                                  axis = 1)     
        return distance
    

    

if __name__ == '__main__':
    '''X = np.random.randn(10000,8)
    Y = np.random.randn(10000,)
    Y[Y>=0] = 1
    Y[Y<0] = -1

    dataset = utils.Dataset(X, Y)
    train_data, test_data = dataset.split_in_ratio()

    knn = KNNClassifier()
    knn.k_fold_cross_validation(dataset)'''

    