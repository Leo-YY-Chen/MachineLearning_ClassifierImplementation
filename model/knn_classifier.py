import sys
sys.path.append('..')
import numpy as np
from classifier import Classifier, Calculator_Interface, Presenter_Interface






class KNNClassifier(Classifier):
    def __init__(self, calculator:Calculator_Interface, presenter:Presenter_Interface, n_neighbor = 3, p_norm = 2):
        super().__init__(calculator, presenter)
        self.information.type = 'KNN'
        self.attributes["hyper_parameters"] = {'p_norm': p_norm, 'n_neighbor': n_neighbor}
    
    def update_parameters(self, features, labels):
        self.attributes['parameters'] = {"train_features":features, "train_labels":labels}
    
    def get_predictions(self, features):
        return np.array([self.predict_a_label(features[instance,:]) for instance in range(features.shape[0])])
    













    def predict_a_label(self, a_feature):
        for i in range(len(self.attributes['parameters']['train_labels'])):      
            if (a_feature == self.attributes['parameters']['train_features'][i, :]).all():
                return self.attributes['parameters']['train_labels'][i]
        
        distances = self.calcualte_distances(a_feature)
        indices = self.get_neighbors_indices(distances)
        label = self.get_major_label_of_neighbors(indices)
        return label
    

    
    def calcualte_distances(self, the_feature):
        features = self.attributes['parameters']['train_features']
        distance = np.linalg.norm(features - the_feature,
                                  ord = self.attributes['hyper_parameters']['p_norm'], 
                                  axis = 1)     
        return distance
    
    def get_neighbors_indices(self, distances_to_neighbors):
        indices = np.argsort(distances_to_neighbors)[:self.attributes['hyper_parameters']['n_neighbor']]
        return indices

    def get_major_label_of_neighbors(self, indices_of_neighbor):
        major_label = 1 if np.sum(self.attributes['parameters']['train_labels'][indices_of_neighbor]) >= 0 else -1
        return major_label