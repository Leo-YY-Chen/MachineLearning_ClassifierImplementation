import numpy as np
import sys
sys.path.append('..')
import model.utils as utils
from reference.classifier import Classifier

class DTClassifier(Classifier):
    def __init__(self, max_leaf_impurity=0.2, max_samples_leaf = 10, max_tree_depth=10):
        super().__init__()
        self.hyper_parameters = {'max_leaf_impurity':max_leaf_impurity, # the maximum of entropy in each leaves
                                 'max_samples_leaf':max_samples_leaf,   # the maximum required samples in each leaves
                                 'max_tree_depth':max_tree_depth}       # the maximum of depth of the tree
        self.name = 'DT'
        self.root = None

    def update_weights(self, dataset):
        self.weights = {"features":dataset.features, "labels":dataset.labels}
        self.root = utils.DecisionTreeNode(**self.hyper_parameters)
        self.root.build_the_tree(dataset)
        return None
    
    def predict_labels(self, dataset):
        labels = self.root.predict_labels(dataset)
        return labels

    def test(self, data): 
        if self.weights == None:
            raise NameError('Train Decision Tree before testing!')
        self.set_accuracy(data)
        return None

if __name__ == '__main__': 
    X = np.random.randn(30,8)
    Y = np.random.randn(30,)
    Y[Y>=0] = 1
    Y[Y<0] = -1
    
    dataset = utils.Dataset(X, Y)
    train_data, test_data = dataset.split_in_ratio()

    dt = DTClassifier()
    dt.k_fold_cross_validation(dataset)