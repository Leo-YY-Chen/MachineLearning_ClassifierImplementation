import numpy as np
import sys
sys.path.append('..')
import utils
import classifier as clf
'''
class Treenode():
    def __init__(self, X, Y, max_leaf_entrpoy, max_leaf_size, max_tree_depth, depth=1):
        # Samples in the node
        self.X = X
        self.Y = Y
        self.dominant_label = 0
        # Attributes for create the leaves
        self.max_leaf_entrpoy = max_leaf_entrpoy
        self.feature_idx = 0
        self.feature_median = 0
        self.left = None
        self.right = None
        # Pre-pruning attributes 
        self.depth = depth
        self.max_leaf_size = max_leaf_size
        self.max_tree_depth= max_tree_depth


        self.label = 0
        self.entropy = self.calculate_entropy(self.Y)
        self.rules_of_building_children = {'feature_index':None, 'feature_median':None}
        self.left_child = None  # bulid with datatset whose feature value > feature_median
        self.right_child = None # bulid with datatset whose feature value <= feature_median
        

    def calculate_entropy(self, labels):
        label_list = list(set(labels))
        label_prob_list = [np.sum(labels==lb)/labels.shape[0] for lb in label_list]
        label_entropy_list = [(0 if prob==0 else prob*np.log(prob)) for prob in label_prob_list]
        entropy = -np.sum(label_entropy_list)
        return entropy

    def grow_tree(self):
        # For each node, check the (self.dominant_label, self.entropy) by the train data (X, Y)
        self.dominant_label = mode(self.Y)
        if self.entropy <= self.max_leaf_entrpoy:
            return None 

        # Pre-pruning the tree by (self.max_leaf_size, self.max_tree_depth)
        if self.Y.shape[0] < self.max_leaf_size:
            return None
        if self.depth >= self.max_tree_depth:
            return None

        # Calulate information gain (IG) of each split pairs determined by corresponding feature median 
        feature_median_list = np.median(self.X, axis=0)
        children_entropy_list = []
        for i,median in enumerate(feature_median_list):
            left_Y = self.Y[self.X[:,i] > median]
            right_Y = self.Y[self.X[:,i] <= median]
            children_entropy = self.calculate_entropy(left_Y) + self.calculate_entropy(right_Y)
            children_entropy_list.append(children_entropy)
        IG_list = self.entropy - children_entropy_list

        # Determine the dominant feature and median (self.feature_idx, self.feature_median) of the node
        # as the attribiute to create the leaves
        IG_list[IG_list==0] = -100 # IG = 0 means the feature owing only two kind of values and it should be ignore  
        self.feature_idx = np.argmax(IG_list)
        self.feature_median = feature_median_list[np.argmax(IG_list)]

        # Create the leaves (self.left, self.right) by 
        left_samples_id_list = (self.X[:, self.feature_idx] > self.feature_median)
        right_samples_id_list = (self.X[:, self.feature_idx] <= self.feature_median)
        self.left = Treenode(self.X[left_samples_id_list, :], self.Y[left_samples_id_list],self.max_leaf_entrpoy, self.max_leaf_size, self.max_tree_depth, self.depth+1)
        self.right= Treenode(self.X[right_samples_id_list, :], self.Y[right_samples_id_list],self.max_leaf_entrpoy, self.max_leaf_size, self.max_tree_depth, self.depth+1)

        # Recursively grow the tree
        self.left.grow_tree()
        self.right.grow_tree()

    def traverse_tree(self, X_test, Y_test):
        # Stop criteria
        if self.left == None:
            return None

        # Classify the test data (X_test, Y_test) by current tree attributes (self.feature_idx, self.feature_median)
        left_samples_id_list = (X_test[:, self.feature_idx] > self.feature_median)
        right_samples_id_list = (X_test[:, self.feature_idx] <= self.feature_median)
        self.left.X, self.left.Y = X_test[left_samples_id_list, :], Y_test[left_samples_id_list]
        self.right.X, self.right.Y = X_test[right_samples_id_list, :], Y_test[right_samples_id_list]

        # Recursively traverse the tree
        self.left.traverse_tree(self.left.X, self.left.Y)
        self.right.traverse_tree(self.right.X, self.right.Y)'''

class DTClassifier(clf.Classifier):
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