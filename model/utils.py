#import csv
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import numpy as np
import sys
sys.path.append('..')
import data.config as cfg
from statistics import mode
from calculator import Performance_Calculator as calculator
from data_processor import Data_Processor as data_processor

class Dataset:
    def __init__(self, features = None, labels = None):
        self.features = features    #2D array
        self.labels = labels        #1D array
        self.k_fold_indices = None

    def load(self, datasest_path): ##### REFINE: this function depends on cfg files
        self.features = pd.read_csv(datasest_path, usecols = cfg.feature_names)
        self.labels = pd.read_csv(datasest_path, usecols = cfg.label_names)
        
    def preprocess(self):
        self.relabel_into_minus_or_plus_one()
        self.do_data_balancing()
        self.do_min_max_normalization()
        
    def relabel_into_minus_or_plus_one(self):
        self.labels[self.labels > 0] = 1
        self.labels[self.labels <= 0] = -1
        
    def do_data_balancing(self):
        self.do_oversampling()
        
    def do_oversampling(self):
        labels, numbers = self.get_labels_and_numbers_of_oversamples()
        self.clone_subset_randomly_by_labels_and_numbers(labels, numbers)
        
    def get_labels_and_numbers_of_oversamples(self):
        lebels, counts = np.unique(self.labels, return_counts=True)
        number_oversamples = np.max(counts) - counts
        return lebels, number_oversamples
    
    def clone_subset_randomly_by_labels_and_numbers(self, labels, numbers):
        for i, label in enumerate(labels):
            indices = self.get_indices_of_oversamples_with_label(numbers[i], label)
            self.clone_subset_by_indices(indices)
        
    def get_indices_of_oversamples_with_label(self, number_oversamples, label):
        indices = self.get_indices_with_label(label)
        indices_of_oversamples = np.random.choice(indices, number_oversamples)
        all_indices = np.array(range(len(self.features)))
        indices_of_oversamples = np.in1d(all_indices, indices_of_oversamples)
        return indices_of_oversamples
    
    def clone_subset_by_indices(self, indices):
        np.concatenate((self.features, self.features[indices]))
        np.concatenate((self.labels, self.labels[indices]))
        
    def get_indices_with_label(self, label):
        indices = np.where(self.labels == label)[0]
        return indices.tolist()
    
    def do_min_max_normalization(self):
        min = np.min(self.features, axis=0)
        max = np.max(self.features, axis=0)
        self.features = (self.features-min)/(max-min)
        
    def remove_unimportant_feature(self, feature_id_list): 
        np.delete(self.features, feature_id_list, axis=1)
        
    def split_in_ratio(self, split_ratio=[0.8, 0.2]): 
        train_set = Dataset()
        test_set = Dataset()
        train_set.features, test_set.features, train_set.labels, test_set.labels = train_test_split(self.features, self.labels, test_size = split_ratio[1], random_state=83)
        train_set.squeeze()
        test_set.squeeze()
        return train_set, test_set
    
    def get_subset(self, indices):
        dataset = Dataset()
        dataset.features = self.features[indices]
        dataset.labels = self.labels[indices]
        return dataset
   
    def squeeze(self):
        self.features = np.squeeze(self.features)
        self.labels = np.squeeze(self.labels)
        

    def set_k_fold_indices(self, n_splits):
        kf = KFold(n_splits=n_splits, random_state=83, shuffle=True)
        self.k_fold_indices = [(train_index, valid_index) for (train_index, valid_index) in kf.split(self.features)]
        

    def get_kth_fold_datasets(self, kth = 0):
        (train_index, valid_index) = self.k_fold_indices[kth]
        train_dataset = self.get_subset(train_index)
        valid_dataset = self.get_subset(valid_index)
        return train_dataset, valid_dataset

    def split_in_ratio_for_k_fold(self, n_splits):
        temp_dataset, test_dataset = self.split_in_ratio()
        temp_dataset.set_k_fold_indices(n_splits)
        return temp_dataset, test_dataset





class Decision_Tree_Hyperparameters:
    def __init__(self):
        self.max_leaf_impurity = None   # The maximum of entropy in each leaf node
        self.max_samples_leaf = None    # The maximum required samples in each leaf node
        self.max_tree_depth = None      # The maximum of depth of the tree

        # To do cost complexity pruning,
        # ref: https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning
        self.ccp_alpha = None           # Pruning stops when the pruned tree’s minimal alpha_eff is greater than the ccp_alpha parameter.




class TreeNode:
    def __init__(self, depth=1):
        self.depth = depth
        self.left_child = None
        self.right_child = None

    def build_tree(self):
        if self.is_building_finished():
            return
        self.set_child_nodes()
        self.left_child.build_tree()
        self.right_child.build_tree()
    
    def is_building_finished(self) -> bool:
        pass

    



 
    def set_child_nodes(self):
        self.left_child = TreeNode()
        self.right_child = TreeNode()

    def remove_child_nodes(self):
        self.left_child = None
        self.right_child = None

    def is_leaf(self):
        if self.left_child==None and self.right_child==None:
            return True
        else:
            return False

    def get_leaf_nodes_number(self):
        if self.is_leaf():
            return 1
        else:
            leaves_number = 0
            leaves_number += self.left_child.get_leaf_nodes_number()
            leaves_number += self.right_child.get_leaf_nodes_number()
            return leaves_number





    def show_the_branch(self): 
        pass

    
    


class DecisionTreeNode(TreeNode):
    def __init__(self, hyper_parameters:Decision_Tree_Hyperparameters, depth = 1):
        super().__init__(depth)
        self.hyper_parameters = hyper_parameters
        self.major_feature = {'index':None, 'median':None}
        self.major_label = np.nan
        self.entropy = 0

    def build_tree(self, dataset):  ##### REFACTOR: ugly get_dataset approach
        self.set_attributes(dataset.features, dataset.labels)
        if self.is_building_finished(dataset.labels):
            return
        self.set_child_nodes()      
        self.left_child.build_tree(data_processor().get_dataset_bigger_than_median(dataset, self.major_feature['index']))
        self.right_child.build_tree(data_processor().get_dataset_not_bigger_than_median(dataset, self.major_feature['index']))

    def is_building_finished(self, labels):
        if len(labels) == 1:
            return True 
        elif self.entropy <= self.hyper_parameters.max_leaf_impurity:
            return True 
        elif len(labels) <= self.hyper_parameters.max_samples_leaf:
            return True
        elif self.depth >= self.hyper_parameters.max_tree_depth:
            return True
        else:
            return False




    def set_child_nodes(self):
        self.left_child = DecisionTreeNode(self.hyper_parameters, self.depth+1)
        self.right_child = DecisionTreeNode(self.hyper_parameters, self.depth+1)
        
    def set_attributes(self, features, labels):
        self.set_major_feature(features, labels)
        self.set_major_label(labels)
        self.set_entropy(labels)

    def set_major_feature(self, features, labels):
        IGs = calculator().calculate_information_gains(features, labels, self.entropy)
        self.major_feature['index'] = np.argmax(IGs)
        self.major_feature['median'] = np.median(features, axis=0)[np.argmax(IGs)]
        
    def set_major_label(self, labels):
        self.major_label = mode(labels)
        
    def set_entropy(self, labels):
        self.entropy = calculator().calculate_entropy(labels)



    def get_predictions(self, features, labels):
        return [self.get_a_prediction(features[i,:]) for i in range(len(labels))]
            
    def get_a_prediction(self, feature):
        if self.is_leaf():
            return self.major_label
        else:
            if self.is_data_in_left_child(feature):
                return self.left_child.get_a_prediction(feature)
            else:
                return self.right_child.get_a_prediction(feature)

    def is_data_in_left_child(self, feature):   ##### REFACTOR: change name
        if feature[0, self.major_feature['index']] > self.major_feature['median']:
            return True
        else:
            return False
        





class PrunedDecisionTreeNode(DecisionTreeNode):
    def __init__(self, hyper_parameters:Decision_Tree_Hyperparameters, depth=1):
        super().__init__(hyper_parameters, depth)
        self.hyper_parameters = hyper_parameters
        self.alpha_eff = np.nan
        self.leaf_nodes_number = 0
        self.leaf_nodes_entropy = 0

    def build_tree(self, dataset):
        super().build_tree(dataset)
        self.get_prunning()

    def get_prunning(self):
        # By cost complexity pruning,
        # ref: https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning
        self.set_leaf_nodes_entropy()
        self.set_alpha_eff(self.set_leaf_nodes_number())
        while self.get_min_alpha_eff() < self.hyper_parameters.ccp_alpha:
            self.remove_weakest_branch(self.get_min_alpha_eff())
            self.set_alpha_eff(self.set_leaf_nodes_number())




    def set_child_nodes(self):
        self.left_child = PrunedDecisionTreeNode(self.hyper_parameters, self.depth+1)
        self.right_child = PrunedDecisionTreeNode(self.hyper_parameters, self.depth+1)
        
    def set_leaf_nodes_entropy(self):
        # TRAVERSE THE BRANCH and set it.
        if self.is_leaf():
            self.leaf_nodes_entropy = self.entropy
            return self.leaf_nodes_entropy
        else:
            self.leaf_nodes_entropy += self.left_child.set_leaf_nodes_entropy()
            self.leaf_nodes_entropy += self.right_child.set_leaf_nodes_entropy()
            return self.leaf_nodes_entropy
  
    def set_leaf_nodes_number(self):
        # TRAVERSE THE BRANCH and set it.
        if self.is_leaf():
            self.leaf_nodes_number = 1
            return self.leaf_nodes_number
        else:
            self.leaf_nodes_number = 0
            self.leaf_nodes_number += self.left_child.set_leaf_nodes_number()
            self.leaf_nodes_number += self.right_child.set_leaf_nodes_number()
            return self.leaf_nodes_number
    
    def set_alpha_eff(self, leaf_nodes_number_of_root):
        # TRAVERSE THE BRANCH and set it.
        if self.is_leaf():
            self.alpha_eff = np.nan
        else:
            self.alpha_eff = (self.entropy - self.leaf_nodes_entropy) / (leaf_nodes_number_of_root - 1)
            self.left_child.set_alpha_eff(leaf_nodes_number_of_root)
            self.right_child.set_alpha_eff(leaf_nodes_number_of_root)
        
    def get_min_alpha_eff(self):
        if self.is_leaf():
            min_alpha_eff = self.alpha_eff
        else:
            min_alpha_eff = np.nanmin([self.alpha_eff, self.left_child.get_min_alpha_eff(), self.right_child.get_min_alpha_eff()])
        return min_alpha_eff
    
    def remove_weakest_branch(self, min_alpha_eff):
        if self.is_leaf():
            return
        else:
            if self.alpha_eff == min_alpha_eff:
                self.remove_child_nodes()
            else:
                self.left_child.remove_weakest_branch(min_alpha_eff)
                self.right_child.remove_weakest_branch(min_alpha_eff)
            
        
    
    



if __name__ == '__main__':
    #######################
    # TEST decision tree
    #######################
    '''def test_build_tree():
        dt = DecisionTreeNode(**{'max_leaf_impurity':0.2, # the maximum of entropy in each leaves
                                 'max_samples_leaf':2,   # the maximum required samples in each leaves
                                 'max_tree_depth':2})
        dataset = Dataset()
        dataset.features = np.array([[0,0],[1,1],[2,2],[3,3],[4,4],[5,5]])
        dataset.labels = np.array([1, -1, 1, -1, -1, -1])

        dt.build_tree(dataset)
    test_build_tree()

    def test_get_leaf_nodes_number():
        dt = DecisionTreeNode(**{'max_leaf_impurity':0.2, # the maximum of entropy in each leaves
                                 'max_samples_leaf':2,   # the maximum required samples in each leaves
                                 'max_tree_depth':2})
        dataset = Dataset()
        dataset.features = np.array([[0,0],[1,1],[2,2],[3,3],[4,4],[5,5]])
        dataset.labels = np.array([1, -1, 1, -1, -1, -1])

        dt.build_tree(dataset)
        if dt.get_leaf_nodes_number() == 2:
            print("passing")
        else:
            print("fail")
    test_get_leaf_nodes_number()'''




    #######################
    # TEST pruned decision tree
    #######################
    '''def test_build_tree():
        pdt = PrunedDecisionTreeNode(**{'max_leaf_impurity':0.2, # the maximum of entropy in each leaves
                                 'max_samples_leaf':2,   # the maximum required samples in each leaves
                                 'max_tree_depth':3,
                                 'ccp_alpha':0.1})
        dataset = Dataset()
        dataset.features = np.array([[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[0,0],[1,1],[2,2],[3,3],[4,4],[5,5]])
        dataset.labels = np.array([1, -1, 1, -1, -1, -1,1, -1, 1, -1, -1, -1])

        pdt.build_tree(dataset)
    test_build_tree()'''

