import numpy as np
import sys
sys.path.append('..')
from statistics import mode
from calculator import Performance_Calculator as calculator
from data_processor import Data_Processor as data_processor



class Decision_Tree_Hyperparameters:
    def __init__(self, max_leaf_impurity = None, max_samples_leaf = None, max_tree_depth = None ,ccp_alpha = None):
        self.max_leaf_impurity = max_leaf_impurity  # The maximum of entropy in each leaf node
        self.max_samples_leaf = max_samples_leaf    # The maximum required samples in each leaf node
        self.max_tree_depth = max_tree_depth        # The maximum of depth of the tree

        # To do cost complexity pruning,
        # ref: https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning
        self.ccp_alpha = ccp_alpha  # Pruning stops when the pruned tree’s minimal alpha_eff is greater than the ccp_alpha parameter.







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

    def build_tree(self, features, labels):  ##### REFACTOR: (1)ugly get_dataset approach (2) reduce is_building_finished() arg
        self.set_attributes(features, labels)
        if self.is_building_finished(labels):
            return
        self.set_child_nodes()      
        self.left_child.build_tree(*data_processor().get_data_bigger_than_median(features, labels, self.major_feature['index']))
        self.right_child.build_tree(*data_processor().get_data_not_bigger_than_median(features, labels, self.major_feature['index']))

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

    def set_major_feature(self, features, labels):##### REFACTOR: try to remove dependency on calculator()
        IGs = calculator().calculate_information_gains(features, labels, self.entropy)  
        self.major_feature['index'] = np.argmax(IGs)
        self.major_feature['median'] = np.median(features, axis=0)[np.argmax(IGs)]
        
    def set_major_label(self, labels):
        self.major_label = mode(labels)
        
    def set_entropy(self, labels):##### REFACTOR: try to remove dependency on calculator()
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
        if feature[self.major_feature['index']] > self.major_feature['median']:
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

    def build_tree(self, features, labels):
        super().build_tree(features, labels)
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
    # TEST TreeNode
    #######################
    '''def test_get_leaf_nodes_number():
        hyperparameters = Decision_Tree_Hyperparameters(0.2, 2, 2)
        dt = DecisionTreeNode(hyperparameters)
        features = np.array([[0,0],[1,1],[2,2],[3,3],[4,4],[5,5]])
        labels = np.array([1, -1, 1, -1, -1, -1])

        dt.build_tree(features, labels)
        if dt.get_leaf_nodes_number() == 2:
            print("passing")
        else:
            print("fail")
    test_get_leaf_nodes_number()'''





    #######################
    # TEST DecisionTreeNode
    #######################
    def test_build_tree():
        hyperparameters = Decision_Tree_Hyperparameters(0.2, 2, 2)
        dt = DecisionTreeNode(hyperparameters)
        features = np.array([[0,0],[1,1],[2,2],[3,3],[4,4],[5,5]])
        labels = np.array([1, -1, 1, -1, -1, -1])

        dt.build_tree(features, labels)
    test_build_tree()

    




    #######################
    # TEST PrunedDecisionTreeNode
    #######################
    '''def test_build_tree():
        hyperparameters = Decision_Tree_Hyperparameters(0.2, 2, 3, 0.1)
        pdt = PrunedDecisionTreeNode(hyperparameters)
        features = np.array([[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[0,0],[1,1],[2,2],[3,3],[4,4],[5,5]])
        labels = np.array([1, -1, 1, -1, -1, -1,1, -1, 1, -1, -1, -1])

        pdt.build_tree(features, labels)
    test_build_tree()'''

