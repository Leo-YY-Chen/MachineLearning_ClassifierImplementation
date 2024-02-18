import sys
sys.path.append('..')
import model.utils as utils
import model.naive_decision_tree_classifer as dt
from classifier import Calculator_Interface, Presenter_Interface

class PDTClassifier(dt.DTClassifier):
    def __init__(self, calculator:Calculator_Interface, presenter:Presenter_Interface, max_leaf_impurity=0.2, max_samples_leaf = 1, max_tree_depth=100, ccp_alpha = 0.1):
        super().__init__(calculator, presenter, max_leaf_impurity, max_samples_leaf, max_tree_depth)
        self.information.type = 'PDT'
        self.attributes['hyper_parameters'] = {'max_leaf_impurity':max_leaf_impurity, # the maximum of entropy in each leaves
                                 'max_samples_leaf':max_samples_leaf,   # the maximum required samples in each leaves
                                 'max_tree_depth':max_tree_depth,       # the maximum of depth of the tree
                                 'ccp_alpha':ccp_alpha}                 # arg for cost_complexity_prunning
        self.root = None






    def update_parameters(self, features, labels):
        self.attributes['parameters'] = {"features":features, "labels":labels}
        self.root = utils.PrunedDecisionTreeNode(utils.Decision_Tree_Hyperparameters(*tuple(self.attributes['hyper_parameters'].values())))
        self.root.build_tree(features, labels)




    def load_classifier(self, pkl_file_name):
        super().load_classifier(pkl_file_name)
        self.root = utils.PrunedDecisionTreeNode(utils.Decision_Tree_Hyperparameters(*tuple(self.attributes['hyper_parameters'].values())))
        self.root.build_tree(*tuple(self.attributes['parameters'].values()))