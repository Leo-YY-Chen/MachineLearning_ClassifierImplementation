import sys
sys.path.append('..')
import model.utils as utils
from classifier import Classifier, Calculator_Interface, Presenter_Interface

class DTClassifier(Classifier):
    def __init__(self, calculator:Calculator_Interface, presenter:Presenter_Interface, max_leaf_impurity=0.2, max_samples_leaf = 10, max_tree_depth=10):
        super().__init__(calculator, presenter)
        self.information.type = 'DT'
        self.attributes['hyper_parameters'] = {'max_leaf_impurity':max_leaf_impurity, # the maximum of entropy in each leaves
                                                'max_samples_leaf':max_samples_leaf,   # the maximum required samples in each leaves
                                                'max_tree_depth':max_tree_depth}       # the maximum of depth of the tree
        self.root = None




    def test(self, features, labels): 
        if self.root is None:
            raise NameError('Train Decision Tree before testing.')
        super().test(features, labels)

    def valid(self, features, labels): 
        if self.root is None:
            raise NameError('Train Decision Tree before testing.')
        super().valid(features, labels)





    def update_parameters(self, features, labels):
        self.attributes['parameters'] = {"features":features, "labels":labels}
        self.root = utils.DecisionTreeNode(utils.Decision_Tree_Hyperparameters(*tuple(self.attributes['hyper_parameters'].values())))
        self.root.build_tree(features, labels)
    
    def get_predictions(self, features, labels):
        return self.root.get_predictions(features, labels)
    
    def get_performance(self, features, labels):
        self.set_metrics(labels, self.get_predictions(features, labels))
        self.show_performance()
        self.save_performance()
    
    




    def load_classifier(self, pkl_file_name):
        super().load_classifier(pkl_file_name)
        self.root = utils.DecisionTreeNode(utils.Decision_Tree_Hyperparameters(*tuple(self.attributes['hyper_parameters'].values())))
        self.root.build_tree(*tuple(self.attributes['parameters'].values()))