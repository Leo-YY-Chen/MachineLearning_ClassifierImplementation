from calculator import Performance_Calculator
from presenter import Performance_Presenter, Iterative_Performance_Presenter


from model.linear_classifier import LinearClassifier
from model.knn_classifier import KNNClassifier
from model.naive_decision_tree_classifer import DTClassifier
from model.pruned_decision_tree_classifer import PDTClassifier








class Classifier_Factory:
    def __init__(self) -> None:
        pass

    def make_classifier(self, classifier_name, **hyperparameter):
        # Assume that   
        #               hyperparameter:    dictionary
        args = (self.__make_calculator(classifier_name), 
                self.__make_presenter(classifier_name), 
                *tuple(hyperparameter.values()))

        if classifier_name == 'LinearClassifier':
            return LinearClassifier(*args)
        elif classifier_name == 'KNNClassifier':
            return KNNClassifier(*args)
        elif classifier_name == 'DTClassifier':
            return DTClassifier(*args)
        elif classifier_name == 'PDTClassifier':
            return PDTClassifier(*args)
        else:
            raise NameError('Please input valid classifier name.')
        



    def __make_calculator(self, classifier_name):
        if classifier_name == 'LinearClassifier' or 'KNNClassifier' or 'DTClassifier' or 'PDTClassifier':
            return Performance_Calculator()
        else:
            raise NameError('Please input valid classifier name.')
        



    def __make_presenter(self, classifier_name):
        if classifier_name == 'LinearClassifier':
            return Iterative_Performance_Presenter()
        elif classifier_name == 'KNNClassifier' or 'DTClassifier' or 'PDTClassifier':
            return Performance_Presenter()
        else:
            raise NameError('Please input valid classifier name.')
        
    
            