import numpy as np
import matplotlib.pyplot as plt
import os
import classifier
import model.utils



class Performance_Presenter(classifier.Presenter_Interface):
    def __init__(self):
        self.information = classifier.Information()
        self.metrics = classifier.Metrics()

    def show_performance(self, information:classifier.Information, metrics:classifier.Metrics):
        self.set_information_and_metrics(information, metrics)
        self.show_message()
    
    def save_performance(self, information:classifier.Information, metrics:classifier.Metrics):
        self.set_information_and_metrics(information, metrics)
        if not self.is_classifier_name_existing():
            self.save_classifier_name()
        self.save_message()





    def set_information_and_metrics(self, information:classifier.Information, metrics:classifier.Metrics):
        self.information = information
        self.metrics = metrics

    def show_message(self):
        print(self.get_message())

    def save_message(self):
        with open(os.path.join(os.getcwd(), 'results\log', self.information.type) +'.txt', 'a') as f:
                f.write(f"{self.get_message()} \n")

    def is_classifier_name_existing(self):
        with open(os.path.join(os.getcwd(), 'results\log', self.information.type) +'.txt', 'r') as fp:
            for line_number, line in enumerate(fp):
                if self.information.timestamp in line:
                    return True
        return False
        
    def save_classifier_name(self):
        with open(os.path.join(os.getcwd(), 'results\log', self.information.type) +'.txt', 'a') as f:
            f.write(f"------------------------------------------------------------------------------- \n")
            f.write(f"{self.information.timestamp} \n")



    def get_message(self):
        return self.get_header_message() + self.get_metrics_message()

    def get_header_message(self):
        fold_info = "" if self.information.fold_quantity == None else f" Fold {self.information.fold_number + 1}/{self.information.fold_quantity}" 
        epoch_info = "" if self.information.epoch_quantity == None else f" Epoch {self.information.epoch_number + 1}/{self.information.epoch_quantity}"
        return f"{self.information.state + fold_info + epoch_info}: "
    
    def get_metrics_message(self):
        accuracy_message = "" if self.metrics.accuracy == None else f"accuracy = {self.metrics.accuracy:.3f}    "
        loss_message = "" if self.metrics.loss == None else f"loss = {self.metrics.loss:.3f}    "
        return accuracy_message + loss_message
    
    def get_log_path(self):
        return os.path.join(os.getcwd(), 'results\log', self.information.type) +'.txt'


    



class Iterative_Performance_Presenter(Performance_Presenter):
    def __init__(self):
        super().__init__()
    
    def show_performance(self, information:classifier.Information, metrics:classifier.Metrics):
        super().show_performance(information, metrics)
        if self.is_looping_over():
            self.show_figure()
    
    def save_performance(self, information:classifier.Information, metrics:classifier.Metrics):
        super().save_performance(information, metrics)
        if self.is_looping_over():
            self.save_figure()




    def show_figure(self):
        self.show_train_valid_accuracy_figure()
        self.show_train_valid_loss_figure()

    def save_figure(self):
        self.save_train_valid_accuracy_figure()
        self.save_train_valid_loss_figure()

    def is_looping_over(self):
        if len(self.metrics.train_accuracy) == self.information.epoch_quantity:
            if len(self.metrics.valid_accuracy) == self.information.epoch_quantity:
                if self.information.state == "Train" or self.information.state == "Valid": 
                    return True
        else:
            return False




    def show_train_valid_accuracy_figure(self):
        self.get_figure(self.metrics.train_accuracy, self.metrics.valid_accuracy, 'accuracy')
        plt.show()
        plt.close()

    def show_train_valid_loss_figure(self):
        self.get_figure(self.metrics.train_loss, self.metrics.valid_loss, 'loss')
        plt.show()
        plt.close()

    def save_train_valid_accuracy_figure(self):
        metric = 'accuracy'
        self.get_figure(self.metrics.train_accuracy, self.metrics.valid_accuracy, metric)
        plt.savefig(os.path.join(os.getcwd(), 'results', 'figure', f'{metric}', f'{self.get_file_infomation()}') + '.png')
        plt.close()

    def save_train_valid_loss_figure(self):
        metric = 'loss'
        self.get_figure(self.metrics.train_loss, self.metrics.valid_loss, metric)
        plt.savefig(os.path.join(os.getcwd(), 'results', 'figure', f'{metric}', f'{self.get_file_infomation()}') + '.png')
        plt.close()
                 


    def get_figure(self, train_list, valid_list, metric = "accuracy/loss/..."):
        fig, ax = plt.subplots(figsize=(8,4))
        plt.title(f"{metric} {self.get_file_infomation()}")
        plt.plot(train_list, label=f'train {metric}')
        plt.plot(valid_list, label=f'valid {metric}', linestyle='--')
        plt.legend()

    def get_file_infomation(self):
        return f'{self.information.type}_{self.information.timestamp}_Fold{self.information.fold_number}'





'''class Feature_Importance_Presenter:
    def __init__(self) -> None:
        pass

    def set_show_and_save_feature_importance(self, train_data, feature_names,  number_repetition=10): ##### refine: depend on cfg, too many lines
        filepath = os.path.join(cfg.absolute_ML_HW1_path, f'results\plots', f'feature_importance_{self.filename}')
        utils.save_bar_chart(filepath, feature_names, self.features_importance) 
        message = self.get_feature_importance_message()
        self.show_message(message)
        self.save_message(message)
        return None
    
    def get_feature_importance_message(self): ##### refine: depend on cfg. and too ugly
        message = f""
        for i in range(len(self.features_importance)):
            message = message + f"{cfg.feature_names[i] + ': ':<25}{self.features_importance[i]:.3f} \n"
        return message

'''



'''class Tree_Presenter:
    def __init__(self):
        pass

    def show_the_branch(self) -> None:
        pass
'''

'''class Decision_Tree_Presenter(Tree_Presenter):
    def __init__(self, treenode:model.utils.TreeNode):
        self.treenode = treenode

    def show_the_branch(self):
        if self.treenode.is_leaf():
            self.print_indent(self.treenode)
            print(f"class: {self.treenode.major_label}")
            
        else:
            self.print_indent(self.treenode)
            print(f"{cfg.feature_names[self.treenode.decision_arguments['feature_index']]} > {self.treenode.decision_arguments['feature_median']:.3f}")
            self.treenode.left_child.show_the_branch()
            self.print_indent(self.treenode)
            print(f"{cfg.feature_names[self.treenode.decision_arguments['feature_index']]} <= {self.treenode.decision_arguments['feature_median']:.3f}")
            self.treenode.right_child.show_the_branch()

    def print_indent(self, treenode:model.utils.TreeNode):
        for i in range(treenode.depth - 1):
            print(f"|   ", end ='')
        print(f"|---", end='')
      
'''




if __name__ == "__main__":
    import unittest

    class Test_Performance_Presenter(unittest.TestCase):

        def setUp(self) -> None:
            self.presenter = Performance_Presenter()
            self.presenter.information.type="TESTING"
            self.presenter.information.epoch_quantity = 1000
            self.presenter.information.epoch_number = 666
            self.presenter.information.fold_number = 0
            self.presenter.metrics.accuracy = 100
            self.presenter.metrics.loss = None




        def test_is_classifier_name_existing(self):
            self.assertFalse(self.presenter.is_classifier_name_existing())

        def test_get_metrics_message(self):
            self.assertMultiLineEqual(self.presenter.get_metrics_message(), "accuracy = 100.000    ")

        def test_get_header_message(self):
            self.assertMultiLineEqual(self.presenter.get_header_message(), "Train/Test/Valid Epoch 667/1000: ")

        def test_get_message(self):
            self.presenter.save_message()
            target_message = "Train/Test/Valid Epoch 667/1000: accuracy = 100.000    "
            self.assertTrue(self._is_target_message_in_log(target_message))

        def test_save_classifier_name(self):
            self.presenter.save_classifier_name()
            target_message= f"{self.presenter.information.timestamp} \n"
            self.assertTrue(self._is_target_message_in_log(target_message))




        def tearDown(self):
            del self.presenter





        def _is_target_message_in_log(self, target_message):
            with open(self.presenter.get_log_path(), 'r') as fp:
                for line_number, line in enumerate(fp):
                    if target_message in line:
                        return True
            return False 

    unittest.main()




    #######################
    # TEST figure
    #######################
    '''def test_save_train_valid_loss_figure():
        information = classifier.Information(type="TESTING")
        information.epoch_quantity = 1000
        information.epoch_number = 666
        information.fold_number = 0
        metrics = classifier.Metrics()
        metrics.train_loss = [i*np.pi for i in range(1000)]
        metrics.valid_loss = [i for i in range(1000)]
        presenter = Iterative_Performance_Presenter(information, metrics)
        
        presenter.save_train_valid_loss_figure()
    test_save_train_valid_loss_figure()
        


    def test_save_train_valid_accuracy_figure():
        information = classifier.Information(type="TESTING")
        information.epoch_quantity = 1000
        information.epoch_number = 666
        information.fold_number = 0
        metrics = classifier.Metrics()
        metrics.train_accuracy = [i*np.e for i in range(1000)]
        metrics.valid_accuracy = [i*np.pi for i in range(1000)]
        presenter = Iterative_Performance_Presenter(information, metrics)
        
        presenter.save_train_valid_accuracy_figure()
    test_save_train_valid_accuracy_figure()'''


    '''def test_show_train_valid_loss_figure():
        presenter = Iterative_Performance_Presenter()
        presenter.information = classifier.Information(type="TESTING")
        presenter.information.epoch_quantity = 1000
        presenter.information.epoch_number = 666
        presenter.information.fold_number = 0
        presenter.metrics = classifier.Metrics()
        presenter.metrics.train_loss = [i*np.pi for i in range(1000)]
        presenter.metrics.valid_loss = [i for i in range(1000)]
        
        presenter.show_train_valid_loss_figure()
    test_show_train_valid_loss_figure()
        


    def test_show_train_valid_accuracy_figure():
        presenter = Iterative_Performance_Presenter()
        presenter.information = classifier.Information(type="TESTING")
        presenter.information.epoch_quantity = 1000
        presenter.information.epoch_number = 666
        presenter.information.fold_number = 0
        presenter.metrics = classifier.Metrics()
        presenter.metrics.train_accuracy = [i*np.e for i in range(1000)]
        presenter.metrics.valid_accuracy = [i*np.pi for i in range(1000)]
        
        presenter.show_train_valid_accuracy_figure()
    test_show_train_valid_accuracy_figure()'''



    '''def test_save_figure():
        information = classifier.Information(type="TESTING")
        metrics = classifier.Metrics()
        metrics.train_loss = [i*np.pi for i in range(1000)]
        metrics.valid_loss = [i for i in range(1000)]
        metrics.train_accuracy = [i*np.e for i in range(1000)]
        metrics.valid_accuracy = [i*np.pi for i in range(1000)]
        presenter = Performance_Presenter(information, metrics)
        
        presenter.save_figure()
    test_save_figure()'''



    '''def test_show_figure():
        information = classifier.Information(type="TESTING")
        metrics = classifier.Metrics()
        metrics.train_loss = [i*np.pi for i in range(1000)]
        metrics.valid_loss = [i for i in range(1000)]
        metrics.train_accuracy = [i*np.e for i in range(1000)]
        metrics.valid_accuracy = [i*np.pi for i in range(1000)]
        presenter = Performance_Presenter(information, metrics)
        
        presenter.show_figure()
    test_show_figure()'''



    '''def test_show_performance():
        information = classifier.Information(type="TESTING")
        information.epoch_quantity = 1000
        information.epoch_number = 666
        information.fold_quantity = 3
        information.fold_number = 0
        metrics = classifier.Metrics()
        metrics.accuracy = 100
        metrics.loss = None
        metrics.train_loss = [i*np.pi for i in range(1000)]
        metrics.valid_loss = [i for i in range(1000)]
        metrics.train_accuracy = [i*np.e for i in range(1000)]
        metrics.valid_accuracy = [i*np.pi for i in range(1000)]
        presenter = Performance_Presenter(information, metrics)
        
        presenter.show_performance()
    test_show_performance()



    def test_save_performance():
        information = classifier.Information(type="TESTING")
        information.epoch_quantity = 1000
        information.epoch_number = 666
        information.fold_quantity = 3
        information.fold_number = 0
        metrics = classifier.Metrics()
        metrics.accuracy = 100
        metrics.loss = None
        metrics.train_loss = [i*np.pi for i in range(1000)]
        metrics.valid_loss = [i for i in range(1000)]
        metrics.train_accuracy = [i*np.e for i in range(1000)]
        metrics.valid_accuracy = [i*np.pi for i in range(1000)]
        presenter = Performance_Presenter(information, metrics)
        
        presenter.save_performance()
    test_save_performance()'''

    

    #######################
    # TEST tree presenter
    #######################
    '''def test_build_the_tree():
        dt = model.utils.DecisionTree(**{'max_leaf_impurity':0.2, # the maximum of entropy in each leaves
                                 'max_samples_leaf':2,   # the maximum required samples in each leaves
                                 'max_tree_depth':2})
        dataset = model.utils.Dataset()
        dataset.features = np.array([[0,0],[1,1],[2,2],[3,3],[4,4],[5,5]])
        dataset.labels = np.array([1, -1, 1, -1, -1, -1])
        dt.build_the_tree(dataset)
        presenter = Tree_Presenter(dt)

        presenter.show_the_branch()
    test_build_the_tree()'''