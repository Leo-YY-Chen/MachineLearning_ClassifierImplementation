import numpy as np
import matplotlib.pyplot as plt
import os
import classifier



class Performance_Presenter:
    def __init__(self):
        self.information = classifier.Information()
        self.metrics = classifier.Metrics()

    def show_performance(self, information:classifier.Information, metrics:classifier.Metrics):
        self.set_information_and_metrics(information, metrics)
        self.show_message()
        self.show_figure()
    
    def save_performance(self, information:classifier.Information, metrics:classifier.Metrics):
        self.set_information_and_metrics(information, metrics)
        if not self.is_classifier_name_existing():
            self.save_classifier_name()
        self.save_message()
        self.save_figure()






    def set_information_and_metrics(self, information:classifier.Information, metrics:classifier.Metrics):
        self.information = information
        self.metrics = metrics


    


    def show_message(self):
        print(self.get_message())
        
    def show_figure(self):
        self.show_train_valid_accuracy_figure()
        self.show_train_valid_loss_figure()

    def save_classifier_name(self):
        with open(os.path.join(os.getcwd(), 'results\log', self.information.type) +'.txt', 'a') as f:
            f.write(f"------------------------------------------------------------------------------- \n")
            f.write(f"{self.information.timestamp} \n")

    def save_message(self):
        with open(os.path.join(os.getcwd(), 'results\log', self.information.type) +'.txt', 'a') as f:
                f.write(f"{self.get_message()} \n")

    def save_figure(self):
        self.save_train_valid_accuracy_figure()
        self.save_train_valid_loss_figure()
        







    def get_message(self):
        return self.get_header_message() + self.get_metrics_message()
    
    def show_train_valid_accuracy_figure(self):
        self.get_figure(self.metrics.train_accuracy, self.metrics.valid_accuracy, 'accuracy')
        plt.show()

    def show_train_valid_loss_figure(self):
        self.get_figure(self.metrics.train_loss, self.metrics.valid_loss, 'loss')
        plt.show()

    def save_train_valid_accuracy_figure(self):
        metric = 'accuracy'
        self.get_figure(self.metrics.train_accuracy, self.metrics.valid_accuracy, metric)
        plt.savefig(os.path.join(os.getcwd(), 'results', 'figure', f'{metric}', f'{self.information.type}_{self.information.timestamp}') + '.png')

    def save_train_valid_loss_figure(self):
        metric = 'loss'
        self.get_figure(self.metrics.train_loss, self.metrics.valid_loss, metric)
        plt.savefig(os.path.join(os.getcwd(), 'results', 'figure', f'{metric}', f'{self.information.type}_{self.information.timestamp}') + '.png')






    def get_header_message(self):
        fold_info = "" if self.information.fold_quantity == None else f" Fold {self.information.fold_number + 1}/{self.information.fold_quantity}" 
        epoch_info = "" if self.information.epoch_quantity == None else f" Epoch {self.information.epoch_number + 1}/{self.information.epoch_quantity}"
        return f"{self.information.state + fold_info + epoch_info}: "
    
    def get_metrics_message(self):
        accuracy_message = "" if self.metrics.accuracy == None else f"accuracy = {self.metrics.accuracy:.3f}    "
        loss_message = "" if self.metrics.loss == None else f"loss = {self.metrics.loss:.3f}    "
        return accuracy_message + loss_message

    def get_figure(self, train_list, valid_list, metric = "accuracy/loss/..."):
        fig, ax = plt.subplots(figsize=(8,4))
        plt.title(f"{metric}")
        plt.plot(train_list, label=f'train {metric}')
        plt.plot(valid_list, label=f'valid {metric}', linestyle='--')
        plt.legend()







    def is_classifier_name_existing(self):
        with open(os.path.join(os.getcwd(), 'results\log', self.information.type) +'.txt', 'r') as fp:
            for line_number, line in enumerate(fp):
                if self.information.timestamp in line:
                    return True
        return False
                 






class Feature_Importance_Presentor:
    def __init__(self) -> None:
        pass







if __name__ == "__main__":
    #######################
    # TEST perfomance
    #######################
    '''def test_get_state_message():
        clf = Performance_Presenter()
        message = clf.get_state_message("Train", 1, "number")
        if message == "Train Fold 1 Epoch number: accuracy = nan, loss = nan":
            print("passing")
        else:
            print("fail")
    test_get_state_message()'''




    #######################
    # TEST perfomance
    #######################
    '''def test_is_classifier_name_existing():
        info = classifier.Information(type="TESTING")
        mtcs = classifier.Metrics()
        psntr = Performance_Presenter(info, mtcs)

        if psntr.is_classifier_name_existing() == False:
            print("passing")
        else:
            print("fail")
    test_is_classifier_name_existing()
        


    def test_get_metrics_message():
        info = classifier.Information(type="TESTING")
        mtcs = classifier.Metrics()
        mtcs.accuracy = 100
        mtcs.loss = None
        psntr = Performance_Presenter(info, mtcs)
        
        if psntr.get_metrics_message() == "accuracy = 100.000    ":
            print("passing")
        else:
            print("fail")
    test_get_metrics_message()


    def test_get_header_message():
        info = classifier.Information(type="TESTING")
        info.epoch_quantity = 1000
        info.epoch_number = 666
        info.fold_number = 0
        mtcs = classifier.Metrics()
        psntr = Performance_Presenter(info, mtcs)
        
        if psntr.get_header_message() == "Train/Test/Valid Epoch 667/1000: ":
            print("passing")
        else:
            print("fail")
    test_get_header_message()


    def test_save_train_valid_loss_figure():
        information = classifier.Information(type="TESTING")
        information.epoch_quantity = 1000
        information.epoch_number = 666
        information.fold_number = 0
        metrics = classifier.Metrics()
        metrics.train_loss = [i*np.pi for i in range(1000)]
        metrics.valid_loss = [i for i in range(1000)]
        presenter = Performance_Presenter(information, metrics)
        
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
        presenter = Performance_Presenter(information, metrics)
        
        presenter.save_train_valid_accuracy_figure()
    test_save_train_valid_accuracy_figure()'''


    '''def test_show_train_valid_loss_figure():
        information = classifier.Information(type="TESTING")
        information.epoch_quantity = 1000
        information.epoch_number = 666
        information.fold_number = 0
        metrics = classifier.Metrics()
        metrics.train_loss = [i*np.pi for i in range(1000)]
        metrics.valid_loss = [i for i in range(1000)]
        presenter = Performance_Presenter(information, metrics)
        
        presenter.show_train_valid_loss_figure()
    test_show_train_valid_loss_figure()
        


    def test_show_train_valid_accuracy_figure():
        information = classifier.Information(type="TESTING")
        information.epoch_quantity = 1000
        information.epoch_number = 666
        information.fold_number = 0
        metrics = classifier.Metrics()
        metrics.train_accuracy = [i*np.e for i in range(1000)]
        metrics.valid_accuracy = [i*np.pi for i in range(1000)]
        presenter = Performance_Presenter(information, metrics)
        
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
        


    '''def test_get_message():
        information = classifier.Information(type="TESTING")
        information.epoch_quantity = 1000
        information.epoch_number = 666
        information.fold_number = 0
        metrics = classifier.Metrics()
        metrics.accuracy = 100
        metrics.loss = None
        presenter = Performance_Presenter(information, metrics)
        
        presenter.save_message()

        message= "Train/Test/Valid Epoch 667/1000: accuracy = 100.000    "
        with open(os.path.join(os.getcwd(), 'results\log', presenter.info.type) +'.txt', 'r') as fp:
            for line_number, line in enumerate(fp):
                if message in line:
                    message = "passing"
        if message == "passing":
            print("passing")
        else:
            print("fail")
    test_get_message()'''



    '''def test_save_classifier_name():
        information = classifier.Information(type="TESTING")
        information.epoch_quantity = 1000
        information.epoch_number = 666
        information.fold_number = 0
        metrics = classifier.Metrics()
        metrics.accuracy = 100
        metrics.loss = None
        presenter = Performance_Presenter(information, metrics)
        
        presenter.save_classifier_name()

        message= f"{presenter.info.timestamp} \n"
        with open(os.path.join(os.getcwd(), 'results\log', presenter.info.type) +'.txt', 'r') as fp:
            for line_number, line in enumerate(fp):
                if message in line:
                    message = "passing"
        if message == "passing":
            print("passing")
        else:
            print("fail")
    test_save_classifier_name()'''



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

    