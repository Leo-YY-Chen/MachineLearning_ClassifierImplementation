import numpy as np
import classifier
import os
import matplotlib.pyplot as plt




class Performance_Presenter:
    def __init__(self, info = classifier.Information(), metrics = classifier.Metrics()):
        self.info = info
        self.metrics = metrics

    def show_result(self):
        self.show_message()
        self.show_figure()
    
    def save_result(self):
        if not self.is_classifier_name_existing():
            self.save_classifier_name()
        self.save_message()
        self.save_figure()
    


    


    def show_message(self):
        print(self.get_message())
        
    def show_figure(self):
        self.show_train_valid_accuracy_figure()
        self.show_train_valid_loss_figure()

    def save_classifier_name(self):
        with open(os.path.join(os.getcwd, 'results\log', self.info.type) +'.txt', 'a') as f:
            f.write(f"------------------------------------------------------------------------------- \n")
            f.write(f"{self.info.timestamp} \n")

    def save_message(self):
        with open(os.path.join(os.getcwd, 'results\log', self.info.type) +'.txt', 'a') as f:
                f.write(f"{self.get_message()} \n")

    def save_figure(self):
        self.save_train_valid_accuracy_figure()
        self.save_train_valid_loss_figure()
        







    def get_message(self):
        return self.get_header_message() + self.get_metrics_message()

    def get_header_message(self):
        fold_info = "" if self.info.fold_quantity == np.nan else f" Fold {self.info.fold_number}/{self.info.fold_quantity}" 
        epoch_info = "" if self.info.epoch_quantity == np.nan else f" Epoch {self.info.epoch_number}/{self.info.epoch_quantity}"
        return f"{self.info.state + fold_info + epoch_info}: "
    
    def get_metrics_message(self):
        accuracy_message = "" if self.metrics.accuracy == np.nan else f"accuracy = {self.metrics.accuracy:.3f}    "
        loss_message = "" if self.metrics.loss == np.nan else f"loss = {self.metrics.loss:.3f}    "
        return accuracy_message + loss_message
    
    def show_train_valid_accuracy_figure(self):
        self.get_figure(self.metrics.train_accuracy, self.metrics.valid_accuracy, 'accuracy')
        plt.show()

    def show_train_valid_loss_figure(self):
        self.get_figure(self.metrics.train_loss, self.metrics.valid_loss, 'loss')
        plt.show()

    def save_train_valid_accuracy_figure(self):
        metric = 'accuracy'
        self.get_figure(self.metrics.train_accuracy, self.metrics.valid_accuracy, metric)
        plt.savefig(os.path.join(os.getcwd, 'results', 'figure', f'{metric}', f'{self.info.type}_{self.info.timestamp}') + '.png')

    def save_train_valid_loss_figure(self):
        metric = 'loss'
        self.get_figure(self.metrics.train_loss, self.metrics.valid_loss, metric)
        plt.savefig(os.path.join(os.getcwd, 'results', 'figure', f'{metric}', f'{self.info.type}_{self.info.timestamp}') + '.png')



    def get_figure(self, train_list, valid_list, metric = "accuracy/loss/..."):
        fig, ax = plt.subplots(figsize=(8,4))
        plt.title(f"{metric}")
        plt.plot(train_list, label=f'train {metric}')
        plt.plot(valid_list, label=f'valid {metric}', linestyle='--')
        plt.legend()



    def is_classifier_name_existing(self):
        with open(os.path.join(os.getcwd, 'results\log', self.info.type) +'.txt', 'r') as fp:
            for line_number, line in enumerate(fp):
                if self.info.timestamp in line:
                    return True
        return False
                 






class Feature_Importance_Presentor:
    def __init__(self) -> None:
        pass







if __name__ == "__main__":
    #######################
    # TEST perfomance
    #######################
    def test_get_state_message():
        clf = Performance_Presenter()
        message = clf.get_state_message("Train", 1, "number")
        if message == "Train Fold 1 Epoch number: accuracy = nan, loss = nan":
            print("passing")
        else:
            print("fail")
    test_get_state_message()