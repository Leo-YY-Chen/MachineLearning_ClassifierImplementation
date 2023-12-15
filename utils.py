import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import numpy as np
import config as cfg
from statistics import mode

class Dataset:
    def __init__(self, features = None, labels = None):
        self.features = features    #2D array
        self.labels = labels        #1D array
        self.k_fold_indices = None

    def load(self, datasest_path): ##### REFINE: this function depends on cfg files
        self.features = pd.read_csv(datasest_path, usecols = cfg.feature_names)
        self.labels = pd.read_csv(datasest_path, usecols = cfg.label_names)
        return None
    
    def preprocess(self):
        self.relabel_into_minus_or_plus_one()
        self.do_data_balancing()
        self.do_min_max_normalization()
        return None
    
    def relabel_into_minus_or_plus_one(self):
        self.labels[self.labels > 0] = 1
        self.labels[self.labels <= 0] = -1
        return None
    
    def do_data_balancing(self):
        self.do_oversampling()
        return None
    
    def do_oversampling(self):
        labels, numbers = self.get_labels_and_numbers_of_oversamples()
        self.clone_subset_randomly_by_labels_and_numbers(labels, numbers)
        return None
    
    def get_labels_and_numbers_of_oversamples(self):
        lebels, counts = np.unique(self.labels, return_counts=True)
        number_oversamples = np.max(counts) - counts
        return lebels, number_oversamples
    
    def clone_subset_randomly_by_labels_and_numbers(self, labels, numbers):
        for i, label in enumerate(labels):
            indices = self.get_indices_of_oversamples_with_label(numbers[i], label)
            self.clone_subset_by_indices(indices)
        return None
    
    def get_indices_of_oversamples_with_label(self, number_oversamples, label):
        indices = self.get_indices_with_label(label)
        indices_of_oversamples = np.random.choice(indices, number_oversamples)
        all_indices = np.array(range(len(self.features)))
        indices_of_oversamples = np.in1d(all_indices, indices_of_oversamples)
        return indices_of_oversamples
    
    def clone_subset_by_indices(self, indices):
        np.concatenate((self.features, self.features[indices]))
        np.concatenate((self.labels, self.labels[indices]))
        return None

    def get_indices_with_label(self, label):
        indices = np.where(self.labels == label)[0]
        return indices.tolist()
    
    def do_min_max_normalization(self):
        min = np.min(self.features, axis=0)
        max = np.max(self.features, axis=0)
        self.features = (self.features-min)/(max-min)
        return None
    
    def remove_unimportant_feature(self, feature_id_list): 
        np.delete(self.features, feature_id_list, axis=1)
        return None
    
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
        return None

    def set_k_fold_indices(self, n_splits):
        kf = KFold(n_splits=n_splits, random_state=83, shuffle=True)
        self.k_fold_indices = [(train_index, valid_index) for (train_index, valid_index) in kf.split(self.features)]
        return None

    def get_kth_fold_datasets(self, kth = 0):
        (train_index, valid_index) = self.k_fold_indices[kth]
        train_dataset = self.get_subset(train_index)
        valid_dataset = self.get_subset(valid_index)
        return train_dataset, valid_dataset

    def split_in_ratio_for_k_fold(self, n_splits):
        temp_dataset, test_dataset = self.split_in_ratio()
        temp_dataset.set_k_fold_indices(n_splits)
        return temp_dataset, test_dataset

class TreeNode:
    def __init__(self, depth = 1):
        self.depth = depth
        self.rules_of_building_children = None
        self.left_child = None  # bulid with datatset whose feature value > feature_median
        self.right_child = None # bulid with datatset whose feature value <= feature_median
  
    def build_the_tree(self):
        if self.is_stopping_criteria_being_met():
            return None
        self.left_child = TreeNode()
        self.right_child = TreeNode()
        self.left_child.build_the_tree()
        self.right_child.build_the_tree()
        return None
        
    def remove_the_children(self):
        self.left_child = None
        self.right_child = None
        return None
    
    def show_the_branch(self):
        return None

    def is_stopping_criteria_being_met(self):
        # return bool
        return True
    
    def is_leaf(self):
        if self.left_child==None and self.right_child==None:
            return True
        else:
            return False

class DecisionTreeNode(TreeNode):
    def __init__(self, depth = 1, **hyper_parameters):
        super().__init__(depth)
        self.hyper_parameters = hyper_parameters # max_leaf_impurity, max_samples_leaf, max_tree_depth
        self.decision_arguments = {'feature_index':None, 'feature_median':None}
        self.major_label = np.nan
        self.entropy = 0

    def build_the_tree(self, dataset): ##### REFINE: too many actions in a function
        self.set_node_attributes(dataset)
        if self.is_stopping_criteria_being_met(dataset):
            return None
        self.set_children()
        left_dataset, right_dataset = self.get_datasets_for_children(dataset)
        self.left_child.build_the_tree(left_dataset)
        self.right_child.build_the_tree(right_dataset)

    def set_children(self):
        self.left_child = DecisionTreeNode(self.depth+1, **self.hyper_parameters)
        self.right_child = DecisionTreeNode(self.depth+1, **self.hyper_parameters)
        return None

    def predict_labels(self, dataset):
        labels = []
        for i in range(len(dataset.labels)):
            predict_label = self.predict_by_decisions(dataset.get_subset([i]))
            labels.append(predict_label)
        return labels
    
    def show_the_branch(self):
        if self.is_leaf():
            self.print_indent()
            print(f"class: {self.major_label}")
            return None
        else:
            self.print_indent()
            print(f"{cfg.feature_names[self.decision_arguments['feature_index']]} > {self.decision_arguments['feature_median']:.3f}")
            self.left_child.show_the_branch()
            self.print_indent()
            print(f"{cfg.feature_names[self.decision_arguments['feature_index']]} <= {self.decision_arguments['feature_median']:.3f}")
            self.right_child.show_the_branch()
            return None
    
    def predict_by_decisions(self, datum):
        if self.is_leaf():
            return self.major_label
        else:
            if self.is_datum_in_left_child(datum):
                return self.left_child.predict_by_decisions(datum)
            else:
                return self.right_child.predict_by_decisions(datum)

    def is_stopping_criteria_being_met(self, dataset):
        if self.entropy <= self.hyper_parameters['max_leaf_impurity']:
            return True 
        elif dataset.labels.shape[0] <= self.hyper_parameters['max_samples_leaf']:
            return True
        elif self.depth >= self.hyper_parameters['max_tree_depth']:
            return True
        else:
            return False
    
    def set_node_attributes(self, dataset):
        self.set_decision_arguments(dataset)
        self.set_major_label(dataset)
        self.set_entropy(dataset)
        return None

    def set_decision_arguments(self, dataset):
        IGs = self.get_information_gains(dataset)
        self.decision_arguments['feature_index'] = np.argmax(IGs)
        self.decision_arguments['feature_median'] = np.median(dataset.features, axis=0)[np.argmax(IGs)]
        return None

    def set_major_label(self, dataset):
        self.major_label = mode(dataset.labels)
        return None
    
    def set_entropy(self, dataset):
        self.entropy = self.calculate_entropy(dataset.labels)
        return None

    def get_indices_by_decision_arguments(self, dataset):
        left_indices = dataset.features[:, self.decision_arguments['feature_index']] > self.decision_arguments['feature_median']
        right_indices = dataset.features[:, self.decision_arguments['feature_index']] <= self.decision_arguments['feature_median']
        return left_indices, right_indices

    def get_datasets_for_children(self, dataset):
        left_indices, right_indices = self.get_indices_by_decision_arguments(dataset)
        left_dataset = dataset.get_subset(left_indices)
        right_dataset = dataset.get_subset(right_indices)
        return left_dataset, right_dataset

    def calculate_entropy(self, labels):
        label_list = list(set(labels))
        label_prob_list = [np.sum(labels==lb)/labels.shape[0] for lb in label_list]
        label_entropy_list = [(0 if prob==0 else prob*np.log(prob)) for prob in label_prob_list]
        entropy = -np.sum(label_entropy_list)
        return entropy

    def get_information_gains(self, dataset):
        IGs = []
        for i, median in enumerate(np.median(dataset.features, axis=0)):
            left_labels = dataset.labels[dataset.features[:,i] > median]
            right_labels = dataset.labels[dataset.features[:,i] <= median]
            children_entropy = self.calculate_entropy(left_labels) + self.calculate_entropy(right_labels)
            IGs.append(self.entropy - children_entropy)
        IGs[IGs==0] = -100 # IG = 0 means the feature owing only two kind of values and it should be ignore  
        return IGs

    def is_datum_in_left_child(self, dataset):
        if dataset.features[0, self.decision_arguments['feature_index']] > self.decision_arguments['feature_median']:
            return True
        else:
            return False

    def print_indent(self):
        for i in range(self.depth - 1):
            print(f"|   ", end ='')
        print(f"|---", end='')
        return None

class PrunedDecisionTreeNode(DecisionTreeNode):
    def __init__(self, depth=1, **hyper_parameters):
        super().__init__(depth, **hyper_parameters)
        # Post-prunning attributes (cost-complexity prunning)
        self.hyper_parameters = hyper_parameters # max_leaf_impurity, max_samples_leaf, max_tree_depth, alpha
        self.alpha_eff = np.nan
        self.num_branch_leaves = 0
        self.branch_entropy = 0

    def build_and_prune_the_tree(self, dataset):
        self.build_the_tree(dataset)
        self.cost_complexity_prunning()

    def set_children(self):
        self.left_child = PrunedDecisionTreeNode(self.depth+1, **self.hyper_parameters)
        self.right_child = PrunedDecisionTreeNode(self.depth+1, **self.hyper_parameters)
        return None

    def set_num_branch_leaves(self):
        if self.is_leaf():
            self.num_branch_leaves = 1
            return self.num_branch_leaves
        else:
            self.num_branch_leaves = 0
            self.num_branch_leaves += self.left_child.set_num_branch_leaves()
            self.num_branch_leaves += self.right_child.set_num_branch_leaves()
            return self.num_branch_leaves
        
    def set_branch_entropy(self):
        if self.is_leaf():
            self.branch_entropy = self.entropy
            return self.branch_entropy
        else:
            self.branch_entropy += self.left_child.set_branch_entropy()
            self.branch_entropy += self.right_child.set_branch_entropy()
            return self.branch_entropy

    def set_alpha_eff(self, num_tree_leaves):
        if self.is_leaf():
            self.alpha_eff = np.nan
        else:
            self.alpha_eff = (self.entropy - self.branch_entropy) / (num_tree_leaves - 1)
            self.left_child.set_alpha_eff(num_tree_leaves)
            self.right_child.set_alpha_eff(num_tree_leaves)
        return None

    def remove_weakest_branch(self, min_alpha_eff):
        if self.is_leaf():
            return None
        if self.alpha_eff == min_alpha_eff:
            self.remove_the_children()
            return None
        else:
            self.left_child.remove_weakest_branch(min_alpha_eff)
            self.right_child.remove_weakest_branch(min_alpha_eff)
            return None
        
    def cost_complexity_prunning(self):
        # ref: https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning
        self.set_branch_entropy()
        self.set_alpha_eff(self.set_num_branch_leaves())
        while self.get_min_alpha_eff() < self.hyper_parameters['ccp_alpha']:
            self.remove_weakest_branch(self.get_min_alpha_eff())
            self.set_alpha_eff(self.set_num_branch_leaves())

    def get_min_alpha_eff(self):
        if self.is_leaf():
            min_alpha_eff = self.alpha_eff
        else:
            min_alpha_eff = np.nanmin([self.alpha_eff, self.left_child.get_min_alpha_eff(), self.right_child.get_min_alpha_eff()])
        return min_alpha_eff
    



def save_dict(filename, dict):
    with open(filename, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=dict.keys())
        writer.writeheader()
        writer.writerow(dict)

def save_plot(filename, train_list, valid_list, plotname='accuracy'):
    fig, ax = plt.subplots(figsize=(8,4))
    plt.title(plotname)
    plt.plot(train_list, label='train '+plotname)
    plt.plot(valid_list, label='valid '+plotname, linestyle='--')
    plt.legend()
    plt.savefig(filename)
    plt.show()

def save_bar_chart(filename, label_list, data_list, plotname='Feature Importance'):
    plt.title(plotname)
    plt.bar(label_list, data_list)
    plt.xticks(rotation=30, ha='right')
    plt.savefig(filename)
    plt.show()




if __name__ == '__main__':

    data = Dataset()
    data.load(cfg.datapath)
    data.preprocess()
    train_data, test_data = data.split_in_ratio()
    