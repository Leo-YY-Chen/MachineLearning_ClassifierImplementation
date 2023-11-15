import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold
import sys
sys.path.append('..')
import function as fn
from statistics import mode


class Treenode():
    def __init__(self, X, Y, max_leaf_entrpoy, max_leaf_size, max_tree_depth, depth=1):
        # Samples in the node
        self.X = X
        self.Y = Y
        self.dominant_label = 0
        
        # Attributes for create the leaves
        self.feature_idx = 0
        self.feature_median = 0
        self.entropy = self.cal_entropy(self.Y)
        self.max_leaf_entrpoy = max_leaf_entrpoy
        self.left = None
        self.right = None

        # Pre-pruning attributes
        self.depth = depth
        self.max_leaf_size = max_leaf_size
        self.max_tree_depth= max_tree_depth

        # Post-prunning attributes (cost-complexity prunning)
        self.alpha_eff = np.nan
        self.num_branch_leaves = 0
        self.branch_entropy = 0

    def cal_entropy(self, labels):
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
        # print(f"depth = {self.depth}")
        # print(f"num_samples = {self.Y.shape[0]}")
        # print(f"self.dominant_label={self.dominant_label}")

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
            children_entropy = self.cal_entropy(left_Y) + self.cal_entropy(right_Y)
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

    def cal_num_branch_leaves(self):
        if self.left == None:
            self.num_branch_leaves = 1
            return self.num_branch_leaves
        else:
            self.num_branch_leaves = 0
            self.num_branch_leaves += self.left.cal_num_branch_leaves()
            self.num_branch_leaves += self.right.cal_num_branch_leaves()
            return self.num_branch_leaves
        
    def cal_branch_entropy(self):
        if self.left == None:
            self.branch_entropy = self.entropy
            #print(f"depth, entropy, branch_entropy = {self.depth, self.entropy, self.branch_entropy}")
            return self.branch_entropy
        else:
            self.branch_entropy += self.left.cal_branch_entropy()
            self.branch_entropy += self.right.cal_branch_entropy()
            #print(f"depth, entropy, branch_entropy = {self.depth, self.entropy, self.branch_entropy}")
            return self.branch_entropy

    def cal_alpha_eff(self, num_tree_leaves, alpha_eff_table):
        if self.left == None:
            self.alpha_eff = np.nan
            alpha_eff_table.append(self.alpha_eff)
            #print(f"depth, alpha_eff = {self.depth, self.alpha_eff}")
            return None
        else:
            self.alpha_eff = (self.entropy - self.branch_entropy) / (num_tree_leaves-1)
            alpha_eff_table.append(self.alpha_eff)
            self.left.cal_alpha_eff(num_tree_leaves, alpha_eff_table)
            self.right.cal_alpha_eff(num_tree_leaves, alpha_eff_table)
            #print(f"depth, alpha_eff = {self.depth, self.alpha_eff}")
            return None

    def remove_branch(self, min_alpha_eff):
        # check if meet the leaves
        if self.left == None:
            return None
        
        # remove branch depend on the value of (min_alpha_eff)
        if self.alpha_eff == min_alpha_eff:
            self.left = None
            self.right = None
            #print(f"depth, alpha_eff = {self.depth, self.alpha_eff}")
            #print(f"removed")
            return None
        else:
            #print(f"depth, alpha_eff = {self.depth, self.alpha_eff}")
            self.left.remove_branch(min_alpha_eff)
            self.right.remove_branch(min_alpha_eff)
            return None

    def replace_data(self, X_test, Y_test):
        # Replace data of the nodes
        # Stop criteria
        if self.left == None:
            self.entropy = self.cal_entropy(Y_test)
            return None
        else:
            # For each node, classify the data (X_test, Y_test) and update entropy (self.entropy)
            # by current attributes (self.feature_idx, self.feature_median)
            left_samples_id_list = (X_test[:, self.feature_idx] > self.feature_median)
            right_samples_id_list = (X_test[:, self.feature_idx] <= self.feature_median)
            self.left.X, self.left.Y = X_test[left_samples_id_list, :], Y_test[left_samples_id_list]
            self.right.X, self.right.Y = X_test[right_samples_id_list, :], Y_test[right_samples_id_list]
            self.left.entropy = self.cal_entropy(self.left.Y)
            self.right.entropy = self.cal_entropy(self.right.Y)

            # Recursively update the data in each node
            self.left.replace_data(self.left.X, self.left.Y)
            self.right.replace_data(self.right.X, self.right.Y)


class PDTClassifier:
    def __init__(self, max_leaf_entrpoy=0.2, max_leaf_size = 1, max_tree_depth=100, alpha = 0.1):
        # param for model construction
        # max_leaf_entrpoy: the maximum of entropy in each leaves
        # max_leaf_size: the maximum of samples in each leaves
        # max_tree_depth: the maximum of depth of the tree
        self.max_leaf_entrpoy = max_leaf_entrpoy
        self.max_leaf_size = max_leaf_size 
        self.max_tree_depth = max_tree_depth
        self.root = None
        self.alpha = alpha
        self.min_alpha_eff = -1
        self.alpha_eff_table = []

        # param for coding convinience
        self.model_name = 'PDT'
        self.time = datetime.now().strftime('%Y%m%d_%H%M%S')

    def cost_complexity_prunning(self):
        # ref: https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning
        # Find (min_alpha_eff)
        self.root.cal_branch_entropy()
        self.root.cal_alpha_eff(self.root.cal_num_branch_leaves(), self.alpha_eff_table)
        self.min_alpha_eff = np.nanmin(self.alpha_eff_table)

        # Prune when there exist the leaves and min_alpha_eff(nodes) < alpha
        while (self.root.left != None) and (self.min_alpha_eff < self.alpha):
            # Delete the weakest branches
            self.root.remove_branch(self.min_alpha_eff)
            # Recalculate the min_alpha_eff
            self.alpha_eff_table = []
            self.root.cal_alpha_eff(self.root.cal_num_branch_leaves(), self.alpha_eff_table)
            self.min_alpha_eff = np.nanmin(self.alpha_eff_table)

    def fit(self, X, Y):
        # Given train data (X, Y), construct the tree by recursively splitting
        self.root = Treenode(X, Y, self.max_leaf_entrpoy, self.max_leaf_size, self.max_tree_depth)
        self.root.grow_tree()
        self.cost_complexity_prunning()

        # Calculate the accuracy (train_acc)
        true_pred = self.find_leaf_and_cal_acc(self.root, 0)
        train_acc = true_pred/Y.shape[0]
        return train_acc
        
    def pred(self, X_test, Y_test):
        # Predict the test data by the trained tree
        self.root.replace_data(X_test, Y_test)
        true_pred = self.find_leaf_and_cal_acc(self.root, 0)
        train_acc = true_pred/Y_test.shape[0]
        return train_acc

    def find_leaf_and_cal_acc(self, node, true_pred):
        if node.left == None:
            true_pred = np.sum(node.Y == node.dominant_label)
            return true_pred
        else:
            true_pred = true_pred + self.find_leaf_and_cal_acc(node.left, 0)
            true_pred = true_pred + self.find_leaf_and_cal_acc(node.right, 0)
            return true_pred
    
    def train(self, X_train, X_valid, Y_train, Y_valid, k_fold=1, k_fold_iter=0):
        # Train and valid the model by (X_train, X_valid, Y_train, Y_valid)
        train_acc = self.fit(X_train, Y_train)
        valid_acc = self.pred(X_valid, Y_valid)
        print(f"train_acc = {train_acc}, valid_acc = {valid_acc}")

        # Save the acc
        tag = f'kfold{k_fold}-{k_fold_iter}'
        filepath = f'{self.model_name}_{tag}_{self.time}'
        lists = {"train_acc":train_acc, "valid_acc":valid_acc}
        fn.save_dict('../results/acc_loss_lists/' + filepath + '.csv', lists)

        # Save the model
        param = {"X_train":X_train, 
                 "Y_train":Y_train, 
                 "max_leaf_entrpoy": self.max_leaf_entrpoy,
                 "max_leaf_size":self.max_leaf_size, 
                 "max_tree_depth":self.max_tree_depth}
        fn.save_dict('./checkpoint/' + filepath + '.csv', param)

    def train_k_fold(self, X, Y, k_fold = 3):
        kf = KFold(n_splits=k_fold, random_state=83, shuffle=True)
        for i, (train_index, valid_index) in enumerate(kf.split(X,y=Y)):
            print(f"Fold {i}:")
            self.train( X[train_index], X[valid_index], Y[train_index], Y[valid_index], k_fold=k_fold, k_fold_iter=i)


if __name__ == '__main__': 
    X = np.random.randn(30,8)
    Y = np.random.randn(30,)
    Y[Y>=0] = 1
    Y[Y<0] = -1
    #print(f"X = {X}")
    #print(f"Y = {Y}")
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=83)
    #print(x_test.shape)

    cls = PDTClassifier()
    cls.train(x_train, x_test, y_train, y_test)

    cls2 = PDTClassifier()
    cls2.train_k_fold(X,Y)