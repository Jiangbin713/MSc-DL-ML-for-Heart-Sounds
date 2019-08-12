#  -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 18:40:14 2019

@author: jiang

What has been done in this file?

    1. GMM unsupervised learning to check the features
    2. Grid search to tune the params, also use cross validation
    3. 5 classical machine learning algorithm : SVM ,Decision tree, Random forest, AdaBoost, GradientBoosting

"""
import pickle
from __future__ import division, print_function
import numpy as np
from sklearn import svm
from sklearn.mixture import GMM
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.utils import compute_class_weight
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings('ignore')

# evaluate testing data
def evaluate_on_test_data(model):
    predictions = model.predict(X_test)
    correct = 0
    for i in range(len(y_test)):
        if predictions[i] == y_test[i]:
            correct += 1
    accuracy = 100*correct/len(y_test)
    return accuracy

# evaluate training data
def evaluate_on_train_data(model):
    predictions = model.predict(X_train)
    correct = 0
    for i in range(len(y_train)):
        if predictions[i] == y_train[i]:
            correct += 1
    accuracy = 100*correct/len(y_train)
    return accuracy

path = r'D:\CVML\Project\Heartchallenge_sound\Py_code\MachineLearn\Self\Saved_params'

"""
Load data
"""
x_train_A = pickle.load(open(path +r'\x_train_A.txt', 'rb') )
x_train_B = pickle.load(open(path +r'\x_train_B.txt', 'rb') )


y_train_A = pickle.load(open(path +r'\y_train_labelA.txt', 'rb') )
y_train_B = pickle.load(open(path +r'\y_train_labelB.txt', 'rb') )


x_test_A = pickle.load(open(path +r'\x_test_A.txt', 'rb') )
x_test_B = pickle.load(open(path +r'\x_test_B.txt', 'rb') )



"""
Compute classes weights
"""
classes_A = [0,1,2,3]
classes_B = [0,1,2]
classes_weight_A_list = compute_class_weight('balanced',classes_A,y_train_A)
classes_weight_B_list = compute_class_weight('balanced',classes_B,y_train_B)
classes_weight_A = {}
classes_weight_B = {}

for num in range(0, len(classes_weight_A_list)):

    if num < 3:
    
        classes_weight_A[num] = classes_weight_A_list[num]
        classes_weight_B[num] =classes_weight_B_list[num]
    else:
        classes_weight_A[num] = classes_weight_A_list[num]


class config():
    def __init__(self,
                 ###########  data   ############
                 X_A = x_train_A, y_A = y_train_A,
                 X_B = x_train_B, y_B = y_train_B,
                 test_A = x_test_A, test_B = x_test_B,
                 ########### class_weights #############
                 classes_weight_A = classes_weight_A,
                 classes_weight_B = classes_weight_B,        
                 ):
        # training data
        self.X_A = X_A 
        self.X_B = X_B  
        
        # training labels
        self.y_A = y_train_A
        self.y_B = y_train_B 
        
        # testing data
        self.test_A = x_test_A 
        self.test_B = x_test_B
        
        # classes_weight
        self.classes_weight_A = classes_weight_A 
        self.classes_weight_B = classes_weight_B
        
                 
config = config()



#####################################################################
##################   Gaussian Mixture Models     ####################
#####################################################################

"""
Evaluate the features: to see if the real label and the GMM output results align each other
"""
gmm_model = GMM(n_components = 4, 
                covariance_type = 'full',         #'full' ‘tied’ 'diag' 'spherical'
                n_iter = 300, 
                verbose = 1,
                tol = 1e-3,
                n_init = 10   #run many time and return the best result
                            ).fit(config.X_A)

results = gmm_model.predict(config.X_A).tolist()

x_index = np.arange(0, len(config.y_A))

plt.figure(figsize = (10,5))
plt.scatter(x_index, config.y_A, c = 'r', marker= 'o', label = 'Real labels')
plt.scatter(x_index, results, c='b', marker= '^', label = 'GMM results')
plt.legend()
plt.show()


'''
Given a model, we can use one of several means to evaluate how
well it fits the data. For example, there is the Aikaki information Criterion(AIC)
and the Bayesian information Crriterion(BIC)

'''
print(gmm_model.bic(config.X))
print(gmm_model.aic(config.X))
n_estimators = np.arange(1,8)
gmm_models = [GMM(n_components = n, 
                covariance_type = 'full',         
                n_iter = 300, 
                verbose = 1,
                tol = 1e-3,
                n_init = 10).fit(config.X) for n in n_estimators]
bics = [gmm_model.bic(config.X) for gmm_model in gmm_models]
aics = [gmm_model.aic(config.X) for gmm_model in gmm_models]

plt.plot(n_estimators, bics, label='BIC')
plt.plot(n_estimators, aics, label='AIC')
plt.legend()


#see gmm means covariance and weights
#gmm_model.means_
#gmm_model.covars_
#gmm_model.weights_



#####################################################################
########################   SVM     ##################################
#####################################################################
   
X_train, X_test, y_train, y_test = train_test_split(config.X_B, config.y_B, test_size = 0.2, random_state = 20) # Split train_valid data

"""
Grid Search
    find the best params
    test one by one
"""
gamma =[6.7]   #np.arange(5, 10, 0.1)
C = np.arange(5, 10, 0.2)
degree = [3]  #np.arange(1, 15, 1)

tuned_parameters = [
                    #{'kernel': ['rbf'], 'C': C, 'gamma': gamma,'degree': degree},
                   # {'kernel': ['linear'], 'C': C},
                    {'kernel': ['poly'], 'C': C, 'gamma': gamma, 'degree': degree}
                    ]

scores = ['precision', 'recall']


for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

     # 调用 GridSearchCV，将 SVC(), tuned_parameters, cv=5, 还有 scoring 传递进去，
    svm_model = GridSearchCV(svm.SVC(max_iter=1500,class_weight=config.classes_weight_B), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    # 用训练集训练这个学习器 clf
    svm_model.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    
    # 再调用 clf.best_params_ 就能直接得到最好的参数搭配结果
    print(svm_model.best_params_)
    
    print("Grid scores on development set:")
    means = svm_model.cv_results_['mean_test_score']
    stds = svm_model.cv_results_['std_test_score']
    
    # check mean and std scores 
    for mean, std, params in zip(means, stds, svm_model.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

"""
Directly train
"""                                            
predictions = []
accuracies = []


svm_model = svm.SVC(kernel='poly', C=0.5 ,class_weight=config.classes_weight_B, 
                gamma = 4, degree = 5 ,max_iter=1500)

svm_model.fit(X_train, y_train)

acc= evaluate_on_test_data(svm_model)

accuracies.append(acc)

print("{} % accuracy obtained ".format(acc))

results = svm_model.predict(config.test_B)

print(results)



#####################################################################
########################   Decision Tree    #########################
#####################################################################

X_train, X_test, y_train, y_test = train_test_split(config.X_A, config.y_A, test_size = 0.2)

"""
Grid Search
    find the best params
    test one by one
"""

max_depth = np.arange(10,200,5)

tuned_parameters = [ {'max_depth': max_depth} ]

scores = ['precision', 'recall']


for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

     # 调用 GridSearchCV，将 SVC(), tuned_parameters, cv=5, 还有 scoring 传递进去，
    svm_model = GridSearchCV(tree.DecisionTreeClassifier(
                                            criterion = 'gini',   # gini or entropy
                                            #max_features =
                                            max_depth = 100,
                                            min_samples_split = 3,
                                            min_samples_leaf = 3,
                                            class_weight = config.classes_weight_B,
                                                ), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    # 用训练集训练这个学习器 clf
    svm_model.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    
    # 再调用 clf.best_params_ 就能直接得到最好的参数搭配结果
    print(svm_model.best_params_)
    
    print("Grid scores on development set:")
    means = svm_model.cv_results_['mean_test_score']
    stds = svm_model.cv_results_['std_test_score']
    
    # 看一下具体的参数间不同数值的组合后得到的分数是多少
    for mean, std, params in zip(means, stds, svm_model.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
              
    print("Detailed classification report:")
  
    
"""
Directly train
"""   
decision_tree_model = tree.DecisionTreeClassifier(
                                            criterion = 'gini',   # gini or entropy
                                            #max_features =
                                            max_depth = 50,
                                            min_samples_split = 3,
                                            min_samples_leaf = 3,
                                            class_weight = config.classes_weight_B,
                                                ).fit(X_train,y_train)

acc_train= evaluate_on_train_data(decision_tree_model)
acc_test= evaluate_on_test_data(decision_tree_model)

print("{} % train accuracy obtained ".format(acc_train))
print("{} % test accuracy obtained ".format(acc_test))

results = decision_tree_model.predict(config.test_A)
print(results)

#####################################################################
########################   Random Forests   #########################
#####################################################################

X_train, X_test, y_train, y_test = train_test_split(config.X_B, config.y_B, test_size = 0.2)


random_forest_model = RandomForestClassifier(
                                            n_estimators=10,
                                            criterion = 'gini',   # gini or entropy
                                            max_depth = 20,
                                            min_samples_split = 3,
                                            min_samples_leaf = 3,
                                            class_weight = config.classes_weight_B,
                                            oob_score = True
                                                ).fit(X_train,y_train)

acc_train= evaluate_on_train_data(random_forest_model)
acc_test= evaluate_on_test_data(random_forest_model)

print("{} % train accuracy obtained ".format(acc_train))
print("{} % test accuracy obtained ".format(acc_test))

results = random_forest_model.predict(config.test_B)
print(results)

#####################################################################
############################   AdaBoost   ###########################
#####################################################################

X_train, X_test, y_train, y_test = train_test_split(config.X_A, config.y_A, test_size = 0.2)    

AdaBoost_model = AdaBoostClassifier(        
                                            base_estimator = tree.DecisionTreeClassifier(criterion = 'gini',   # gini or entropy    
                                                                                         max_depth = 30,
                                                                                         min_samples_split = 3,
                                                                                         min_samples_leaf = 3,
                                                                                         class_weight = config.classes_weight_B),
                                            n_estimators=50, 
                                            learning_rate=0.09, 
                                            algorithm='SAMME.R', 
                                            
                                                ).fit(X_train,y_train)

acc_train= evaluate_on_train_data(AdaBoost_model)
acc_test= evaluate_on_test_data(AdaBoost_model)

print("{} % train accuracy obtained ".format(acc_train))
print("{} % test accuracy obtained ".format(acc_test))

results = AdaBoost_model.predict(config.test_A)
results = AdaBoost_model.predict(config.test_A)
print(results)

#####################################################################
######################GradientBoosting ##############################
#####################################################################

X_train, X_test, y_train, y_test = train_test_split(config.X_B, config.y_B, test_size = 0.2)    

GradientBoost_model = GradientBoostingClassifier(
                                            #Boost parameters
                                            n_estimators=8, 
                                            learning_rate=0.5, 
                                            subsample = 0.65,  #正则子采样 [0.5,0.8]
                                            loss = 'deviance',
                                            #weak classifier parameters
                                            max_depth = 20, 
                                            min_samples_split = 3,
                                            min_samples_leaf = 3,
                                            max_features = 'auto'
                                            
                                            
                                                ).fit(X_train,y_train)

acc_train= evaluate_on_train_data(GradientBoost_model)
acc_test= evaluate_on_test_data(GradientBoost_model)

print("{} % train accuracy obtained ".format(acc_train))
print("{} % test accuracy obtained ".format(acc_test))

results = GradientBoost_model.predict(config.test_B)
results = GradientBoost_model.predict(config.test_B)
print(results)

























