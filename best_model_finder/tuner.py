from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics  import roc_auc_score,accuracy_score,confusion_matrix
from datetime import datetime
from sklearn.naive_bayes import GaussianNB

class Model_Finder:
    """
        This class shall  be used to find the model with best accuracy and AUC score.
    """

    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.clf = RandomForestClassifier()
        self.xgb = XGBClassifier(objective='binary:logistic')
        self.logreg = LogisticRegression()
        self.svc = SVC()
        self.gnb = GaussianNB()
        self.knn = KNeighborsClassifier()
        self.best_param_list = []
        self.perf_data = []
        self.model_list = []
        self.model_acc = []

    def get_best_params_for_naive_bayes(self,train_x,train_y):
        """
                Method Name: get_best_params_for_naive_bayes
                Description: get the parameters for the Naive Bayes's Algorithm which give the best accuracy.
                             Use Hyper Parameter Tuning.
                Output: The model with the best parameters
                On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_naive_bayes method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = {"var_smoothing": [1e-9,0.1, 0.001, 0.5,0.05,0.01,1e-8,1e-7,1e-6,1e-10,1e-11]}

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.gnb, param_grid=self.param_grid, cv=3,  verbose=3, n_jobs=-1)
            #finding the best parameters
            self.grid.fit(train_x, train_y)

            #extracting the best parameters
            self.var_smoothing = self.grid.best_params_['var_smoothing']

            tmp_dict = {'Model_Name': 'Gaussian NB', 'var_smoothing': self.var_smoothing}
            self.best_param_list.append(tmp_dict)

            #creating a new model with the best parameters
            self.gnb = GaussianNB(var_smoothing=self.var_smoothing)
            # training the mew model
            self.gnb.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'Naive Bayes best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_naive_bayes method of the Model_Finder class')

            return self.gnb
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_naive_bayes method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Naive Bayes Parameter tuning  failed. Exited the get_best_params_for_naive_bayes method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_svc(self,train_x,train_y):
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_svc method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = {'C':[0.1,1,10,50],'gamma':[1,0.5,0.1,0.01]}

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.svc, param_grid=self.param_grid, cv=5, verbose=3, n_jobs=-1)
            #finding the best parameters
            self.grid.fit(train_x, train_y)

            #extracting the best parameters
            self.C_value = self.grid.best_params_['C']
            self.gamma_value = self.grid.best_params_['gamma']

            tmp_dict = {'Model_Name':'Support Vector Classifier','C': self.C_value,'gamma': self.gamma_value}
            self.best_param_list.append(tmp_dict)

            #creating a new model with the best parameters
            self.clf = SVC(C=self.C_value, gamma=self.gamma_value)
            # training the mew model
            self.clf.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'SVC best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_svc method of the Model_Finder class')

            return self.clf
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_svc method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Support Vector Classifier Parameter tuning  failed. Exited the get_best_params_for_svc method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_logistic_reg(self,train_x,train_y):
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_logistic_reg method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = {'C':[0.1,1,10,100]}

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.logreg, param_grid=self.param_grid, cv=5,  verbose=3, n_jobs=-1)
            #finding the best parameters
            self.grid.fit(train_x, train_y)

            #extracting the best parameters
            self.c_value = self.grid.best_params_['C']

            tmp_dict = {'Model_Name': 'Logistic Regression','C': self.c_value}
            self.best_param_list.append(tmp_dict)

            #creating a new model with the best parameters
            self.clf = LogisticRegression(C=self.c_value)
            # training the mew model
            self.clf.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'Logistic Regression best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_logistic_reg method of the Model_Finder class')

            return self.clf
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_logistic_reg method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Logistic Regression Parameter tuning  failed. Exited the get_best_params_for_logistic_reg method of the Model_Finder class')
            raise Exception()


    def get_best_params_for_knn(self,train_x,train_y):
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_knn method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = {'algorithm' : ['ball_tree', 'kd_tree', 'brute'],
               'leaf_size' : [5,10,15,20,24,30],
               'n_neighbors' : [3,5,7,9,10,11]}

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.knn, param_grid=self.param_grid, cv=5, verbose=3, n_jobs=-1)
            #finding the best parameters
            self.grid.fit(train_x, train_y)

            #extracting the best parameters
            self.algo = self.grid.best_params_['algorithm']
            self.leaf_size = self.grid.best_params_['leaf_size']
            self.neigh = self.grid.best_params_['n_neighbors']

            tmp_dict = {'Model_Name': 'KNN Classifier','algorithm': self.algo,'leaf_size': self.leaf_size,'n_neighbors':self.neigh}
            self.best_param_list.append(tmp_dict)

            #creating a new model with the best parameters
            self.clf = KNeighborsClassifier(algorithm = self.algo, leaf_size =self.leaf_size, n_neighbors =self.neigh)
            # training the mew model
            self.clf.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'K-Nearest Neighbors Classification best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_knn method of the Model_Finder class')

            return self.clf
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_knn method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'K-Nearest Neighbors Classification Parameter tuning  failed. Exited the get_best_params_for_knn method of the Model_Finder class')
            raise Exception()


    def get_best_params_for_random_forest(self,train_x,train_y):
        """
            Method Name: get_best_params_for_random_forest
            Description: get the parameters for Random Forest Algorithm which give the best accuracy.
                         Use Hyper Parameter Tuning.
            Output: The model with the best parameters
            On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_random_forest method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = {"n_estimators": [10, 50, 100, 130], "criterion": ['gini', 'entropy'],
                               "max_depth": range(2, 4, 1), "max_features": ['auto', 'log2']}

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.clf, param_grid=self.param_grid, cv=5, verbose=3, n_jobs=-1)
            #finding the best parameters
            self.grid.fit(train_x, train_y)

            #extracting the best parameters
            self.criterion = self.grid.best_params_['criterion']
            self.max_depth = self.grid.best_params_['max_depth']
            self.max_features = self.grid.best_params_['max_features']
            self.n_estimators = self.grid.best_params_['n_estimators']

            tmp_dict = {'Model_Name': 'RandomForest Classifier', 'criterion': self.criterion, 'max_depth': self.max_depth,
                             'max_features': self.max_features, 'n_estimators': self.n_estimators}
            self.best_param_list.append(tmp_dict)

            #creating a new model with the best parameters
            self.clf = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion,
                                              max_depth=self.max_depth, max_features=self.max_features)
            # training the mew model
            self.clf.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'Random Forest best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_random_forest method of the Model_Finder class')

            return self.clf
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_random_forest method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Random Forest Parameter tuning  failed. Exited the get_best_params_for_random_forest method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_xgboost(self,train_x,train_y):

        """
            Method Name: get_best_params_for_xgboost
            Description: get the parameters for XGBoost Algorithm which give the best accuracy.
                         Use Hyper Parameter Tuning.
            Output: The model with the best parameters
            On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_xgboost = {

                'learning_rate': [0.5, 0.1, 0.01, 0.001],
                'max_depth': [3, 5, 10],
                'n_estimators': [10, 50, 100]

            }
            # Creating an object of the Grid Search class
            self.grid= GridSearchCV(XGBClassifier(objective='binary:logistic'),self.param_grid_xgboost, verbose=3,cv=5, n_jobs=-1)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.learning_rate = self.grid.best_params_['learning_rate']
            self.max_depth = self.grid.best_params_['max_depth']
            self.n_estimators = self.grid.best_params_['n_estimators']

            tmp_dict = {'Model_Name': 'XGBClassifier', 'learning_rate': self.learning_rate,
                            'max_depth': self.max_depth,
                            'n_estimators': self.n_estimators}
            self.best_param_list.append(tmp_dict)

            # creating a new model with the best parameters
            self.xgb = XGBClassifier(learning_rate=self.learning_rate, max_depth=self.max_depth, n_estimators=self.n_estimators)
            # training the mew model
            self.xgb.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'XGBoost best params: ' + str(
                                       self.grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return self.xgb
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            print(e)
            raise Exception()


    def get_performance_parameters(self,train_x,train_y,test_x,test_y,model_name,model,cluster_no):
        try:
            pref_dict={}

            now = datetime.now()
            date = now.date()
            current_time = now.strftime("%H:%M:%S")
            insert_date = str(date) + ' ' + str(current_time)
            pref_dict['Insert_Date']=str(insert_date)
            pref_dict['Cluster_No']=int(cluster_no)
            pref_dict['Model_Name']=model_name

            #Train Accuracy
            train_pred=model.predict(train_x)
            train_score=accuracy_score(train_y,train_pred)
            pref_dict['Train_Accuracy'] = round(train_score,2)
            self.logger_object.log(self.file_object, 'Train_Accuracy for ' + model_name + ' : ' + str(train_score))

            #Test Accuracy
            test_pred=model.predict(test_x)
            test_score=accuracy_score(test_y,test_pred)
            pref_dict['Test_Accuracy'] = round(test_score,2)
            self.logger_object.log(self.file_object, 'Test_Accuracy for ' + model_name + ' : ' + str(test_score))

            #Confusion Matrix
            conf_mat = confusion_matrix(test_y, test_pred)
            true_positive = conf_mat[0][0]
            false_positive = conf_mat[0][1]
            false_negative = conf_mat[1][0]
            true_negative = conf_mat[1][1]

            #Precision
            Precision = true_positive / (true_positive + false_positive)
            pref_dict['Precision'] = round(Precision,2)
            self.logger_object.log(self.file_object, 'Precision for ' + model_name + ' : ' + str(Precision))

            #Recall
            Recall = true_positive / (true_positive + false_negative)
            pref_dict['Recall'] = round(Recall,2)
            self.logger_object.log(self.file_object, 'Recall for ' + model_name + ' : ' + str(Recall))

            #F1 Score
            F1_Score = 2 * (Recall * Precision) / (Recall + Precision)
            pref_dict['F1_Score'] = round(F1_Score,2)
            self.logger_object.log(self.file_object, 'F1 Score for ' + model_name + ' : ' + str(F1_Score))

            #ROC AUC Score
            if len(test_y.unique()) != 1:
                auc = roc_auc_score(test_y, test_pred)
                pref_dict['ROC_AUC_Score'] = round(auc,2)
            self.logger_object.log(self.file_object, 'ROC AUC Score for ' + model_name + ' : ' + str(auc))

            self.perf_data.append(pref_dict)
            self.model_list.append(model_name)
            if len(test_y.unique()) == 1:
                self.model_acc.append(test_score)
            else:
                self.model_acc.append(auc)
        except Exception as e:
            print('Exception Occured: ', e)
            raise e


    def get_best_model(self,train_x,train_y,test_x,test_y,cluster_no):
        """
            Method Name: get_best_model
            Description: Find out the Model which has the best AUC score.
            Output: The best model name and the model object
            On Failure: Raise Exception
        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_model method of the Model_Finder class')

        try:
            model_list1=[]

            # create best model for GaussianNB
            self.gaussiannb = self.get_best_params_for_naive_bayes(train_x, train_y)
            model_list1.append(self.gaussiannb)
            print('Setting Performance Parameters GaussianNB: ')
            self.get_performance_parameters(train_x, train_y, test_x, test_y, 'GaussianNB', self.gaussiannb, cluster_no)

            # create best model for XGBoost
            print('Training XgBoost Model: ')
            self.xgboost= self.get_best_params_for_xgboost(train_x,train_y)
            model_list1.append(self.xgboost)
            print('Setting Performance Parameters XGBoost: ')
            self.get_performance_parameters(train_x, train_y, test_x, test_y, 'XGBoost', self.xgboost, cluster_no)

            # create best model for Random Forest
            print('Training Random Forest Model: ')
            self.random_forest=self.get_best_params_for_random_forest(train_x,train_y)
            model_list1.append(self.random_forest)
            print('Setting Performance Parameters RandomForest: ')
            self.get_performance_parameters(train_x, train_y, test_x, test_y, 'RandomForest', self.random_forest, cluster_no)

            # create best model for SVC
            self.support_vector=self.get_best_params_for_svc(train_x,train_y)
            model_list1.append(self.support_vector)
            self.get_performance_parameters(train_x, train_y, test_x, test_y, 'SVC', self.support_vector, cluster_no)

            # create best model for Logistic Regression
            self.logistic_regr = self.get_best_params_for_logistic_reg(train_x, train_y)
            model_list1.append(self.logistic_regr)
            self.get_performance_parameters(train_x, train_y, test_x, test_y, 'LogisticRegression', self.logistic_regr, cluster_no)

            '''# create best model for KNN Classification
            self.knearest_neigh = self.get_best_params_for_knn(train_x, train_y)
            model_list1.append(self.knearest_neigh)
            self.get_performance_parameters(train_x, train_y, test_x, test_y, 'KNN', self.knearest_neigh, cluster_no)'''

            print('Best Param List: ',self.best_param_list)
            print('Model Names: ',self.model_list)
            print('Model Accuracy: ', self.model_acc)

            best_acc_score=max(self.model_acc)
            temp_idx=self.model_acc.index(best_acc_score)
            best_model_name=self.model_list[temp_idx]
            print(best_model_name,best_acc_score)

            return best_model_name,model_list1[temp_idx]

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occurred in get_best_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            print(e)
            raise Exception()