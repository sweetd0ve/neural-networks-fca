import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import gc
import tensorflow as tf
import keras
import kerastuner
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Dropout, Input, Embedding,Reshape, Concatenate
from kerastuner.tuners import RandomSearch
from tensorflow.keras.optimizers import SGD
from tensorflow import keras
from tensorflow.keras import layers


import os
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


import warnings
warnings.filterwarnings("ignore")


project_path = os.path.abspath(os.pardir)
data_path = os.path.join(project_path, 'data/')
models_path = os.path.join(project_path, 'data/models_files')


# for binary dataset
class BaseClassifiers:
    def __init__(self, x_train, x_test, y_train, y_test, dataset_name=''):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.dataset_name = dataset_name

    def kNeighboursClassifier(self):
        """
        :return: accuracy, f1_score, classification_report by kNN classifier on x_test dataset
        """
        knn = KNeighborsClassifier()
        k_range = list(range(1, 31))
        param_grid = dict(n_neighbors=k_range)

        # defining parameter range
        grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', return_train_score=False, verbose=1)

        # fitting the model for grid search
        grid_search = grid.fit(self.x_train, self.y_train)
        logging.info(f'{self.dataset_name}\nkNN best params: {grid.best_params_}')

        preds_knn = grid_search.predict(self.x_test)
        res_knn = self._evaluate_model(preds_knn)
        return res_knn

    def naiveBayes(self):
        """
        :return: accuracy, f1_score, classification_report by Gaussian Naive Bayes on x_test dataset
        """
        gaussian = GaussianNB()
        gaussian.fit(self.x_train, self.y_train)
        preds_nb = gaussian.predict(self.x_test).astype(int)
        # accuracy on training data
        accuracy = round(gaussian.score(self.x_train, self.y_train) * 100, 2)

        res_nb = self._evaluate_model(preds_nb)
        return res_nb

    def decisionTree(self):
        """
        :return: accuracy, f1_score by Decision Tree Classifier on x_test dataset
        """
        decision_tree = DecisionTreeClassifier(random_state=12)
        decision_tree.fit(self.x_train, self.y_train)
        preds_dt = decision_tree.predict(self.x_test).astype(int)
        # accuracy on training data
        accuracy = round(decision_tree.score(self.x_train, self.y_train) * 100, 2)

        res_dt = self._evaluate_model(preds_dt)
        return res_dt

    def randomForest(self):
        """
        :return: accuracy, f1_score, classification_report by RandomForest classifier on x_test dataset
        """
        rf = RandomForestClassifier(n_estimators=30, random_state=12, max_depth=10)
        rf.fit(self.x_train, self.y_train)
        preds_rf = rf.predict(self.x_test)

        res_rf = self._evaluate_model(preds_rf)
        return res_rf

    def xgbclassifier(self):
        """
        :return: results on testing dataset by tuned XGBoost classifier
        """
        params = {
            "learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
            "max_depth": [3, 4, 5, 6, 8, 10, 12, 15],
            "min_child_weight": [1, 3, 5, 7],
            "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
            "colsample_bytree": [0.3, 0.4, 0.5, 0.7]
        }
        classifier = xgb.XGBClassifier()
        random_search = RandomizedSearchCV(classifier, param_distributions=params, n_iter=10,
                                           scoring='accuracy', n_jobs=-1, cv=10, verbose=3)
        random_search.fit(self.x_train, self.y_train)
        best_params = random_search.best_params_
        logging.info(f'{self.dataset_name}\nXGBClassifier best params: {best_params}')


        classifier = xgb.XGBClassifier(
            base_score=0.5, booster='gbtree', colsample_bylevel=1,
            colsample_bynode=1, colsample_bytree=best_params['colsample_bytree'],
            gamma=best_params['gamma'], gpu_id=0,
            importance_type='gain', interaction_constraints='',
            learning_rate=0.1, max_delta_step=0, max_depth=best_params['max_depth'],
            min_child_weight=best_params['min_child_weight'],
            n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=0,
            reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
            tree_method='exact', validate_parameters=1, verbosity=2)

        score = cross_val_score(classifier, self.x_train, self.y_train, cv=10)
        score.mean()
        classifier.fit(self.x_train, self.y_train)
        joblib.dump(classifier, models_path + f'/xgboost_{self.dataset_name}.pkl')
        preds_xgb = classifier.predict(self.x_test)

        res_xgb = self._evaluate_model(preds_xgb)
        return res_xgb

    def mlp(self):
        """
        :return: results on testing dataset by tuned Multilayer Perceptron
        """
        # for binary classification
        train_x, valid_x, train_y, valid_y = train_test_split(self.x_train, self.y_train, test_size=0.2)
        input_dim_nn = train_x.shape[1]
        def build_model(hp):
            model = keras.Sequential()
            counter = 0
            for i in range(hp.Int('num_layers', min_value=1, max_value=10)):
                if counter == 0:
                    model.add(layers.Dense(hp.Int('units_' + str(i), min_value=16, max_value=256, step=16),
                                           activation='relu', kernel_initializer='he_uniform',
                                           input_dim=input_dim_nn))
                    model.add(Dropout(hp.Choice('dropout' + str(i), values=[0.1, 0.2, 0.3, 0.4, 0.5])))
                else:
                    model.add(layers.Dense(units=hp.Int('units_' + str(i), min_value=16, max_value=256, step=16),
                                           activation='relu', kernel_initializer='he_uniform'))
                    model.add(Dropout(hp.Choice('dropout' + str(i), values=[0.1, 0.2, 0.3, 0.4, 0.5])))
                counter += 1
            model.add(layers.Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform'))
            model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                          # case for binary classification
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
            return model

        tuner = RandomSearch(
            build_model,
            objective=kerastuner.Objective('accuracy', direction="max"),
            seed=42,
            max_trials=5,
            executions_per_trial=3,
            directory=models_path,
            project_name='kerastuner_params')

        tuner.search(train_x, train_y, epochs=100, batch_size=128, validation_data=(valid_x, valid_y))
        model = tuner.get_best_models(num_models=1)[0]
        model.fit(train_x, train_y, epochs=200, initial_epoch=20, validation_data=(valid_x, valid_y))

        gc.collect()

        logging.info(f'{self.dataset_name}\n MLP model summary:\n {model.summary()}')
        model.evaluate(self.x_test, self.y_test)
        preds_mlp = model.predict(self.x_test)
        preds_mlp[preds_mlp <= 0.5] = 0
        preds_mlp[preds_mlp > 0.5] = 1

        res_mlp = self._evaluate_model(preds_mlp)
        return res_mlp

    def _evaluate_model(self, preds):
        """
        :param preds: predictions by classifier
        :return: accuracy, f1_score and the values of true positive
                 and false negative rate
        """
        acc = accuracy_score(self.y_test, preds)
        f1 = f1_score(self.y_test, preds)
        report = classification_report(self.y_test, preds)
        logging.info(f'{self.dataset_name} classification report:\n {report}')

        # for binary classification
        roc_auc = roc_auc_score(self.y_test, preds)

        # for the case when we have binary classification
        fpr, tpr, _ = roc_curve(self.y_test, preds)
        return acc, f1, fpr, tpr, roc_auc

    def compare_models(self) -> pd.DataFrame:
        """
        Compares the perfomance of the state-of-the-art models on the testing data
        :return: pd.DataFrame
        """
        models = ['K-Nearest Neighbours', 'Gaussian Naive Bayes', 'Decision Tree Classifier',
                  'Random Forest', 'XGBoost Classifier', 'Multilayer Perceptron (NN)']
        data = [self.kNeighboursClassifier(),
                self.naiveBayes(),
                self.decisionTree(),
                self.randomForest(),
                self.xgbclassifier(),
                self.mlp()]
        models_df = pd.DataFrame(data, columns=['accuracy', 'f1_score', 'fpr', 'tpr', 'roc_auc'])
        models_df['Model'] = models
        models_df.sort_values(by=['accuracy'])
        models_df.to_csv(data_path + f'/{self.dataset_name}_res.csv')
        print(models_df)
        return models_df

    @staticmethod
    def plot_roc_auc(df: pd.DataFrame):
        pass


if __name__ == '__main__':
    train = pd.read_csv(os.path.join(data_path, 'heart_disease_train.csv'))
    test = pd.read_csv(os.path.join(data_path, 'heart_disease_train.csv'))
    data = pd.concat([train, test])
    train_ids = train.index
    test_ids = test.index

    features_to_be_scaled = [c for c in data.columns if len(data[c].unique()) < 10]
    features_to_be_scaled.remove('target')
    df = pd.get_dummies(data, columns=features_to_be_scaled)
    std = StandardScaler()
    df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']] = std.fit_transform(
        df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']])
    df_train = df.iloc[train_ids]
    df_test = df.iloc[test_ids]

    X_train = df_train.drop(columns=['target'], inplace=False)
    X_test = df_test.drop(columns=['target'], inplace=False)
    y_train = df_train.target
    y_test = df_test.target

    correlation_new = df.corr()
    print(correlation_new)
    print(correlation_new['target'].sort_values(ascending=False))

    models = BaseClassifiers(X_train, X_test, y_train, y_test, 'HeartDisease')
    models.compare_models()
