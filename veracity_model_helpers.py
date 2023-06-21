import pandas as pd
from scipy.optimize import minimize
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# example_df = pd.read_excel(r"C:\Users\tzach\OneDrive\מסמכים\MA\For demo\example_df.xlsx")
# example_df = example_df.fillna(0)
#
# model = VeracityModel(data=example_df, prediction_unit_column='prediction_unit')


class VeracityModel():
    '''
    Class used for learning to predict the veracity of reports based on their features
    and then use it to improve the reported-value estimations regarding prediction units
    '''

    def __init__(self, data: pd.DataFrame, prediction_unit_column: str = 'prediction_unit',
                 true_value_column: str = 'true_y', estimated_value_column: str = 'reported_y',
                 category_columns: [str] = None, processed_data=None, train_set=None, test_set=None, scaler=None):
        '''

        :param data:  A pandas dataframe. Expected to hold both train set and test set.
         Should have a column for prediction unit, a column for the true value, and a column for the reported value.
         Other columns are features.
        :param prediction_unit_column: The name of the prediction unit column
        :param true_value_column:  The name of the true value column
        :param estimated_value_column: The name of the estimated value column
        :param category_columns:  A list of column names that are categorical. If not passed, will be parsed automatically
        :param processed_data: The data after preproccessing. Will be used for learning if passed
        :param train_set: Train set. Same as "data" but only for train
        :param test_set:  Test set. Same as "data" but only for test
        :param scaler: A scaler object. Should have fit and transform methods (e.g sklearn's MinMax Scaler)
        '''
        self.raw_data = data
        self.prediction_unit_column = prediction_unit_column
        self.true_value_column = true_value_column
        self.estimated_value_column = estimated_value_column

        self.category_columns = category_columns

        if category_columns is None:  # Parsing categorical columns
            category_columns = self.get_category_columns()
            if len(category_columns) > 0:
                category_columns.remove(prediction_unit_column)
            self.category_columns = category_columns
            print(F'''As you didn't explicitly pass categorical columns, they were parsed automatically"
                  parsed columns are {self.category_columns}''')

        self.processed_data = processed_data
        self.train_set = train_set
        self.test_set = test_set

        self.feature_coefficients = None

        self.scaler = scaler
        if scaler is None:
            self.scaler = MinMaxScaler()

        self.test_mse_and_usefulness = None
        self.train_mse_and_usefulness = None

    def get_category_columns(self):
        df = self.raw_data
        cat_cols = []

        for column in df.columns:
            if df[column].dtype == 'object':
                cat_cols.append(column)
        return cat_cols

    def preprocess_data(self):
        '''
        Baisically does one hot encoding to categorical columns
        :return:
        '''
        df = self.raw_data
        res = df
        if self.category_columns is not None and len(self.category_columns) > 0:
            res = pd.get_dummies(df, columns=self.category_columns)
        self.processed_data = res

    def split_train_test(self, train_ratio=0.7, shuffle_units=True):
        '''
        Splits the data to two sets based on prediction_unit units so that samples from the same unit will be in the same set
        :param shuffle_units: If true, will shuffle the prediction units for the sets.
         Otherwise, will just take the first ones for train
        :param train_ratio: How many of the samples will be in the train set
        :return:
        '''

        df = self.processed_data
        pu_col = self.prediction_unit_column
        predcition_unit_sizes = df[pu_col].value_counts()
        n_train_samples = int(train_ratio * len(df))

        if shuffle_units:
            predcition_unit_sizes = predcition_unit_sizes.sample(frac=1)

        cumsum = predcition_unit_sizes.cumsum()

        first_test_pu_amount = cumsum[cumsum > n_train_samples].values[0]

        train_indices = predcition_unit_sizes.cumsum() <= first_test_pu_amount

        train_units = predcition_unit_sizes[train_indices].index
        test_units = predcition_unit_sizes[~train_indices].index

        self.train_set = df[df[pu_col].isin(train_units)]
        self.test_set = df[df[pu_col].isin(test_units)]

    def train_model_nelder_mead(self):
        '''
        Trains the model usning Nelder Mead optimization
        :return:
        '''
        print("Training model using Nelder Mead optimization")

        df = self.train_set
        if df is None:
            raise Exception("Must set train_set before training the model")

        true_y = df[self.true_value_column]
        estimated_y = df[self.estimated_value_column]

        # Features for training. Doesn't include prediction unit
        X = df.drop(columns=[self.true_value_column, self.estimated_value_column, self.prediction_unit_column])

        n_features = len(X.columns)

        # Create a vector of 0.5s for the optimization
        alphas_0 = np.ones(n_features) * 0.5

        alphas = minimize(self.likelihood_alphas, alphas_0,
                          (X, estimated_y, true_y, True),
                          method='Nelder-Mead')['x']

        self.feature_coefficients = alphas
        print("Done Training")

    def scale_features(self):
        '''
        Scales the features using self.scaler
        :return:
        '''

        X = self.train_set.drop(
            columns=[self.prediction_unit_column, self.true_value_column, self.estimated_value_column])
        scaled_train = self.scaler.fit_transform(X)
        self.train_set[X.columns] = scaled_train

        scaled_test = self.scaler.transform(self.test_set[X.columns])

        self.test_set[X.columns] = scaled_test

    def calc_model_predictions(self, df):
        '''
        Uses the optimized feature coefficients to give a prediction for every prediction unit
        based on the calculated weight of every sample in it. Also calculates the "naive mean" when all weights are the same.
        :param df: dataframe to do it on. For example self.test
        :return:
        '''

        alphas = self.feature_coefficients
        prediction_unit_col = self.prediction_unit_column
        true_y_col = self.true_value_column
        estimated_y_col = self.estimated_value_column

        X = df.drop(
            columns=[self.true_value_column, self.estimated_value_column, self.prediction_unit_column])

        samples_variance = self.generate_variances(X, alphas)

        df = df.assign(veracity=1 / samples_variance)

        total_weights_per_pu = df[[prediction_unit_col, 'veracity']].groupby(prediction_unit_col).sum()
        total_weights_per_pu = total_weights_per_pu.rename(columns={'veracity': 'weights_sum'})

        df = df.merge(total_weights_per_pu, on=prediction_unit_col)
        df['sample_contribution'] = df['veracity'] * df[self.estimated_value_column] / df['weights_sum']

        means_df = df[[prediction_unit_col, true_y_col, estimated_y_col]].groupby(prediction_unit_col).mean()
        model_pred = df[[prediction_unit_col, 'sample_contribution']].groupby(prediction_unit_col).sum()
        results_df = pd.DataFrame({"true_mean": means_df[true_y_col], "naive_mean": means_df[estimated_y_col],
                                   'model_mean': model_pred['sample_contribution']})
        return results_df

    def generate_variances(self, X, alphas, noise=None):
        '''
        Uses the provided coefficients to get the variance of each sample
        :param X:
        :param alphas:
        :param noise:
        :return:
        '''
        if noise is None:
            noise = np.zeros(X.shape[0])
        else:
            assert (len(noise) == X.shape[0])
        return np.exp(X @ alphas + noise)

    def likelihood_alphas(self, alphas: np.array, features: np.array, reported_y, true_y, return_minus=True):
        '''
        Calculates the likelihood of seeing the distances in the data between reoported and true values.
        :param true_y: A vector/Series of the true values
        :param reported_y: A vector/Series of the reported values
        :param alphas:np.array Weight of every feature, dimension d
        :param features: features of every local sample, dimensions n,d
        :return:
        '''
        distances = np.square(true_y - reported_y)  # Distance between reported y and true y
        variances = self.generate_variances(features, alphas)  # calculates the variance of each sample based on weights

        res = -0.5 * (
            sum(distances / variances + np.log(variances)))  # The likelihood of seeing the distances given the weights

        if return_minus:  # might be necessary for optimization with Nelder Mead
            res = -res
        return res

    def get_mse_and_usefulness(self, df):
        '''
        Caclulates the model MSE and also the naive MSE (when all samples have the same weight in a prediction unit)
        And usefulness, which  is a count of the prediction units the model was better for
        :param df:
        :return:
        '''
        predictions = self.calc_model_predictions(df)
        naive_mse = mean_squared_error(predictions['true_mean'], predictions['naive_mean'])
        model_mse = mean_squared_error(predictions['true_mean'], predictions['model_mean'])
        useful = predictions.apply(
            lambda row: abs(row['true_mean'] - row['model_mean']) < abs(row['true_mean'] - row['naive_mean']), axis=1)

        usefulness = sum(useful)/len(useful)

        return {"naive_mse": naive_mse, "model_mse": model_mse,"usefulness":usefulness}

    def train_predict_flow(self, train_ratio: float = 0.7, shuffle_prediction_units: bool = True, scale_features=True):
        '''
        Runs the standard flow of preprocessing, training and predicting. 
        :param train_ratio:
        :param shuffle_prediction_units:
        :param scale_features:
        :return:
        '''

        print("Preprocessing Data")
        self.preprocess_data()

        print(
            f"Splitting train and test set. Train ratio = {train_ratio}, shuffle prediction units = {shuffle_prediction_units}")
        self.split_train_test(train_ratio=train_ratio, shuffle_units=shuffle_prediction_units)

        if scale_features:
            print("Scaling features")
            self.scale_features()
        else:
            print("Not scaling features")

        self.train_model_nelder_mead()

        self.train_mse_and_usefulness = self.get_mse_and_usefulness(self.train_set)
        print(f"Train stats {self.train_mse_and_usefulness}")

        self.test_mse_and_usefulness = self.get_mse_and_usefulness(self.test_set)
        print(f"Test stats {self.test_mse_and_usefulness}")
