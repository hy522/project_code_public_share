import random
import pickle
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import hmean
from scipy.spatial.distance import cdist
from scipy import stats
import numbers

class Agent(object):
    def __init__(self, agent_number, params={}):
        self.this_agent_number = agent_number  # index for this agent
        self.opponent_number = 1 - agent_number  # index for opponent
        self.n_items = params["n_items"]

        # Unpickle the trained model
        # Complications: pickle should work with any machine learning models
        # However, this does not work with custom defined classes, due to the way pickle operates
        # TODO you can replace this with your own model
        self.filename = 'machine_learning_model/logisticRegression'
        self.trained_model = pickle.load(open(self.filename, 'rb'))
        self.embedding = pd.read_csv('machine_learning_model/embedding')
        self.item0_embedding = self.openpickle('data/item0embedding')
        self.item1_embedding = self.openpickle('data/item1embedding')
        self.alpha = 0.8
        
    def _process_last_sale(self, last_sale, profit_each_team):
        # print("last_sale: ", last_sale)
        # print("profit_each_team: ", profit_each_team)
        my_current_profit = profit_each_team[self.this_agent_number]
        opponent_current_profit = profit_each_team[self.opponent_number]

        my_last_prices = last_sale[2][self.this_agent_number]
        opponent_last_prices = last_sale[2][self.opponent_number]

        did_customer_buy_from_me = last_sale[1] == self.this_agent_number
        did_customer_buy_from_opponent = last_sale[1] == self.opponent_number

        which_item_customer_bought = last_sale[0]

        # print("My current profit: ", my_current_profit)
        # print("Opponent current profit: ", opponent_current_profit)
        # print("My last prices: ", my_last_prices)
        # print("Opponent last prices: ", opponent_last_prices)
        # print("Did customer buy from me: ", did_customer_buy_from_me)
        # print("Did customer buy from opponent: ",
        #       did_customer_buy_from_opponent)
        # print("Which item customer bought: ", which_item_customer_bought)

        # TODO - add your code here to potentially update your pricing strategy based on what happened in the last round
        if did_customer_buy_from_opponent:  # can increase prices
            smaller = opponent_last_prices / my_last_prices
            alpha_others =  self.alpha * smaller
            alpha_ours =  self.alpha - 0.2
            self.alpha  = min(alpha_others, alpha_ours)

    # Given an observation which is #info for new buyer, information for last iteration, and current profit from each time
    # Covariates of the current buyer, and potentially embedding. Embedding may be None
    # Data from last iteration (which item customer purchased, who purchased from, prices for each agent for each item (2x2, where rows are agents and columns are items)))
    # Returns an action: a list of length n_items=2, indicating prices this agent is posting for each item.
    def action(self, obs):
        new_buyer_covariates, new_buyer_embedding, last_sale, profit_each_team = obs
        self._process_last_sale(last_sale, profit_each_team)
        # print(new_buyer_covariates)
        # print(new_buyer_embedding)
        # print(type(new_buyer_embedding))
        
        if new_buyer_embedding is None:
            new_buyer_embedding = [np.nan]*10
            # self.embedding.loc[len(self.embedding.index)] = new_buyer_covariates + new_buyer_embedding
            self.embedding.loc[len(self.embedding.index)] = np.concatenate((new_buyer_covariates,new_buyer_embedding), axis = 0)
            self.embedding[['0','1','2','3','4','5','6','7','8','9']] = self.knn_impute(target= self.embedding[['0','1','2','3','4','5','6','7','8','9']], attributes= self.embedding[['Covariate 1', 'Covariate 2','Covariate 3']],
                                    aggregation_method="median", k_neighbors=18, numeric_distance='euclidean',
                                    categorical_distance='hamming', missing_neighbors_threshold=0.9)
            em = list(self.embedding.loc[len(self.embedding.index)-1])
            self.embedding.drop([len( self.embedding.index)-1],inplace = True)
        else:
            # em = new_buyer_covariates + new_buyer_embedding
            em = np.concatenate((new_buyer_covariates,new_buyer_embedding), axis = 0)
        emdf = pd.DataFrame([em], columns=['Covariate 1', 'Covariate 2','Covariate 3']+list(range(10)))
        emdf['uv0'] = emdf[list(range(10))].dot(self.item0_embedding)
        emdf['uv1'] = emdf[list(range(10))].dot(self.item1_embedding)
        print("emdf:",emdf)
        p0,p1,r = self.predictPrice(self.trained_model, np.array(emdf)[0])
        print("price:",p0,p1)
        return [p0*self.alpha, p1*self.alpha]
        # TODO Currently this output is just a deterministic 2-d array, but the students are expected to use the buyer covariates to make a better prediction
        # and to use the history of prices from each team in order to create prices for each item.
    
    def openpickle(self, filename):
        with open(filename, "rb") as readfile:
            loaded = pickle.load(readfile)
        return loaded

    def predictPrice(self, model, original_array):
        #item0: largest price 2.22; least price 0
        #item1: largest price 4; least price 0
        max_price0 = 3
        min_price0 = 0
        max_price1 = 5
        min_price1 = 0
        step0 = (max_price0 - min_price0) / 10
        step1 = (max_price1 - min_price1) / 10
        revenue = 0
        pre_revenue = -1000
        while abs(revenue - pre_revenue) > 0.01:
            max_p0 = min_price0
            max_p1 = min_price1
            pre_revenue = revenue
            for p0 in np.arange(min_price0, max_price0, step0):
                for p1 in np.arange(min_price1, max_price1, step1):
                    arrayWithPrice = np.insert(original_array,-2,[p0,p1],axis=0)
                    arrayWithPrice = np.expand_dims(arrayWithPrice, axis=0)
                    print("probabilities:",model.predict_proba(arrayWithPrice)[0][1],model.predict_proba(arrayWithPrice)[0][2])
                    print("p0p1:",p0,p1)
                    
                    temp = model.predict_proba(arrayWithPrice)[0][1] * p0 + model.predict_proba(arrayWithPrice)[0][2] * p1
                    print("temp:",temp)
                    if temp > revenue:
                        max_p0 = p0
                        max_p1 = p1
                        revenue = temp


        max_price0 = max_p0 + step0
        min_price0 = max_p0
        max_price1 = max_p1 + step1
        min_price1 = max_p1
        step0 = (max_price0 - min_price0) / 10
        step1 = (max_price1 - min_price1) / 10
        return max_p0, max_p1, revenue
    
    def weighted_hamming(self, data):
        """ Compute weighted hamming distance on categorical variables. For one variable, it is equal to 1 if
            the values between point A and point B are different, else it is equal the relative frequency of the
            distribution of the value across the variable. For multiple variables, the harmonic mean is computed
            up to a constant factor.
            @params:
                - data = a pandas data frame of categorical variables
            @returns:
                - distance_matrix = a distance matrix with pairwise distance for all attributes
        """
        categories_dist = []

        for category in data:
            X = pd.get_dummies(data[category])
            X_mean = X * X.mean()
            X_dot = X_mean.dot(X.transpose())
            X_np = np.asarray(X_dot.replace(0,1,inplace=False))
            categories_dist.append(X_np)
        categories_dist = np.array(categories_dist)
        distances = hmean(categories_dist, axis=0)
        return distances


    def distance_matrix(self, data, numeric_distance = "euclidean", categorical_distance = "jaccard"):
        """ Compute the pairwise distance attribute by attribute in order to account for different variables type:
            - Continuous
            - Categorical
            For ordinal values, provide a numerical representation taking the order into account.
            Categorical variables are transformed into a set of binary ones.
            If both continuous and categorical distance are provided, a Gower-like distance is computed and the numeric
            variables are all normalized in the process.
            If there are missing values, the mean is computed for numerical attributes and the mode for categorical ones.

            Note: If weighted-hamming distance is chosen, the computation time increases a lot since it is not coded in C 
            like other distance metrics provided by scipy.
            @params:
                - data                  = pandas dataframe to compute distances on.
                - numeric_distances     = the metric to apply to continuous attributes.
                                          "euclidean" and "cityblock" available.
                                          Default = "euclidean"
                - categorical_distances = the metric to apply to binary attributes.
                                          "jaccard", "hamming", "weighted-hamming" and "euclidean"
                                          available. Default = "jaccard"
            @returns:
                - the distance matrix
        """
        possible_continuous_distances = ["euclidean", "cityblock"]
        possible_binary_distances = ["euclidean", "jaccard", "hamming", "weighted-hamming"]
        number_of_variables = data.shape[1]
        number_of_observations = data.shape[0]

        # Get the type of each attribute (Numeric or categorical)
        is_numeric = [all(isinstance(n, numbers.Number) for n in data.iloc[:, i]) for i, x in enumerate(data)]
        is_all_numeric = sum(is_numeric) == len(is_numeric)
        is_all_categorical = sum(is_numeric) == 0
        is_mixed_type = not is_all_categorical and not is_all_numeric

        # Check the content of the distances parameter
        if numeric_distance not in possible_continuous_distances:
            print ("The continuous distance " + numeric_distance + " is not supported.")
            return None
        elif categorical_distance not in possible_binary_distances:
            print ("The binary distance " + categorical_distance + " is not supported.")
            return None

        # Separate the data frame into categorical and numeric attributes and normalize numeric data
        if is_mixed_type:
            number_of_numeric_var = sum(is_numeric)
            number_of_categorical_var = number_of_variables - number_of_numeric_var
            data_numeric = data.iloc[:, is_numeric]
            data_numeric = (data_numeric - data_numeric.mean()) / (data_numeric.max() - data_numeric.min())
            data_categorical = data.iloc[:, [not x for x in is_numeric]]

        # Replace missing values with column mean for numeric values and mode for categorical ones. With the mode, it
        # triggers a warning: "SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame"
        # but the value are properly replaced
        if is_mixed_type:
            data_numeric.fillna(data_numeric.mean(), inplace=True)
            for x in data_categorical:
                data_categorical[x].fillna(data_categorical[x].mode()[0], inplace=True)
        elif is_all_numeric:
            data.fillna(data.mean(), inplace=True)
        else:
            for x in data:
                data[x].fillna(data[x].mode()[0], inplace=True)

        # "Dummifies" categorical variables in place
        if not is_all_numeric and not (categorical_distance == 'hamming' or categorical_distance == 'weighted-hamming'):
            if is_mixed_type:
                data_categorical = pd.get_dummies(data_categorical)
            else:
                data = pd.get_dummies(data)
        elif not is_all_numeric and categorical_distance == 'hamming':
            if is_mixed_type:
                data_categorical = pd.DataFrame([pd.factorize(data_categorical[x])[0] for x in data_categorical]).transpose()
            else:
                data = pd.DataFrame([pd.factorize(data[x])[0] for x in data]).transpose()

        if is_all_numeric:
            result_matrix = cdist(data, data, metric=numeric_distance)
        elif is_all_categorical:
            if categorical_distance == "weighted-hamming":
                result_matrix = self.weighted_hamming(data)
            else:
                result_matrix = cdist(data, data, metric=categorical_distance)
        else:
            result_numeric = cdist(data_numeric, data_numeric, metric=numeric_distance)
            if categorical_distance == "weighted-hamming":
                result_categorical = self.weighted_hamming(data_categorical)
            else:
                result_categorical = cdist(data_categorical, data_categorical, metric=categorical_distance)
            result_matrix = np.array([[1.0*(result_numeric[i, j] * number_of_numeric_var + result_categorical[i, j] *
                                   number_of_categorical_var) / number_of_variables for j in range(number_of_observations)] for i in range(number_of_observations)])

        # Fill the diagonal with NaN values
        np.fill_diagonal(result_matrix, np.nan)

        return pd.DataFrame(result_matrix)


    def knn_impute(self, target, attributes, k_neighbors, aggregation_method="mean", numeric_distance="euclidean",
                   categorical_distance="jaccard", missing_neighbors_threshold = 0.5):
        """ Replace the missing values within the target variable based on its k nearest neighbors identified with the
            attributes variables. If more than 50% of its neighbors are also missing values, the value is not modified and
            remains missing. If there is a problem in the parameters provided, returns None.
            If to many neighbors also have missing values, leave the missing value of interest unchanged.
            @params:
                - target                        = a vector of n values with missing values that you want to impute. The length has
                                                  to be at least n = 3.
                - attributes                    = a data frame of attributes with n rows to match the target variable
                - k_neighbors                   = the number of neighbors to look at to impute the missing values. It has to be a
                                                  value between 1 and n.
                - aggregation_method            = how to aggregate the values from the nearest neighbors (mean, median, mode)
                                                  Default = "mean"
                - numeric_distances             = the metric to apply to continuous attributes.
                                                  "euclidean" and "cityblock" available.
                                                  Default = "euclidean"
                - categorical_distances         = the metric to apply to binary attributes.
                                                  "jaccard", "hamming", "weighted-hamming" and "euclidean"
                                                  available. Default = "jaccard"
                - missing_neighbors_threshold   = minimum of neighbors among the k ones that are not also missing to infer
                                                  the correct value. Default = 0.5
            @returns:
                target_completed        = the vector of target values with missing value replaced. If there is a problem
                                          in the parameters, return None
        """

        # Get useful variables
        possible_aggregation_method = ["mean", "median", "mode"]
        number_observations = len(target)
        is_target_numeric = all(isinstance(n, numbers.Number) for n in target)

        # Check for possible errors
        # if number_observations < 3:
        #     print "Not enough observations."
        #     return None
        # if attributes.shape[0] != number_observations:
        #     print "The number of observations in the attributes variable is not matching the target variable length."
        #     return None
        # if k_neighbors > number_observations or k_neighbors < 1:
        #     print "The range of the number of neighbors is incorrect."
        #     return None
        # if aggregation_method not in possible_aggregation_method:
        #     print "The aggregation method is incorrect."
        #     return None
        # if not is_target_numeric and aggregation_method != "mode":
        #     print "The only method allowed for categorical target variable is the mode."
        #     return None

        # Make sure the data are in the right format
        target = pd.DataFrame(target)
        attributes = pd.DataFrame(attributes)

        # Get the distance matrix and check whether no error was triggered when computing it
        distances = self.distance_matrix(attributes, numeric_distance, categorical_distance)
        if distances is None:
            return None

        # Get the closest points and compute the correct aggregation method
        for i, value in enumerate(target.iloc[:, 0]):
            if pd.isnull(value):
                order = distances.iloc[i,:].values.argsort()[:k_neighbors]
                closest_to_target = target.iloc[order, :]
                missing_neighbors = [x for x  in closest_to_target.isnull().iloc[:, 0]]
                # Compute the right aggregation method if at least more than 50% of the closest neighbors are not missing
                if sum(missing_neighbors) >= missing_neighbors_threshold * k_neighbors:
                    continue
                elif aggregation_method == "mean":
                    target.iloc[i] = np.ma.mean(np.ma.masked_array(closest_to_target,np.isnan(closest_to_target)))
                elif aggregation_method == "median":
                    target.iloc[i] = np.ma.median(np.ma.masked_array(closest_to_target,np.isnan(closest_to_target)))
                else:
                    target.iloc[i] = stats.mode(closest_to_target, nan_policy='omit')[0][0]

        return target


