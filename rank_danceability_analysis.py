# Imports
import pandas as pd
import numpy as np
import matplotlib
import spotipy
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import pycountry_convert as pc
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Functions
def read_file_into_df(file_path):
    df = pd.read_csv(file_path)
    return df

def drop_coloumn_from_df(df, list_of_column_to_remove):
    df = df.drop(list_of_column_to_remove, axis = 1)
    return df

def drop_rows_from_df(df, firstIndex, secondIndex):
    df = df.drop(df.index[firstIndex:secondIndex])
    return df

def drop_dupicates_from_df(df):
    return df.drop_duplicates()

def convert_charts_names(df):
    df['chart'] = df['chart'].map({'top200': 0, 'viral50': 1}).astype(int)
    return df

def country_to_continent(country_name):
    country_alpha2 = pc.country_name_to_country_alpha2(country_name)
    country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
    country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
    return country_continent_name


def calc_mean_of_feature_for_continent(df, column_name, continent_name):
    mean_of_feature_for_continent = df.loc[df["region"] == continent_name, column_name].mean()
    return mean_of_feature_for_continent

def continent_and_feature_mean(df, feature, continent):
    return (feature, round(calc_mean_of_feature_for_continent(df, feature, continent), 3))

def seperate_df_by_continent(df, continent):
    grouped_continent = df.groupby(df.region)
    df_to_ret = grouped_continent.get_group(continent)
    return df_to_ret

def is_above_avg(df, feature, mean_of_feature):
    df[feature + "_above_avg"] = np.where(df[feature] >= mean_of_feature, 1, 0)
    return df

def is_in_top_of_chart(df):
    df["rank"] = np.where(df["rank"] < 21, 1, 0)
    return df

def logistic_regression(df, list_of_features_to_check, continent):
    x = df[list_of_features_to_check]
    y = df['rank']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    logreg = pipeline.make_pipeline(preprocessing.StandardScaler(), linear_model.SGDClassifier(loss = 'log_loss', alpha = 0, learning_rate = 'constant', eta0 = 0.01))
    logreg.fit(x_train, y_train)
    result = logreg.score(x_test, y_test)
    print('Accuracy score for ' + continent + ": " + str(round(logreg.score(x_test, y_test), 5)))

def decision_tree(df, list_of_features_to_check):
    x = df[list_of_features_to_check]
    y = df['rank']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    clf = DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    df['prediction'] = clf.predict(x)
    accuracy = accuracy_score(y_test, y_pred)
    tree.plot_tree(clf)
    return df, accuracy

def knn(df, predictors, target, n_neighbors=100):
    X_train, X_test, y_train, y_test = train_test_split(df[predictors], df[target], test_size=0.2, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn.score(X_test, y_test)

# Reading the files into dataframes
# Insert the datasets paths
path1 = r""
path2 = r""   
charts = read_file_into_df(path1)
tracks_dataset = read_file_into_df(path2)

# Removing unnecessary columns
tracks_dataset = drop_coloumn_from_df(tracks_dataset, ["artists", "explicit", "duration_ms", "Unnamed: 0", "album_name", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "time_signature", "track_genre"])
charts = drop_coloumn_from_df(charts, ["artist", "url", "streams"])

# Keeping all the values from the chart "top200"
charts = convert_charts_names(charts)
charts = charts[charts.chart == 0]
charts = drop_coloumn_from_df(charts, ["chart"])

# Converting countries to continents
charts = charts[charts.region != "Global"]
charts["region"] = charts["region"].apply(country_to_continent)

# Converting trends to numerical
# charts = convert_trends_to_numerical(charts)

# Droping duplicates from dataframes
charts = drop_dupicates_from_df(charts)
tracks_dataset = drop_dupicates_from_df(tracks_dataset)

# Droping some rows from the dataset
charts = drop_rows_from_df(charts, 1000001, 26173515)

# Joining the datasets based on their shared track names
df_shared = pd.merge(charts, tracks_dataset, how='inner', left_on = 'title', right_on = 'track_name')
df_shared = drop_coloumn_from_df(df_shared, ["track_id", "track_name", "trend"])

# Calculating the mean of each feature by continent
# Europe:
europe_avg_popularity = continent_and_feature_mean(df_shared, "popularity", "Europe")
europe_avg_danceability = continent_and_feature_mean(df_shared, "danceability", "Europe")
europe_avg_energy = continent_and_feature_mean(df_shared, "energy", "Europe")
europe_avg_valance = continent_and_feature_mean(df_shared, "valence", "Europe")
europe_avg_tempo = continent_and_feature_mean(df_shared, "tempo", "Europe")
europe_avg_features_list = [europe_avg_popularity, europe_avg_danceability, europe_avg_energy, europe_avg_valance, europe_avg_tempo]
# South America:
sa_avg_popularity = continent_and_feature_mean(df_shared, "popularity", "South America")
sa_avg_danceability = continent_and_feature_mean(df_shared, "danceability", "South America")
sa_avg_energy = continent_and_feature_mean(df_shared, "energy", "South America")
sa_avg_valance = continent_and_feature_mean(df_shared, "valence", "South America")
sa_avg_tempo = continent_and_feature_mean(df_shared, "tempo", "South America")
sa_avg_features_list = [sa_avg_popularity, sa_avg_danceability, sa_avg_energy, sa_avg_valance, sa_avg_tempo]
# North America:
na_avg_popularity = continent_and_feature_mean(df_shared, "popularity", "North America")
na_avg_danceability = continent_and_feature_mean(df_shared, "danceability", "North America")
na_avg_energy = continent_and_feature_mean(df_shared, "energy", "North America")
na_avg_valance = continent_and_feature_mean(df_shared, "valence", "North America")
na_avg_tempo = continent_and_feature_mean(df_shared, "tempo", "North America")
na_avg_features_list = [na_avg_popularity, na_avg_danceability, na_avg_energy, na_avg_valance, na_avg_tempo]
# Africa:
africa_avg_popularity = continent_and_feature_mean(df_shared, "popularity", "Africa")
africa_avg_danceability = continent_and_feature_mean(df_shared, "danceability", "Africa")
africa_avg_energy = continent_and_feature_mean(df_shared, "energy", "Africa")
africa_avg_valance = continent_and_feature_mean(df_shared, "valence", "Africa")
africa_avg_tempo = continent_and_feature_mean(df_shared, "tempo", "Africa")
africa_avg_features_list = [africa_avg_popularity, africa_avg_danceability, africa_avg_energy, africa_avg_valance, africa_avg_tempo]
# Asia:
asia_avg_popularity = continent_and_feature_mean(df_shared, "popularity", "Asia")
asia_avg_danceability = continent_and_feature_mean(df_shared, "danceability", "Asia")
asia_avg_energy = continent_and_feature_mean(df_shared, "energy", "Asia")
asia_avg_valance = continent_and_feature_mean(df_shared, "valence", "Asia")
asia_avg_tempo = continent_and_feature_mean(df_shared, "tempo", "Asia")
asia_avg_features_list = [asia_avg_popularity, asia_avg_danceability, asia_avg_energy, asia_avg_valance, asia_avg_tempo]
# Oceania:
oc_avg_popularity = continent_and_feature_mean(df_shared, "popularity", "Oceania")
oc_avg_danceability = continent_and_feature_mean(df_shared, "danceability", "Oceania")
oc_avg_energy = continent_and_feature_mean(df_shared, "energy", "Oceania")
oc_avg_valance = continent_and_feature_mean(df_shared, "valence", "Oceania")
oc_avg_tempo = continent_and_feature_mean(df_shared, "tempo", "Oceania")
oc_avg_features_list = [oc_avg_popularity, oc_avg_danceability, oc_avg_energy, oc_avg_valance, oc_avg_tempo]

# Seperating the DF to df per continent:
df_shared_europe = seperate_df_by_continent(df_shared, "Europe")
df_shared_sa = seperate_df_by_continent(df_shared, "South America")
df_shared_na = seperate_df_by_continent(df_shared, "North America")
df_shared_africa = seperate_df_by_continent(df_shared, "Africa")
df_shared_asia = seperate_df_by_continent(df_shared, "Asia")
df_shared_oc = seperate_df_by_continent(df_shared, "Oceania")

# Check for each row if it's features are above avg (# 1 - above avg, 0 - below avg), and droping the feature:
# Europe:
for feature in europe_avg_features_list:
    df_shared_europe = is_above_avg(df_shared_europe, feature[0], feature[1])
df_shared_europe = df_shared_europe.rename(columns={"popularity_above_avg" : "popularity", "danceability_above_avg" : "danceability", "energy_above_avg" : "energy", "valence_above_avg" : "valence", "tempo_above_avg" : "tempo"})
df_shared_europe = drop_coloumn_from_df(df_shared_europe, ["date", "title", "region"])
df_shared_europe = is_in_top_of_chart(df_shared_europe)
# South America:
for feature in sa_avg_features_list:
    df_shared_sa = is_above_avg(df_shared_sa, feature[0], feature[1])
df_shared_sa = df_shared_sa.rename(columns={"popularity_above_avg" : "popularity", "danceability_above_avg" : "danceability", "energy_above_avg" : "energy", "valence_above_avg" : "valence", "tempo_above_avg" : "tempo"})
df_shared_sa = drop_coloumn_from_df(df_shared_sa, ["date", "title", "region"])
df_shared_sa = is_in_top_of_chart(df_shared_sa)
# North America:
for feature in na_avg_features_list:
    df_shared_na = is_above_avg(df_shared_na, feature[0], feature[1])
df_shared_na = df_shared_na.rename(columns={"popularity_above_avg" : "popularity", "danceability_above_avg" : "danceability", "energy_above_avg" : "energy", "valence_above_avg" : "valence", "tempo_above_avg" : "tempo"})
df_shared_na = drop_coloumn_from_df(df_shared_na, ["date", "title", "region"])
df_shared_na = is_in_top_of_chart(df_shared_na)
# Africa:
for feature in africa_avg_features_list:
    df_shared_africa = is_above_avg(df_shared_africa, feature[0], feature[1])
df_shared_africa = df_shared_africa.rename(columns={"popularity_above_avg" : "popularity", "danceability_above_avg" : "danceability", "energy_above_avg" : "energy", "valence_above_avg" : "valence", "tempo_above_avg" : "tempo"})
df_shared_africa = drop_coloumn_from_df(df_shared_africa, ["date", "title", "region"])
df_shared_africa = is_in_top_of_chart(df_shared_africa)
# Asia:
for feature in asia_avg_features_list:
    df_shared_asia = is_above_avg(df_shared_asia, feature[0], feature[1])
df_shared_asia = df_shared_asia.rename(columns={"popularity_above_avg" : "popularity", "danceability_above_avg" : "danceability", "energy_above_avg" : "energy", "valence_above_avg" : "valence", "tempo_above_avg" : "tempo"})
df_shared_asia = drop_coloumn_from_df(df_shared_asia, ["date", "title", "region"])
df_shared_asia = is_in_top_of_chart(df_shared_asia)
# Oceania:
for feature in oc_avg_features_list:
    df_shared_oc = is_above_avg(df_shared_oc, feature[0], feature[1])
df_shared_oc = df_shared_oc.rename(columns={"popularity_above_avg" : "popularity", "danceability_above_avg" : "danceability", "energy_above_avg" : "energy", "valence_above_avg" : "valence", "tempo_above_avg" : "tempo"})
df_shared_oc = drop_coloumn_from_df(df_shared_oc, ["date", "title", "region"])
df_shared_oc = is_in_top_of_chart(df_shared_oc)

# Activating logistic regression function on each df for each continent:
logistic_regression(df_shared_europe, ["danceability"], "Europe")
logistic_regression(df_shared_na, ["danceability"], "North America")
logistic_regression(df_shared_sa, ["danceability"], "South America")
logistic_regression(df_shared_asia, ["danceability"], "Asia")
logistic_regression(df_shared_africa, ["danceability"], "Africa")
logistic_regression(df_shared_oc, ["danceability"], "Oceania")

# Activating decision tree algorithm:
df_shared_after_top_of_chart = is_in_top_of_chart(df_shared)
df_after_tree, tree_result = decision_tree(df_shared_after_top_of_chart, ["danceability"])
print("Accuracy result for Decision Tree algorithm: " + str(round(tree_result, 5)))

# Activating KNN algorithm:
knn_result = knn(df_shared, ["danceability"], "rank")
print("Accuracy result for KNN algorithm: " + str(round(knn_result, 5)))