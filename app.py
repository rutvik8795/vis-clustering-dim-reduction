from flask import Flask, render_template, request, redirect, Response, jsonify

import random

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import scipy.spatial.distance

app = Flask(__name__)

@app.route('/')
def index():
    # The data frame below contains all the data and attributes regarding 5000 popular world-renowned movies
    movie_df = pd.read_csv('moviemetadata-imdb-5000-movie-dataset-QueryResult.csv')
    # These are the features selected for analysis and dimensionality reduction using PCA and MDS
    features = ['num_critic_for_reviews', 'duration', 'director_facebook_likes', 'actor_1_facebook_likes',
                'actor_2_facebook_likes', 'actor_3_facebook_likes', 'gross', 'num_voted_users',
                'cast_total_facebook_likes', 'facenumber_in_poster', 'num_user_for_reviews', 'budget',
                'imdb_score', 'movie_facebook_likes']
    movie_df = movie_df[features]

    # Drop all the null values in the columns
    movie_df = movie_df.dropna()

    # Declare the data dictionary to be sent to the HTML file where it will be displayed using D3.js
    d3_data = dict()

    # Random sample generation using random.sample() method
    random_samples = random_sample_generation(movie_df)

    # Calculate the sum of squared errors for all the 14 features
    sum_of_squared_errors = sum_of_squared_errors_generation(movie_df)

    # Display the elbow plot
    elbow_plot(sum_of_squared_errors)

    # Stratified sample generation
    stratified_samples = stratified_sample_generation(movie_df)
    stratified_samples = stratified_samples[features]

    # Original data PCA generation
    no_sampling_pca = pca_generation(movie_df[features])
    no_sampling_pca_top_three_attr_i = pca_top_three_attr(movie_df[features])
    no_sampling_pca_top_three_attr = movie_df.columns[no_sampling_pca_top_three_attr_i]
    no_sampling_pca_top_two_comps = pca_top_two_comps(movie_df[features])
    d3_data["no_sampling_scree_plot"] = [{"factor": i + 1, "eigenvalue": no_sampling_pca[i],
                                                   "cumulative_eigenvalue": np.cumsum(no_sampling_pca)[i]} for i in
                                                  range(14)]
    pca_no_sampling_projected_points = np.dot(stratified_samples, no_sampling_pca_top_two_comps.T)
    d3_data["pca_no_sampling_projected_points"] = [{"x": i[1], "y": i[0]} for i in
                                                   pca_no_sampling_projected_points.tolist()]

    # Random sampling PCA generation
    random_samples_pca = pca_generation(random_samples)
    random_samples_pca_top_three_attr_i = pca_top_three_attr(random_samples)
    random_samples_pca_top_three_attr = movie_df.columns[random_samples_pca_top_three_attr_i]
    random_samples_pca_top_two_comps = pca_top_two_comps(random_samples)
    d3_data["random_samples_scree_plot"] = [{"factor": i + 1, "eigenvalue": random_samples_pca[i],
         "cumulative_eigenvalue": np.cumsum(random_samples_pca)[i]} for i in range(14)]

    pca_random_sampling_projected_points = np.dot(stratified_samples, random_samples_pca_top_two_comps.T)
    d3_data["pca_random_sampling_projected_points"] = [{"x": i[1], "y": i[0]} for i in
                                                       pca_random_sampling_projected_points.tolist()]

    # Stratified sampling PCA generation
    stratified_samples_pca = pca_generation(stratified_samples)
    stratified_samples_pca_top_three_attr_i = pca_top_three_attr(stratified_samples)
    stratified_samples_pca_top_three_attr = movie_df.columns[stratified_samples_pca_top_three_attr_i]
    stratified_samples_pca_top_two_comps = pca_top_two_comps(stratified_samples)
    d3_data["stratified_samples_scree_plot"] = [{"factor": i + 1, "eigenvalue": stratified_samples_pca[i],
         "cumulative_eigenvalue": np.cumsum(stratified_samples_pca)[i]} for i in range(14)]

    pca_stratified_sampling_projected_points = np.dot(stratified_samples, stratified_samples_pca_top_two_comps.T)
    d3_data["pca_stratified_sampling_projected_points"] = [{"x": i[1], "y": i[0]}
                                                           for i in pca_stratified_sampling_projected_points.tolist()]


    # MDS calculation

    # Euclidian MDS for original samples
    euclidian_mds = MDS(dissimilarity='euclidean')
    movie_df_top_three = np.array(movie_df[no_sampling_pca_top_three_attr])
    euclidian_mds.fit(movie_df_top_three)

    euclidian_points = np.array(euclidian_mds.embedding_ * 10, dtype=int)
    d3_data["euclidian_points_no_sampling"] = [{"x": euclidian_point[0], "y": euclidian_point[1]}
                                   for euclidian_point in euclidian_points]

    # Correlation MDS for original samples
    correlation_mds = MDS(dissimilarity='precomputed')

    dissim_matrix = scipy.spatial.distance.cdist(movie_df_top_three, movie_df_top_three,
                                                        metric='correlation')
    np.fill_diagonal(dissim_matrix, np.zeros(len(dissim_matrix)))
    dissim_matrix = np.nan_to_num(dissim_matrix)
    correlation_mds.fit(dissim_matrix)

    correlation_points = np.array(correlation_mds.embedding_ * 10, dtype=int)
    d3_data["correlation_points_no_sampling"] = [{"x": correlation_point[0], "y": correlation_point[1]}
                                     for correlation_point in correlation_points]

    # Euclidian MDS for random samples
    euclidian_mds = MDS(dissimilarity='euclidean')
    movie_df_top_three = np.array(random_samples[random_samples_pca_top_three_attr_i])
    euclidian_mds.fit(movie_df_top_three)

    euclidian_points = np.array(euclidian_mds.embedding_ * 10, dtype=int)
    d3_data["euclidian_points_random_samples"] = [{"x": euclidian_point[0], "y": euclidian_point[1]}
                                   for euclidian_point in euclidian_points]

    # Correlation MDS for random samples
    correlation_mds = MDS(dissimilarity='precomputed')

    dissim_matrix = scipy.spatial.distance.cdist(movie_df_top_three, movie_df_top_three,
                                                        metric='correlation')
    np.fill_diagonal(dissim_matrix, np.zeros(len(dissim_matrix)))
    dissim_matrix = np.nan_to_num(dissim_matrix)
    correlation_mds.fit(dissim_matrix)

    correlation_points = np.array(correlation_mds.embedding_ * 10, dtype=int)
    d3_data["correlation_points_random_samples"] = [{"x": correlation_point[0], "y": correlation_point[1]}
                                     for correlation_point in correlation_points]

    # Euclidian MDS for stratified samples
    euclidian_mds = MDS(dissimilarity='euclidean')
    movie_df_top_three = np.array(stratified_samples[stratified_samples_pca_top_three_attr])
    euclidian_mds.fit(movie_df_top_three)

    euclidian_points = np.array(euclidian_mds.embedding_ * 10, dtype=int)
    d3_data["euclidian_points_stratified_samples"] = [{"x": euclidian_point[0], "y": euclidian_point[1]}
                                   for euclidian_point in euclidian_points]

    # Correlation MDS for stratified samples
    correlation_mds = MDS(dissimilarity='precomputed')

    dissim_matrix = scipy.spatial.distance.cdist(movie_df_top_three, movie_df_top_three,
                                                        metric='correlation')
    np.fill_diagonal(dissim_matrix, np.zeros(len(dissim_matrix)))
    dissim_matrix = np.nan_to_num(dissim_matrix)
    correlation_mds.fit(dissim_matrix)

    correlation_points = np.array(correlation_mds.embedding_ * 10, dtype=int)
    d3_data["correlation_points_stratified_samples"] = [{"x": correlation_point[0], "y": correlation_point[1]}
                                     for correlation_point in correlation_points]

    d3_data["scatterplot_matrix_data_no_sampling"] = np.array(movie_df[no_sampling_pca_top_three_attr]).tolist()
    d3_data["scatterplot_matrix_data_random_samples"] = np.array(random_samples[random_samples_pca_top_three_attr_i]).tolist()
    d3_data["scatterplot_matrix_data_stratified_samples"] = np.array(stratified_samples[stratified_samples_pca_top_three_attr]).tolist()

    d3_data = {'chart_data': d3_data}

    # Render the main index.html template and pass the data containing all the information
    return render_template('index.html', data=d3_data)


def pca_generation(data):
    pca = PCA()
    pca.fit(data)
    pca_results = pca.explained_variance_ratio_
    return pca_results


def pca_top_three_attr(data):
    pca = PCA()
    pca.fit(data)
    loadings = np.sum(np.square(pca.components_), axis=0)
    return loadings.argsort()[-3:][::-1]


def pca_top_two_comps(data):
    pca = PCA()
    pca.fit(data)
    return pca.components_[:2]


def random_sample_generation(movie_df):
    data_np = np.array(movie_df)
    return data_np[random.sample(range(len(data_np)), int(0.25 * len(data_np)))]


def sum_of_squared_errors_generation(movie_df):
    movie_df_np = np.array(movie_df)
    sum_of_squared_errors = []
    for i in range(2, 15):
        km = KMeans(n_clusters=i)
        km.fit(movie_df_np)
        sum_of_squared_errors.append(km.inertia_)

    return sum_of_squared_errors


def elbow_plot(sum_of_squared_errors):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(2, 15), sum_of_squared_errors)
    plt.grid(True)
    plt.title('SSE vs Number of clusters')
    plt.xlabel('Clusters')
    plt.ylabel('Sum of Squared Errors')
    plt.xticks(range(2, 15, 2))
    plt.show()


def stratified_sample_generation(movie_df):
    movie_df_np = np.array(movie_df)
    km = KMeans(n_clusters=4)
    km.fit(movie_df_np)
    stratified_samples = pd.DataFrame(columns=movie_df.columns)
    size_of_clusters = np.bincount(km.labels_)
    movie_df['Label'] = km.labels_

    for idx in range(4):
        clusters = movie_df[movie_df['Label'] == idx]
        stratified_samples = pd.concat(
            [stratified_samples,
             clusters.iloc[random.sample(range(size_of_clusters[idx]), int(size_of_clusters[idx] * 0.25))]])
    return stratified_samples


if __name__ == '__main__':
    app.run(debug=True)
