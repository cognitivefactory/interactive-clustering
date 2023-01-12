#!/usr/bin/env python3

###
### Prétraitement des données
###

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.manifold import TSNE
import pickle
import json

SIZE = 60

with open(f"data/vectors_{SIZE}.pkl", 'rb') as file:
    dict_of_vectors = pickle.load(file)

vectors = np.vstack([vec.toarray() for vec in dict_of_vectors.values()])

tsne = TSNE(
    n_components=3,
    # learning_rate="auto",  # Error on "scikit-learn==0.24.1" !
    init="random",
    random_state=42,
)

vectors_3D = tsne.fit_transform(vectors)

with open(f"data/results_size_{SIZE}.json") as file:
    dict_of_clustering = json.load(file)




###
### Application web
###

import dash
from dash import (
    Output, Input, State,
    html, dcc
)
from dash.exceptions import PreventUpdate
import plotly.express as px

app = dash.Dash(__name__, title="Interactive Visualization")

app.layout = html.Div(
    [
        html.H1("Interactive Visualisation"),
        dcc.Dropdown(
            id='algorithm',
            options=[
                {'label': "C-DBScan", 'value': 'c_dbscan'},
                {'label': "MPCKmeans", 'value': 'mpckmeans'},
                {'label': "Affinity Propagation", 'value': 'affinity_propagation'},
                {'label': "Kmeans", 'value': 'kmeans'},
            ]
        ),
        dcc.Dropdown(
            id='iteration',
            options=list(range(251)),
        ),
        dcc.Loading(
            dcc.Graph(
                id='the_plot',
                config={'displaylogo': False},
                style={'height': 'calc(100vh - 160px)'},
            ),
            type='cube'
        ),
    ]
)


@app.callback(
    Output('the_plot', 'figure'),
    Input('algorithm', 'value'),
    Input('iteration', 'value'),
)
def update_figure(algorithm, iteration):
    if not algorithm or iteration is None:
        raise PreventUpdate

    dict_of_predicted_clusters = dict_of_clustering[algorithm][str(iteration)]['predicted_clusters']

    # print(dict_of_predicted_clusters)

    figure = px.scatter_3d(vectors_3D, x=0, y=1, z=2,
                           color=[str(dict_of_predicted_clusters[i]) for i in dict_of_vectors],
                           hover_name=list(dict_of_vectors))

    return figure




if __name__ == '__main__':
    app.run(debug=True)
