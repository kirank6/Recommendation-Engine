# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 11:59:07 2021

@author: Kiran Khanal
"""
'''
This program reads a recipe dataframe with source, target and
correspnding weight and convert it into graph.
Implements Node2vec algorithm and returns a dataframe
of nodes with corresponding node embedding vectors.
'''


# Import packages
import networkx as nx
import pandas as pd
import numpy as np

import io, os, sys, types
from tensorflow import keras
from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk
from stellargraph.data import UnsupervisedSampler

from stellargraph.mapper import Node2VecLinkGenerator, Node2VecNodeGenerator
from stellargraph.layer import Node2Vec, link_classification


# Import recipe data
df = pd.read_csv('All_recipe.csv')
# Dataframe consits of 

# Convert the dataframe into Network Graph
GR = nx.from_pandas_edgelist(df,'Source', 'Target',["Weight"])

# Convert the Graph into a StellarGraph 

GS = StellarGraph.from_networkx(GR)
#print(GS.info())

# Unique nodes in the Graph
subjects = GS.nodes()
#print(subjects)

# The Node2Vec algorithm
# The number of walks to take per node, the length of each walk
walk_number = 50  # maximum length of a random walk (default: 80)
walk_length = 5   # number of random walk per node  (default: 10)

# Create the biased random walker to perform context node sampling, 
# with the specified parameters. It generates a string value with node 
# index which will be used for embedding.

walker = BiasedRandomWalk(
    GS,
    n=walk_number,
    length=walk_length,
    p=0.5,  # defines probability, 1/p, of returning to source node (return hyperparameter) (default: 1)
    q=2.0,  # defines probability, 1/q, for moving to a node away from the source node (inout hyperparameter)  (default: 1)
)

# Create the UnsupervisedSampler instance with the biased random walker.
unsupervised_samples = UnsupervisedSampler(GS, nodes=list(GS.nodes()), walker=walker)

# Set the batch size and the number of epochs
batch_size = 50
epochs = 2

# Define an attri2vec training generator, which generates 
# a batch of (index of target node, index of context node, label of node pair) pairs per iteration
generator = Node2VecLinkGenerator(GS, batch_size)

# Build the Node2Vec model, with the dimension of learned node representations set to 128
emb_size = 12 # Embedding dimensions (default: 128)
node2vec = Node2Vec(emb_size, generator=generator)

x_inp, x_out = node2vec.in_out_tensors()

# Use the link_classification function to generate the prediction,
# with the 'dot' edge embedding generation method and the 'sigmoid' activation, 
# which actually performs the dot product of the input embedding of the 
#target node and the output embedding of the context node 
#followed by a sigmoid activation.

prediction = link_classification(
   output_dim=1, output_act="sigmoid", edge_embedding_method="dot"
)(x_out)

# Stack the Node2Vec encoder and prediction layer into a Keras model. 
# Our generator will produce batches of positive and negative context pairs 
#as inputs to the model. Minimizing the binary crossentropy between the 
#outputs and the provided ground truth is much like a regular
# binary classification task.

model = keras.Model(inputs=x_inp, outputs=prediction)
model.compile(
     optimizer=keras.optimizers.Adam(lr=1e-3),
     loss=keras.losses.binary_crossentropy,
     metrics=[keras.metrics.binary_accuracy],
)

 # Train the model.
history = model.fit(
    generator.flow(unsupervised_samples),
    epochs=epochs,
    verbose=1,
    use_multiprocessing=False,
    workers=4,
    shuffle=True,
)

# # Build the node based model for predicting node representations from node 
# # ids and the learned parameters. Below a Keras model is constructed, with 
# #x_inp[0] as input and x_out[0] as output. Note that this model's weights
# # are the same as those of the corresponding node encoder in the 
# #previously trained node pair classifier.

# x_inp_src = x_inp[0]
# x_out_src = x_out[0]
# embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)

# # Get the node embeddings from node ids.
# node_gen = Node2VecNodeGenerator(GS, batch_size).flow(subjects)
# node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=1)

# # Make dataframe of nodes and corresponding embedded vector
# out_df = pd.DataFrame()
# out_df['Nodes'] = pd.Series(subjects)
# out_df['emb_vect'] = [list(i) for i in node_embeddings]

# # Save it as csv file
# out_df.to_csv('embedded_vetors.csv')

