#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import nltk
import networkx as nx
import matplotlib.pyplot as plt
import networkx
from networkx.drawing.nx_agraph import graphviz_layout
import csv
import collections
from networkx.algorithms.community import k_clique_communities

def extractEntities(ne_chunked):
    data = {}
    for entity in ne_chunked:
        if isinstance(entity, nltk.tree.Tree):
            text = " ".join([word for word, tag in entity.leaves()])
            ent = entity.label()
            data[text] = ent
        else:
            continue
    return data


# Extracting input data
moviedict = {}
with open('casts.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if row:
            items = row[0].split(";")
            if(len(items) > 4):
                actortype = items[3].strip('"')
                if actortype != "Grp":
                    movie = items[0].strip('"')
                    actor = items[2].strip('"')
                    #Checking if the actor name is a named entity
                    tokens = nltk.word_tokenize(actor)
                    tagged = nltk.pos_tag(tokens)
                    ne_chunked = nltk.ne_chunk(tagged, binary=True)
                    entities = extractEntities(ne_chunked)
                    if entities:
                        if movie in moviedict:
                            moviedict[movie].add(actor)
                        else:
                            moviedict[movie] = set([actor])

# Process data and convert into an undirected graph
G=nx.Graph()
for movie in moviedict:
    prevactors = set()
    for actor in moviedict[movie]:
        G.add_node(actor)
        for prevactor in prevactors:
            G.add_edge(prevactor, actor)
        prevactors.add(actor)

print("Number of nodes: " + str(len(G)))
print("Number of edges: " + str(G.number_of_edges()))
print("Density: " + str(2*G.number_of_edges()/(len(G)*(len(G)-1))))
print()

# Export as Gephi graph
nx.write_gexf(G, "export.gexf")

# Centralities
degr_centrality = networkx.degree_centrality(G)
sorted_degr_centrality = [(k, degr_centrality[k]) for k in sorted(degr_centrality, key=degr_centrality.get, reverse=True)]
idx = 0
print("Best centralities by degree centrality:")
for key, value in sorted_degr_centrality:
    print(key + ": " + str(value))
    if idx >= 10:
        break
    idx += 1
print()

eigen_centrality = nx.eigenvector_centrality(G)
sorted_eigen_centrality = [(k, eigen_centrality[k]) for k in sorted(eigen_centrality, key=eigen_centrality.get, reverse=True)]
idx = 0
print("Best centralities by eigen centrality:")
for key, value in sorted_eigen_centrality:
    print(key + ": " + str(value))
    if idx >= 10:
        break
    idx += 1
print()

# These wont compute (too complex graph)
#close_centrality = networkx.closeness_centrality(G)
#betw_centrality = networkx.betweenness_centrality(G)

# Communities
c = list(k_clique_communities(G, 8))
c = sorted(c, key=lambda l:len(l), reverse=True)
top_c = c[0]
print("Top community:")
print(top_c)
print("Length: " + str(len(top_c)))
print()

# Visualize the top community
top_community = G.subgraph(top_c)
plt.figure(figsize=(20,10))
pos = graphviz_layout(top_community, prog="fdp")
nx.draw(top_community, pos,
        labels={v:str(v) for v in top_community},
        cmap = plt.get_cmap("bwr"),
        node_color=[top_community.degree(v) for v in top_community],
        font_size=12
       )
plt.show()

# Create Kevin Bacon data
kevin_bacon_dict = networkx.single_source_dijkstra_path_length(G, "Kevin Bacon")

best = [(k, kevin_bacon_dict[k]) for k in sorted(kevin_bacon_dict, key=kevin_bacon_dict.get, reverse=False)]
worst = [(k, kevin_bacon_dict[k]) for k in sorted(kevin_bacon_dict, key=kevin_bacon_dict.get, reverse=True)]
mean = sum(kevin_bacon_dict.values()) / len(kevin_bacon_dict)

idx = 0
top_bacon = set()
print("Best Kevin Bacon numbers:")
for actor, value in best:
    if idx <= 10:
        print(actor + ": " + str(value))
    if idx >= 100:
        break
    top_bacon.add(actor)
    idx += 1

print()

idx = 0
print("Worst Kevin Bacon numbers:")
for actor, value in worst:
    print(actor + ": " + str(value))
    if idx >= 10:
        break
    idx += 1

print()

print("Mean Kevin Bacon number: " + str(int(mean)))
print()

# Visualize the best Kevin Bacon numbers
KevinBacon = G.subgraph(top_bacon)
plt.figure(figsize=(20,10))
pos = graphviz_layout(KevinBacon, prog="fdp")
nx.draw(KevinBacon, pos,
        labels={v:str(v) for v in KevinBacon},
        cmap = plt.get_cmap("bwr"),
        node_color=[KevinBacon.degree(v) for v in KevinBacon],
        font_size=12
       )
plt.show()
