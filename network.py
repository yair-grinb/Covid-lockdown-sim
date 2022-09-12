import numpy as np
import networkx as nx
import scipy.spatial as spatial
from parameters import beta, w_a, w_i, w_d, b_min_prob, a_min_prob


def create_network(agents, households, build):
    """
    Creates a socio-spatial network connecting agents and buildings

    Parameters
    ----------
    agents : numpy.array
        agents matrix
    households : numpy.array
        households matrix
    build : numpy.array
        buildings matrix

    Returns
    -------
    G : networkx.DiGraph
        Directed graph representing the network

    """
    
    # compute gravity model probabilities for trnasition betweenb uildings
    fs = build[:, 3]
    dists = spatial.distance_matrix(build[:, 4:6], build[:, 4:6])**beta
    dists[dists==0] = 0.00001
    scores = fs / dists
    for i in range(len(build)):
        scores[i, i] = 0
    rows_sum = scores.sum(axis=1).reshape((len(build), 1))
    bld_prob = scores / rows_sum
    
    # compute distance between agents based on household incomes, agnet ages, and physical distance
    agent_hh = (households[:, 0][:, None] == agents[:, 1]).argmax(axis=0)
    hh_income = households[agent_hh, 2]
    income_dist = np.abs(np.subtract(hh_income, hh_income.reshape((len(agents), 1))))
    income_dist = 1 - income_dist / np.max(income_dist)
    
    age_dist = np.abs(np.subtract(agents[:, 2], agents[:, 2].reshape((len(agents), 1))))
    age_dist = 1 - age_dist / np.max(age_dist)
    
    agent_homes = (build[:, 0][:, None] == agents[:, 3]).argmax(axis=0)
    agent_dists = spatial.distance_matrix(build[agent_homes, 4:6], build[agent_homes, 4:6])
    agent_dists = 1 - (agent_dists) / np.max(agent_dists)
    
    agents_prob = w_a * age_dist * w_i * income_dist * w_d * agent_dists
    agents_prob[agent_hh == agent_hh.reshape((len(agents), 1))] = 1
    
    # create network
    G = nx.DiGraph()
    
    # create edges between buildings weighted by the computed probabilities only if probability>0.5
    edges = [(build[b,0], build[b1,0], -np.log(bld_prob[b, b1])) for b in range(len(build)) 
             for b1 in range(len(build)) if bld_prob[b, b1] > b_min_prob and b != b1]
    
    # create edges with probability weight=1 between agents and their anchor activities
    edges.extend([(agents[a, 0], build[agent_homes[a], 0], np.log(1)) for a in range(len(agents))])
    edges.extend([(agents[a, 0], agents[a, 4], np.log(1)) for a in range(len(agents)) 
                  if ~np.isnan(agents[a, 4])])
    edges.extend([(agents[a, 0], agents[a, 5], np.log(1)) for a in range(len(agents)) 
                  if ~np.isnan(agents[a, 5])])
    G.add_weighted_edges_from(edges)

    # create edges between agents and agents weighted by probabilities if probability>0.5
    links = np.where(agents_prob>a_min_prob)
    for i in range(len(links[0])):
        if links[0][i] != links[1][i]:
            G.add_weighted_edges_from([(agents[links[0][i], 0],
                                        agents[links[1][i], 0],
                                        -np.log(agents_prob[links[0][i], links[1][i]]))])
    
    return G