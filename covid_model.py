# import required libraries
import numpy as np
import networkx as nx
from random import choice
from scipy.sparse.csgraph import shortest_path

# import model scripts
from parameters import k, norm_factor, recover, a_dist, bld_dist, contagious_risk_day, quarantine, \
    diagnosis, scenario_code, hospital_recover, agents_file, bldgs_file, max_bld_id
from auxilliary_functions import compute_R, building_lockdown
from initialization import create_data
from network import create_network


def initialize_world():
    """
    Create agents and buildings, compute interaction probabilities, and define routines for agents

    Returns
    -------
    agents : numpy.array
        agents array
    build : numpy.array
        building array
    interaction_prob : numpy.array
        probability distances between pairs of nodes
    bld_visits_by_agents : numpy.array
        matrix of routines - set of visited buildings per agent

    """
    agents, households, build, jobs = create_data(agents_file, bldgs_file)
    # agents matrix fields:
    # 0 - id, 1 - household id, 2 - age, 3 - home building, 4 - workplace building,
    # 5 - additional anchor activity, 6 - statistical area, 7 - epidemiological status,
    # 8 - infection probability, 9 - hospitalization probability, 10 - mortaility probability,
    # 11 - infection day, 12 - infection duration, 13 - quarantine day, 14 - quarantine duration, 
    # 15 - hospitalization day, 16 - hospitalization duration, 17 - infecting agent
    
    # buildings matrix fields:
    # 0 - building id, 1 - general land use, 2 - statistical area, 3 - floorspace volume, 
    # 4 - x, 5 - y, 6 - specific land use, 7 - status
        
    G = create_network(agents, households, build) # link agents and buildings
    
    # get network distances between all pairs of nodes and compute interaction probability
    nodes = np.array(G.nodes())
    g = nx.to_scipy_sparse_matrix(G)
    dists = shortest_path(g, directed=True, return_predecessors=False)
    np.fill_diagonal(dists, np.inf) # distance from node to self is infinity
    interaction_prob = np.exp(-dists[np.where(np.isin(nodes, agents[:, 0]))[0]][:,
                                     np.where(np.isin(nodes, agents[:, 0]))[0]])
    
    # get anchor activities for agents
    agents_reg = agents[:, [0,3,4,5,3]]
    
    bld_visits_by_agents = []
    for n in range(len(agents_reg)):
        a = agents_reg[n, 0]
        current_position = agents_reg[n, 1]
        # buildings within distance<a_dist from a
        a_nodes = nodes[(dists[np.where(nodes==a)][0]<a_dist) & (np.isin(nodes, build[:, 0]))] 
        visits = [current_position]
        for i in agents_reg[n, 2:]:
            if ~np.isnan(i):
                # buildings within bld_dist from i
                i_nodes = nodes[(dists[:, np.where(nodes==i)[0][0]]<bld_dist) & (nodes<=max_bld_id)] 
                for j in range(k):
                    if np.random.randint(2) > 0:
                        # buildings within bld_dist from current position
                        c_nodes = nodes[(dists[np.where(nodes==current_position)][0]<bld_dist) &
                                        (nodes<=max_bld_id)] 
                        
                        intersect = np.intersect1d(i_nodes, c_nodes)
                        union = np.union1d(intersect, a_nodes)
                        
                        destination = choice(union)
                        current_position = destination
                        visits.append(current_position)
                        
                current_position = i
                visits.append(current_position)
                
        bld_visits_by_agents.append(visits)
    
    # create building visits matrix
    bld_visits_by_agents = np.array(bld_visits_by_agents)
    zero = np.zeros([len(bld_visits_by_agents),len(max(bld_visits_by_agents,key = lambda x: len(x)))])
    for i,j in enumerate(bld_visits_by_agents):
        zero[i][0:len(j)] = j
    bld_visits_by_agents = zero
    bld_visits_by_agents[bld_visits_by_agents==0] = np.nan
    
    return agents, build, interaction_prob, bld_visits_by_agents


def run_model():
    """
    Run the model - until no infected agents remain, identify agent status change and activate lockdowns 

    Returns
    -------
    outputs : dict
        output data - global indices (Stats) per day, 
        buildings statuses (Buildings) per day,
        neighborhood level R and observable R values (SAs) per day,
        daily virus import-export networks between neighborhoods (IO_mat) per day,
        data for reconstructing contagion chains (Contagion chain)
    """
    
    day = 0
    outputs = {'Stats':{}, 'Buildings':{}, 'SAs':{}, 'IO_mat':{}}
    
    agents, build, interaction_prob, bld_visits_by_agents = initialize_world()
    print('World created')
    
    sas_vis_R = {} # recording neighborhood level Re values
    for sa in np.unique(build[:, 3]):
        sas_vis_R[sa] = 0
        
    # model iterations - while there are infected agents
    while len(agents[agents[:, 7] == 2]) + len(agents[agents[:, 7] == 3.5]) \
        + len(agents[agents[:, 7] == 4]) + len(agents[agents[:, 7] == 5])> 0:    
        
        # a copy of building visits matrix updated in accordance with lockdown
        bld_visits = bld_visits_by_agents * build[
            np.argmax(build[:, 0][None, :] == bld_visits_by_agents[:, :, None], axis=2), 7]
        
        # check if agents are in quarantine and if yes - all activities but first (home) are set to zero
        bld_visits[:, 1:] = bld_visits[:, 1:] * ((agents[:, 7] != 3) & (agents[:, 7] != 4)
                                                  & (agents[:, 7] != 3.5)).reshape((len(agents), 1))
        
        # check if agents are admitted or dead and if yes - all activities are set to zero
        bld_visits[:, 0:] = bld_visits[:, 0:] * ((agents[:, 7] != 5) & (agents[:, 7] != 7)).reshape(
            (len(agents), 1))
        
        bld_visits[bld_visits==0] = np.nan
        
        # update number of days since infection, quarantine, and hospitalization
        infected = (agents[:, 7] == 2) | (agents[:, 7] == 4) | (agents[:, 7] == 3.5)| \
            (agents[:, 7] == 5)
        agents[infected, 12] = day - agents[infected, 11] 
        
        quarantined = (agents[:, 7] == 3) | (agents[:, 7] == 3.5) | (agents[:, 7] == 4)
        agents[quarantined, 14] = day - agents[quarantined, 13] 
        
        hospitalized = (agents[:, 7] == 5)
        agents[hospitalized, 16] = day - agents[hospitalized, 15]
        
        # get buildings visited by infected and by susceptible agents
        infected = np.where((agents[:,7]==2) | (agents[:,7]==4) | (agents[:, 7]==3.5))[0]
        susceptible = np.where((agents[:,7]<2) | (agents[:,7]==3))[0]
        infected_blds = bld_visits[infected]
        susceptible_blds = bld_visits[susceptible]
        
        # update hospitalization probability
        no_hospitalization = np.where((agents[:,11] < 4) | (agents[:,11] > 14) | (agents[:, 7] >= 5) )[0]
        hospitalization_prob = agents[:,9]
        hospitalization_prob = hospitalization_prob.copy()
        hospitalization_prob[susceptible] = 0
        hospitalization_prob[no_hospitalization] = 0
        
        # identify new hospitalized agents
        rand_hospitalization = np.random.random(hospitalization_prob.shape)
        hospitalizations = hospitalization_prob > rand_hospitalization
        new_hospitalization = np.where(hospitalizations == True)
        agents[hospitalizations, 7] = 5
        agents[hospitalizations, 15] = day
        
        # update death probability
        no_death = np.where(agents[:,15] < 3)[0]
        unhospitalized = np.where(agents[:,7]!=5)[0]
        death_prob = agents[:,10]
        death_prob = death_prob.copy()
        death_prob[unhospitalized] = 0
        death_prob[no_death] = 0
        
        # identify new dead agents
        rand_death = np.random.random(death_prob.shape)
        deaths = death_prob > rand_death
        new_deaths = np.where(deaths == True)
        agents[deaths, 7] = 7
        
        # calculate infection probability:
        #interaction p * p for being infected * p for infecting * exposure * n
        
        # check whether susceptibel agent was exposed to an infected agent
        exposure = 1*np.array([np.isin(infected_blds,susceptible_blds[i]).any(axis=1) 
                                for i in range(len(susceptible_blds))])
        
        infection_prob = interaction_prob[susceptible][:, infected] * agents[
            susceptible, 8].reshape((len(susceptible), 1)) 
        infection_prob *= contagious_risk_day.pdf(agents[infected, 12])
        infection_prob *= exposure 
        infection_prob *= norm_factor
        
        # identify new infected
        rand_cont = np.random.random(infection_prob.shape)
        infections = infection_prob > rand_cont
        new_infected = np.where(infections.any(axis=1) & (agents[susceptible, 7]<2))
        new_quarantined_infected = np.where(infections.any(axis=1) & (agents[susceptible, 7]==3))
        
        agents[susceptible[new_infected], 7] = 2
        agents[susceptible[new_infected], 11] = day
        
        agents[susceptible[new_quarantined_infected], 7] = 3.5
        agents[susceptible[new_quarantined_infected], 11] = day
        
        # this allows tracing infection chains - who infected whom and when
        agents[susceptible[np.where(infections)[0]], 17] = agents[infected[np.where(infections)[1]], 0]
        
        # end quarantine
        agents[(agents[:, 7] == 3) & (agents[:, 14] == quarantine), 7] = 1 
        agents[(agents[:, 7] == 3.5) & (agents[:, 14] == quarantine), 7] = 2
        
        # enter agents to quarantine
        agents[(agents[:, 7] == 2) & (agents[:, 12] == diagnosis), 13] = day 
        agents[((agents[:, 7] == 2) | (agents[:, 7] == 3.5)) & (agents[:, 12] == diagnosis), 7] = 4
        
        # agents recover
        agents[(agents[:, 7] == 4) & (agents[:, 12] == recover), 7] = 6
        agents[(agents[:, 7] == 5) & (agents[:, 12] == hospital_recover), 7] = 6
        
        # reset quarantine count for non-quarantined agents
        agents[(agents[:, 7] == 1) | (agents[:, 7] == 2) | (agents[:, 7] == 6) | 
               (agents[:, 7] == 7), 14] = 0
        agents[(agents[:, 7] == 1) | (agents[:, 7] == 2) | (agents[:, 7] == 6) | 
               (agents[:, 7] == 7), 16] = 0
        
        # enter household members to quarantine
        new_diagnosed_agents = agents[(agents[:, 7] == 4) & (agents[:, 12] == diagnosis)]
        if len(new_diagnosed_agents) > 0: # if there are sick agents in quarantine
            # uninfected household members
            new_quar = np.where((np.isin(agents[:, 1],new_diagnosed_agents[:, 1])) & 
                                (agents[:, 7]==1))[0]
            agents[new_quar, 7] = 3
            agents[new_quar, 13] = day
            
            # infected undiagnosed household members
            new_quar_infected = np.where((np.isin(agents[:, 1],new_diagnosed_agents)) & 
                                (agents[:, 7]==2))[0]
            agents[new_quar_infected, 7] = 3.5
            agents[new_quar_infected, 13] = day
        
        # collect output data
        R = compute_R(agents, day)
        new_infections = len(agents[(agents[:, 7] != 1) & (agents[:, 7] != 3) 
                                    & (agents[:, 11] == day)])
        dead = np.where(agents[:,7]==7)[0]
        if day > diagnosis:
            vis_R = outputs['Stats'][day-diagnosis]['R']
        else:
            vis_R = 0
        
        outputs['Stats'][day] = {'Active infected': len(agents[(agents[:, 7] == 2) | 
                                                               (agents[:, 7] == 3.5) | 
                                                               (agents[:, 7] == 4) | 
                                                               (agents[:, 7] == 5)]),
                        'Daily infections': new_infections,
                        'Recovered': len(agents[agents[:, 7] == 6]),
                        'Quarantined': len(agents[(agents[:,7]>=3) & (agents[:, 7] <= 4)]),
                        'Daily quarantined': len(agents[(agents[:,7]>=3) & 
                                                        (agents[:, 7] <= 4) & 
                                                        (agents[:, 13] == day)]),
                        'Hospitalized': len(agents[(agents[:, 7] == 5)]),
                        'Daily hospitalizations': len(new_hospitalization[0]),
                        'Total Dead': len(dead),
                        'Daily deaths': len(new_deaths[0]),
                        'R': R,
                        'Known R': vis_R,
                        'Closed buildings': len(build[build[:, 7] == 0])}
        
        if day==0:
            outputs['Stats'][day]['Total infected'] = outputs['Stats'][day]['Active infected']
        else:
            outputs['Stats'][day]['Total infected'] = outputs['Stats'][day-1]['Total infected'] + new_infections
        outputs['Stats'][day]['Susceptible'] = len(agents) - outputs['Stats'][day]['Total infected']
        
        print('day {}: new infected - {}, total infected - {}, total dead - {}'.format(
            day, outputs['Stats'][day]['Daily infections'],
            outputs['Stats'][day]['Total infected'],
            outputs['Stats'][day]['Total Dead']))
        
        # document neighborhood level R values
        sas_R = {}
        outputs['SAs'][day] = {}
        for sa in np.unique(build[:, 2]):
            sa_agents = agents[agents[:, 6] == sa]
            sas_R[sa] = compute_R(sa_agents, day)
        outputs['SAs'][day]['R'] = sas_R
        
        # document neighborhood level observable (lagged) R value
        sas_vis_R = {}
        for sa in np.unique(build[:, 2]):
            if day > diagnosis:
                sa_agents = agents[agents[:, 6] == sa]
                sas_vis_R[sa] = outputs['SAs'][day-diagnosis]['R'][sa] #compute_vis_R(sa_agents, day)
            else:
                sas_vis_R[sa] = 0
        outputs['SAs'][day]['vis_R'] = sas_vis_R
        
        # document building-level states
        outputs['Buildings'][day] = []
        for b in build:
            b_pop = agents[agents[:, 3] == b[0]]
            if b_pop.shape[0] > 0:
                b_infected = b_pop[(b_pop[:, 7] == 2) | (b_pop[:, 7] == 3.5) | 
                                    (b_pop[:, 7] == 4)].shape[0]
                outputs['Buildings'][day].append([b[0], b_infected, 
                                                              b_infected / b_pop.shape[0]])
        
        # document virus import-export flows between neighborhoods
        sas = np.unique(build[:, 2])
        infect_mat = {sa:{sa2: 0 for sa2 in sas} for sa in sas}
        infected_sas = agents[agents[:, 11]==day, 6]
        infecting_sas = agents[np.where(agents[agents[:, 11]==day, 17][:, None] ==
                                            agents[:, 0][None, :])[1], 6]
        if infecting_sas.size > 0:
            for sa in sas:
                counts = np.unique(infecting_sas[infected_sas == sa], return_counts=True)
                for i in range(len(counts[0])):
                    infect_mat[sa][counts[0][i]] = int(counts[1][i])
        outputs['IO_mat'][day] = infect_mat
        
        # activate lockdowns
        if 'DIFF' in scenario_code:
            for s in sas_vis_R:
                if day!=0:
                    prevVR = outputs['SAs'][day-1]['Vis_R'][s]
                else:
                    prevVR = 0
                lockdown = building_lockdown(build[build[:, 2] == s], scenario_code, 
                                             sas_vis_R[s], prevVR)
                build[build[:, 2] == s, 7] = lockdown
            del s, lockdown
        else: # full lockdown
            if day!=0:
                prevVR = outputs['Stats'][day-1]['Known R']
            else:
                prevVR = 0
            lockdown = building_lockdown(build, scenario_code, vis_R, prevVR)
            build[:, 7] = lockdown
            del lockdown
        del vis_R
        day += 1
    
    # document contagin chains - who infected whom
    outputs['Contagion chain'] = agents[:, [0, 12, 17]].tolist()
    return outputs


run_model()