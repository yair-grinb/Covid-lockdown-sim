import numpy as np
import pandas as pd
from parameters import jobs_per_m, avgIncome, stdIncome, infection_prob, \
    admission_prob, mortality_prob


def create_data(ind_file, blds_file):
    """
    Read input files and create the agets, buildings, households, and jobs arrays

    Parameters
    ----------
    ind_file : Str
        paths to agents csv file
    blds_file : Str
        path to buildings csv file

    Returns
    -------
    agents : numpy.array
        agent matrix
    households : numpy.array
        households matrix
    build : numpy.array
        buildings matrix
    jobs : numpy.array
        jobs matrix

    """
    
    # agents matrix fields:
    # 0 -id, 1 - household id, 2 - employment status, 3 - age, 4 - employed locally,
    # 5 - home building, 6 - religiousity, 7 - income, 8 - workplace building,
    # 9 - additional anchor activity building, 10 - statistical area, 11 - epidemiological status, 
    # 12 - infection risk, 13 - hospitalization risk, 14 - mortality risk, 15 - infection day,
    # 16 - infection duration, 17 - quarantine day, 18 - quarantine duration, 19 - hospitalization day
    # 20 - hospitalization duration, 21 - infecting agent
    
    # households matrix structure:
    # 0 - household id, 1 - home building, 2 - household income, 3 - religiousity
    
    # buildings matrix structure:
    # 0 - building id, 1 - general land use, 2 - statistical area id, 3 - floorspace, 4 - x, 5 - y,
    # 6 - specific land use, 7 - status
    
    # get agents
    agents = np.array(pd.read_csv(ind_file).values)
    agents = agents[:, [0, 1, 2, 3, 4, 5]]
    agents = np.append(agents, np.zeros((len(agents), 16)), axis= 1) 
    agents[:, 6:22] = np.nan
    
    agents[agents[:, 3]==1, 2] = 0
    agents[agents[:, 3]==1, 4] = 0
    agents[(agents[:, 2] == 0) & (agents[:, 4] == 1), 2] = 1 
    
    # define specific ages
    agents[agents[:, 3]==3, 3] = np.random.randint(66, 90, len(agents[agents[:, 3]==3]))
    agents[agents[:, 3]==2, 3] = np.random.randint(19, 65, len(agents[agents[:, 3]==2]))
    agents[agents[:, 3]==1, 3] = np.random.randint(1, 18, len(agents[agents[:, 3]==1]))
    
    # select agents starting as infected
    agents[:, 11] = 1
    agents[np.random.choice(range(len(agents)), 20, replace=False), 11] = 2
    agents[agents[:, 11]==2, 15] = 0
    
    #add infection prob by age per agent
    for inf in infection_prob:
        age_group = (agents[:, 3] >= inf[0]) & (agents[:, 3] < inf[1])
        agents[age_group, 12] = np.random.normal(inf[2], inf[3], len(agents[age_group]))
    agents[agents[:, 12] < 0, 12] = 0
    
    #add admission prob by age per agent
    for inf in admission_prob:
        age_group = (agents[:, 3] >= inf[0]) & (agents[:, 3] < inf[1])
        agents[age_group, 13] = np.random.normal(inf[2], inf[3], len(agents[age_group]))
    agents[agents[:, 13] < 0, 13] = 0
       
    #add mortality prob by age per agent
    for inf in mortality_prob:
        age_group = (agents[:, 3] >= inf[0]) & (agents[:, 3] < inf[1])
        agents[age_group, 14] = np.random.normal(inf[2], inf[3], len(agents[age_group])) 
    agents[agents[:, 14] < 0, 14] = 0
    
    # get households
    households = pd.read_csv(ind_file)
    households = households.iloc[:, [1, 5, 6]].groupby(households.columns[1]).first().reset_index().to_numpy()
    
    #religousity calculations
    households = np.append(households, np.random.randint(0,2,(len(households), 1)), axis=1)
    for h in households:
        members = agents[:, 1] == h[0]
        agents[members, 6] = h[3]
    
    # get buildings
    build=np.array(pd.read_csv(blds_file).values)
    build = np.append(build, np.ones((len(build), 1)), axis= 1)
    
    #add residence stat zone per agent
    for b in build:
        blds = agents[:, 5] == b[0]
        agents[blds, 10] = b[2]
    
    # workplace allocation
    
    # create jobs in buildings
    jobs_num = np.round_((np.choose(build[:, 1].astype(int), 
        (build[:, np.newaxis, 3] * np.array(jobs_per_m)).transpose())).tolist(), 0).astype(int)
    jobs = np.array([[np.random.normal(avgIncome, stdIncome), i, 0] for i in range(len(jobs_num)) 
                        for j in range(jobs_num[i])])
    
    while len(jobs[jobs[:, 0] <= 0]) > 0: 
        jobs[jobs[:, 0] <= 0, 0] = np.random.normal(avgIncome, stdIncome, len(jobs[jobs[:, 0] <= 0]))
    
    # compute expected income
    agents[:, 7] = 0
    expected_income = np.zeros(len(agents))
    for h in households: 
        working_members = (agents[:, 1] == h[0]) & (agents[:, 2] == 1)
        if len(agents[working_members]) > 0:
            expected_income[working_members] = h[2] / len(agents[working_members])
    
    commuters = (agents[:, 2]==1) & (agents[:, 4]==0)
    agents[commuters, 7] = expected_income[commuters]
    
    # match individuals to workplaces
    local_workers = np.where((agents[:, 4] == 1) & (agents[:, 7]==0))[0]
    av_jobs = np.where(jobs[:, 2] == 0)[0]
    while len(local_workers) > 0 and len(av_jobs) > 0:
        c = np.random.choice(local_workers)
        j = av_jobs[np.argmin(np.abs(expected_income[c] - jobs[av_jobs, 0]))]
        agents[c, [7,8]] = [jobs[j,0], build[int(jobs[j, 1]), 0]]
        jobs[j, 2] = 1
        local_workers = np.where((agents[:, 4] == 1) & (agents[:, 7]==0))[0]
        av_jobs = np.where(jobs[:, 2] == 0)[0]
    
    # update household income
    for h in range(len(households)): 
        households[h, 2] = np.sum(agents[agents[:, 1] == households[h, 0], 7])
    
    # identify building sets by specific landuse
    elementry = build[build[:,6]==5310,0]
    elementry_rel = build[build[:,6]==5312,0]
    high_schools = build[build[:,6]==5338,0]
    high_schools_rel = build[build[:,6]==5523,0]
    high_schools_rel = build[build[:,6]==5525,0]
    kinder = build[build[:,6]==5305,0]
    kinder_rel = build[build[:,6]==5300,0]
    religious = build[np.isin(build[:,6], [5501, 5521]), 0]
    yeshiva = build[build[:,6]==5340,0]
    etc = build[np.isin(build[:,6], 
                        [6512, 6520, 6530, 6600, 5740, 5760, 5600, 5700, 5202, 5253]),
                0]
    rel_etc = np.append(etc,religious)
    
    #inserting all non-working agents their activities
    secular_highschool = (agents[:, 6] == 0) & (agents[:, 3] <19)
    agents[secular_highschool, 8] = np.random.choice(high_schools, len(agents[secular_highschool]))
    
    secular_elementary = (agents[:, 6] == 0) & (agents[:, 3] <15)
    agents[secular_elementary, 8] = np.random.choice(elementry, len(agents[secular_elementary]))
    
    secular_kinder = (agents[:, 6] == 0) & (agents[:, 3] <7)
    agents[secular_kinder, 8] = np.random.choice(kinder, len(agents[secular_kinder]))
    
    rel_yeshiva = (agents[:, 6] == 1) & (agents[:, 3] < 25) & (np.isnan(agents[:, 8]))
    agents[rel_yeshiva, 8] = np.random.choice(yeshiva, len(agents[rel_yeshiva]))
    
    rel_highschool = (agents[:, 6] == 1) & (agents[:, 3] < 19)
    agents[rel_highschool, 8] = np.random.choice(high_schools_rel, len(agents[rel_highschool]))
    
    rel_elemntary = (agents[:, 6] == 1) & (agents[:, 3] < 15)
    agents[rel_elemntary, 8] = np.random.choice(elementry_rel, len(agents[rel_elemntary]))
    
    rel_kinder = (agents[:, 6] == 1) & (agents[:, 3] < 7)
    agents[rel_kinder, 8] = np.random.choice(kinder_rel, len(agents[rel_kinder]))
    
    unemployed = (np.isnan(agents[:, 8]))
    # assign an activity to randomly selected set of unemployed agents
    agents[unemployed , 8] = np.random.choice(etc, len(agents[unemployed])) * \
        np.random.randint(2, size=len(agents[unemployed]))
        
    agents[agents[:, 8]==0, 8] = np.nan
    
    #create more regular activities per agent
    rel_agents = agents[:, 6] == 1
    agents[rel_agents, 9] = np.random.choice(rel_etc, len(agents[rel_agents])) * \
        np.random.randint(2, size=len(agents[rel_agents]))
    sec_agents = ~rel_agents
    agents[sec_agents, 9] = np.random.choice(etc, len(agents[sec_agents])) * \
        np.random.randint(2, size=len(agents[sec_agents]))
    agents[agents[:, 9] == 0, 9] = np.nan
    
    return agents[:, [0, 1, 3, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                      20, 21]], households, build, jobs
