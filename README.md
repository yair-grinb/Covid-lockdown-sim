# Covid-lockdown-sim
#### Principal researchers: Dr. Yair Grinberger and Prof. Daniel Felsenstein, The Hebrew University of Jerusalem

**Overview:** This repository contains the code for an agent-based model of COVID-19 contagion process in cities and the effectiveness of various lockdown measures. 
The Agents in the model are residents who visit buildings as part of their daily routine. 
Which buildings are visitsed by which agents is determined based on a socio-spatial network connecting agents and agents, buildings and buildings, and agents and 
buildings.
The edges in this network represent the probability for contact between two entities. 
Contagion is determined stochastically based on co-location of agents in a building, the chance for them to interact, and other epidemiological factors such as age and 
duration since infection.
Lockdowns are exercised when the value of the contagion coefficient R exceeds a pre-defined threshold at which point some buildings will close down. 
Which buildings will close is determined by the type of lockdown. 
The outputs of the simulation model allow observing the spread of the epidemic across various scales and tracing complete contagion chains.

**Dependencies:**
1. Numpy (v. 1.19.2)
2. Pandas (v. 1.2.4)
3. Networkx (v. 2.5.1)
4. Scipy (v. 1.6.2)

**Inputs:** The model requires two inputs in the form of csv files - one containing information on agents (ID, household ID, age, employment status, etc.) and the 
other on buildings (ID, landuse, floorspace, etc.)
The files provided here are entirely synthetic and do not relate to any real world environment.

**Code:** The code is divided into 5 files:
1. parameters.py - used to store the values of all model parameters
2. auxilliary_functions.py - contains two functions, one for computing the R coefficient and the other to update buildings' status in accordance with lockdown policies.
3. initialization.py - contains the function which reads the input files and processes it to create the initial data required for running the model
4. network.py - contains the function structuring the socio-spatial network based on processed agents and buildings data
5. covid_model.py - the main script calling functions from the initialzation and network scripts, defining routines, and computing epidemiological patterns

**Outputs:** The outputs are stored in a dictionary and contain the following information:
1. Global daily indices - new infections, active infections, total infections, hospitalized agents, new and total dead, new and active quarantined, susceptible, 
recovered, the R value, number of closed buildings
2. The daily local R and observable R values per statistical area (census tract)
3. The daily number and rate of infected residents per building
4. The daily virus-import virus-export networks between statistical areas
5. Contagion chain data connecting infecting agents with the agents they infected
