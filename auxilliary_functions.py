import numpy as np
from parameters import recover, contagious_risk_day

def compute_R(a, t):
    """
    Compute the Re contagion coefficientat a given time step for a set of agents

    Parameters
    ----------
    a : numpy.array
        agents matrix
    t : int
        time step

    Returns
    -------
    R : float
        Re coefficient value

    """
    
    new_infections = len(a[(a[:, 7] != 1) & (a[:, 7] != 3) & (a[:, 11] == t)])
    sum_I = 0.
    for i in range(1, recover):
        sum_I += a[(a[:, 7] != 1) 
                   & (a[:, 7] != 3) 
                   & ((t - a[:, 11]) == i)].shape[0] * contagious_risk_day.pdf(i)
    if sum_I > 0:
        R = new_infections / sum_I
    else:
        R = 0
    return R


def building_lockdown(b, sc, vR, prevVR):
    """
    Activating and deactivating lockdown scenarios

    Parameters
    ----------
    b : numpy.array
        building matrix
    sc : String
        scenario name
    vR : Float
        known R value
    prevVR : Float
        R value over previous iteration

    Returns
    -------
    lockdown : numpy.array
        Buildings' updated statuses

    """
    
    lockdown = np.ones(len(b))
    if 'GRADUAL' in sc:
        if 1 < vR < 2: 
            if prevVR <= 1 or prevVR >=2:
                if 'ALL' in sc: # close half of non-residential buildings
                    non_residential = np.where(b[:,1]>=3)[0]
                    lockdown[np.random.choice(non_residential, replace=False,
                                                    size=int(non_residential.size * 0.5))] = 0
                else:
                    if 'EDU' in sc: # close half of the educational buildings
                        education = np.where((b[:, 6] == 5310) | (b[:, 6] == 5312) | 
                                             (b[:, 6] == 5338)| (b[:, 6] == 5523)| 
                                             (b[:, 6] == 5525)| (b[:, 6] == 5305)| 
                                             (b[:, 6] == 5300)| (b[:, 6] == 5340))[0]
                        lockdown[np.random.choice(
                            education, replace=False, size=int(education.size*0.5))] = 0
                    if 'REL' in sc: # close half of the religious buildings
                        religious = np.where((b[:,6] == 5501) | (b[:,6] == 5521))[0]
                        lockdown[np.random.choice(
                            religious, replace=False, size=int(religious.size*0.5))] = 0
            else: # do not update statuses
                lockdown = b[:, 7]
        elif vR >= 2: 
            if 'ALL' in sc:
                lockdown[b[:, 1] >= 3] = 0 #close all non-residential buildings
            else:
                if 'EDU' in sc: # close all educational buildings
                    education = np.where((b[:, 6] == 5310) | (b[:, 6] == 5312) | 
                                         (b[:, 6] == 5338)| (b[:, 6] == 5523)| 
                                         (b[:, 6] == 5525)| (b[:, 6] == 5305)| 
                                         (b[:, 6] == 5300)| (b[:, 6] == 5340))[0]
                    lockdown[education] = 0
                if 'REL' in sc: # close all religious buildings
                    religious = np.where((b[:,6] == 5501) | (b[:,6] == 5521))[0]
                    lockdown[religious] = 0
    else:
        if 1 <  vR:
            if 'ALL' in sc: # close all non-residential buildings
                lockdown[b[:, 1] >= 3] = 0
            else:
                if 'EDU' in sc: # close all educational buildings
                    education = np.where((b[:, 6] == 5310) | (b[:, 6] == 5312) | 
                                         (b[:, 6] == 5338)| (b[:, 6] == 5523)| 
                                         (b[:, 6] == 5525)| (b[:, 6] == 5305)| 
                                         (b[:, 6] == 5300)| (b[:, 6] == 5340))[0]
                    lockdown[education] = 0
                if 'REL' in sc: # close all religious buildings
                    religious = np.where((b[:,6] == 5501) | (b[:,6] == 5521))[0]
                    lockdown[religious] = 0
    return lockdown