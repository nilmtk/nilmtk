import numpy as np
from copy import deepcopy


def compute_RE_MNE(power_states_dict, df_appliances_test):
    '''
    INPUT
    -----
    power_states_dict is an ordered dictionary of the form
    {0 : {'states':case_0_states, 'power': case_0_power},
     1:.....
    }
     Here, each of case_i_states and case_i_power is also a dictionary
     of the format
     case_i_power={ 'refrigerator':[0,1,...]
     }

     OUTPUT
     ------

     '''

    MNE = []
    RE = []

    for i in range(4):
        # We have 4 cases
        numerator={}
        denominator={}
        mne={}
        re={}
        for appliance in power_states_dict[0]['states']:
            numerator[appliance]=np.sum(np.abs(power_states_dict[i]['power'][appliance]-    df_appliances_test[appliance].values))
            denominator[appliance]=np.sum(df_appliances_test[appliance].values)
            mne[appliance]=numerator[appliance]*1.0/denominator[appliance]
            re[appliance]=np.std(power_states_dict[i]['power'][appliance]-  df_appliances_test[appliance].values)

        MNE.append(deepcopy(mne))
        RE.append(deepcopy(re))

    return [MNE,RE]