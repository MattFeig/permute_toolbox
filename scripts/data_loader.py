import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from permute_toolbox.parcel_tools import reorder_indices
from permute_toolbox.rsfc_tools import flatten_upper_triangle_3D, create_corrmat


def data_loader(sub_demo_list):
   
    """Returns an roi x roi x participant np array, 
    for the particpants specified in the input subject demographic list"""


    absolute_path = os.path.dirname(__file__)
    relative_path = '../data/connectivity_data'
    full_path = os.path.join(absolute_path, relative_path)
    conn_directory = full_path

    conn = []

    for index, row in sub_demo_list.iterrows():
        sub = row.VC
        
        # Check which type of subject ID is in the demo list. 
        # Depending on the study ID, there corresponding connectivity csv path will need a different name formatting
    
        if (sub[4:9] == 'MSCPI') | (sub[4:7] == 'NDA') | (sub[4:8] == 'LoTS'):
            csv = f'{sub}_Gordon_Subcort_0.2FDcens_CONCAT_ROIordered_zmat.csv'
        elif sub[4:6] == 'NT':
            csv = f'{sub}ses-screen_task-rest_DCANBOLDProc_v4.0.0_Gordon_subcorticals_0.2_5contig_FD_zmat.csv'  
        elif (sub[:2] == 'vc') | (sub[:2] == 'VC') | (sub[:3] == 'NIC') | (sub[:4] == 'SAIS') :
            if '_2' in sub:
                sub = sub.replace('_2', 'V2')
            if '_' in sub:
                sub = sub.replace('_', '')   
            if sub == 'SAISV209':
                sub = 'SAIS209'
            if sub == 'vctb0053east':
                sub = 'TB0053E'                
            sub = sub.upper()
            csv = f'sub-{sub}_Gordon_Subcort_0.2FDcens_CONCAT_ROIordered_zmat.csv'
        else:
            print(sub) 
            break

        filepath = os.path.join(conn_directory, csv)

        if os.path.exists(filepath):
            sub_conn = np.genfromtxt(filepath, delimiter=',')
            conn.append(sub_conn[:333,:333])
        else:
            print(sub)
            break
    
    return np.stack(conn,2)

def prep_data():
    # Load Demographic data and subject lists
    print(os.getcwd())
   
    absolute_path = os.path.dirname(__file__)
    relative_path = '../data/demo.csv'
    demo_path = os.path.join(absolute_path, relative_path)
    demo = pd.read_csv(demo_path, names=['VC', 'Age', 'Group'])

    demo_ts = demo.where(demo.Group=='TS').dropna().reset_index(drop=True)
    demo_hc = demo.where(demo.Group=='TFC').dropna().reset_index(drop=True)

    # custom data loader reads in connectivity data for each group seperately
    ts_con = data_loader(demo_ts)
    hc_con = data_loader(demo_hc)

    # Connectivity matricies are symettric and square. We need just the flattened upper or lower triangle, of the matrix
    # to create a new design matrix

    ts_con_flat = flatten_upper_triangle_3D(ts_con)
    hc_con_flat = flatten_upper_triangle_3D(hc_con)

    # create feature matrix
    X = np.vstack((hc_con_flat, ts_con_flat))

    # create label vector: 1 for HC, -1 for TS
    y = np.concatenate((np.repeat(1,99), np.repeat(-1,99)))

    scaler = StandardScaler()
    X_scale = scaler.fit_transform(X)

    return X_scale, y


def reformat_vc(name):
    if '_2' in name and len(name[(name.find('_2')+2):]) == 0:
        name = name.replace('_2', 'V2')
    if '_' in name:
        name = name.replace('_', '')  
    name = name.upper()
    return name

def load_timeseries_data():
    
    participants = pd.read_excel('../data/demo_clin_data/participants.xlsx')
    ts202_data_dir = '../data/ts202_timeseries'

    participants_TS = participants[(participants['Class']=='TS') & (participants['Age']<=13.1)]
    participants_TFC = participants[(participants['Class']=='TFC') & (participants['Age']<=13)]

    # participants_TS = participants[(participants['Class']=='TS') ]
    # participants_TFC = participants[(participants['Class']=='TFC') ]
    # participants_TS = participants[(participants['Class']=='TS') & (participants['Age']>=18)]
    # participants_TFC = participants[(participants['Class']=='TFC') & (participants['Age']>=18)]

    participants_TS.VC = participants_TS.VC.apply(reformat_vc)
    participants_TFC.VC = participants_TFC.VC.apply(reformat_vc)

    TS_timeseries_list = []
    for sub in participants_TS.VC:
        filename = f'sub-{sub}_Gordon_Subcort_0.2FDcens_CONCAT_UNORDERED_TIMESERIES.txt'
        filepath = os.path.join(ts202_data_dir,filename)
        TS_timeseries_list.append(np.genfromtxt(filepath))
        
    TFC_timeseries_list = []
    for sub in participants_TFC.VC:
        filename = f'sub-{sub}_Gordon_Subcort_0.2FDcens_CONCAT_UNORDERED_TIMESERIES.txt'
        filepath = os.path.join(ts202_data_dir,filename)
        TFC_timeseries_list.append(np.genfromtxt(filepath))

    ts_con_stack_unordered = np.stack([create_corrmat(x) for x in TS_timeseries_list],2)
    tfc_con_stack_unordered = np.stack([create_corrmat(x) for x in TFC_timeseries_list],2)

    ts_con_stack_ordered = np.copy(ts_con_stack_unordered)[reorder_indices,:,:][:,reorder_indices,:]
    tfc_con_stack_ordered = np.copy(tfc_con_stack_unordered)[reorder_indices,:,:][:,reorder_indices,:]

    TS_con_flat = flatten_upper_triangle_3D(ts_con_stack_ordered)
    TFC_con_flat = flatten_upper_triangle_3D(tfc_con_stack_ordered)

    X = np.vstack((TS_con_flat,TFC_con_flat))

    # create label vector: 1 for TS, -1 for TFC
    y = np.concatenate((np.repeat(1,np.shape(TS_con_flat)[0]), np.repeat(-1,np.shape(TFC_con_flat)[0])))

    return X, y