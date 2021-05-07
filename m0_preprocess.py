import os 
import pickle
import numpy as np 
import pandas as pd 

# set up path 
path = os.path.dirname(os.path.abspath(__file__))

# utils
def clean_data( data):
    data = data.drop( columns = ['Unnamed: 0'])
    data = data.drop( index = data[ (data == -1).values \
                                  ].index.unique().values)
    return data 

def remake_cols( data):
    header = ['subject', 'block', 'setSize', 'trial', 'state', 'img', 
              'imgCat', 'iter', 'correctAct', 'action', 'keyNum', 
              'correct', 'reward', 'RT', 'expCondi', 'pcor', 'delay']
    data.columns = header
    data.state -= 1
    data.action -= 1
    data.correctAct -= 1 
    return data 

def split_data( data, mode = 'block'):
    setlist = np.sort( data.setSize.unique() )
    train_data = dict()
    count = 0

    if mode == 'subject':
        subjects = np.sort(data.subject.unique())
        for sub in subjects:
            train_data[sub] = data[ (data.subject==sub)]

    else:    
        for sz in setlist:
            subdata = data[ data.setSize==sz ]
            subjectlist = np.sort( subdata.subject.unique() )
            data_setsize = dict()
            
            for sub_idx in subjectlist:
                sub2data = subdata[ subdata.subject==sub_idx ]
                block_list = np.sort( sub2data.block.unique() )
                for block_idx in block_list:
                    sub3data = sub2data[ sub2data.block==block_idx ]
                    xi  = sub3data.loc[ :, ['subject','block', 'setSize', 'state', 'action', 'reward', 'iter', 'correctAct'] ]
                    yi_target = sub3data.action
                    xi.reset_index(drop=True, inplace=True)
                    yi_target.reset_index(drop=True, inplace=True)
                    if (mode == 'block') or (mode == 'test'):
                        train_data[ count] = xi
                        #train_label[ count] = yi_target
                    elif mode == 'setSize':
                        data_setsize[ count] = xi 
                    count += 1 
                    if mode=='test':
                        break
                if mode=='test':
                        break
                    
            if mode == 'setSize':
                train_data[ int(sz)] = data_setsize 
                    
    return train_data 

def pre_process_12():
    
    # load data 
    human_data = pd.read_csv( 'data/collins_12_orig.csv')
    human_data = clean_data( human_data)
    human_data = remake_cols( human_data)
    
    # save the remake columns
    human_data.to_csv( f'{path}/data/collins_12.csv')

    # split data into block data
    block_data = split_data( human_data, mode='block')
    with open( f'{path}/data/collins_12.pkl', 'wb')as handle:
        pickle.dump( block_data, handle)
   
    # split data into subject data
    subject_data = split_data( human_data, mode='subject')
    with open( f'{path}/data/collins_12_subject.pkl', 'wb')as handle:
        pickle.dump( subject_data, handle)

def pre_process_14():
    
    # load data 
    human_data = pd.read_csv( 'data/collins_14_orig.csv')
    human_data = clean_data( human_data)
    human_data = remake_cols( human_data)
    human_data = human_data[ human_data.expCondi==0]
    human_data.reset_index(drop=True, inplace=True)
    block_data = split_data( human_data, mode='block')

    with open( f'{path}/data/collins_14.pkl', 'wb')as handle:
        pickle.dump( block_data, handle)

    # split data into subject data
    subject_data = split_data( human_data, mode='subject')
    with open( f'{path}/data/collins_14_subject.pkl', 'wb')as handle:
        pickle.dump( subject_data, handle)
    

if __name__ == '__main__':
    
    # preprocess collins 12 data 
    pre_process_12()

    # preprocess collins 14 data 
    pre_process_14()
    

    

      
