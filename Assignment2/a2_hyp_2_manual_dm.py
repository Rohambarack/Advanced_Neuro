#import
import pickle
import numpy as np
import pandas as pd
import nilearn
import os
import re
from nilearn.glm.first_level import first_level_from_bids
from sklearn.metrics import explained_variance_score
import datetime
from nilearn.glm.first_level import make_first_level_design_matrix as manualdm

#functions
# la = net loss risk aversve
# ls = net loss risk seeking
# ga = net gain risk averse
# gs = net gain risk seeking
def label_risk(row):
    if row["response"] == 1:
        return "a"
    if row["response"] == 2:
        return "s"
    if row["response"] == 3:
        return "a"
    if row["response"] == 4:
        return "s"

def label_loss(row):
    if row["response"] == 1:
        return "l"
    if row["response"] == 2:
        return "l"
    if row["response"] == 3:
        return "g"
    if row["response"] == 4:
        return "g"

def label_total(row):
    if row["total"]  < 0:
      return "neg"
    else :
      return "pos"


 ####################### get eveerything
def set_it_up_no_model():
    derivatives_dir = os.path.join("/work","BIDS_2024E","derivatives")
    data_dir= os.path.join("/work","BIDS_2024E")
    # Name for experiment in the BIDS directory
    task_label = 'boldiowa'
    #Run the function that can gather all the needed info from a BIDS folder
    #
    models, models_run_imgs, models_events, models_confounds = \
        first_level_from_bids(
            data_dir, task_label, derivatives_folder=derivatives_dir, n_jobs=-1,
            verbose=0, minimize_memory = False, slice_time_ref = 0.462,
            img_filters=[('desc', 'preproc')])

    ########################set confounds
    #motion parameters are individually extracted
    confine_36_names = ['trans_x','trans_y','trans_z',
                        'rot_x','rot_y','rot_z',
                       "trans_x_derivative1",'trans_y_derivative1','trans_z_derivative1',
                       'rot_x_derivative1','rot_y_derivative1','rot_z_derivative1',
                       "trans_x_power2",'trans_y_power2','trans_z_power2',
                        'rot_x_power2','rot_y_power2','rot_z_power2',
                       'global_signal','global_signal_derivative1','global_signal_power2',
                       'csf','csf_derivative1','csf_power2',
                       'white_matter','white_matter_derivative1','white_matter_power2',
                       "trans_x_derivative1_power2",'trans_y_derivative1_power2','trans_z_derivative1_power2',
                       'rot_x_derivative1_power2','rot_y_derivative1_power2','rot_z_derivative1_power2',
                       'global_signal_derivative1_power2','csf_derivative1_power2','white_matter_derivative1_power2']


     # Subset confounds with selection
    for ii in range(len(models_confounds)):
        #copy var to save previous separate for participant
        confounds1=models_confounds[ii][:].copy()
        for i in range(len(confounds1)):
            #copy again, separate for run
            confounds2=confounds1[i].copy()
            #get confounds cos motion is separate for each participant
            #make list of confound names
            confound_list = []
            for k in confounds2.columns :
                confound_list.append(k)
            #find indeces of columns starting with "motion"
            motion_spike_index_list = []
            for g in confound_list:
                #tries to find the name
                motion_spike = re.findall("motion",g)
                #if the motion spike is found in the confound list, take its index 
                if motion_spike:
                    motion_spike_index = (confound_list.index(g))
                    motion_spike_index_list.append(motion_spike_index)
            #get motion variables by their index
            motion_spikes = []
            for h in motion_spike_index_list:
                spike = confound_list[h]
                motion_spikes.append(spike)
            
            #add motion spikes to confound list
            full_confounds = confine_36_names + motion_spikes
            #subset all confounds with selected ones
            confounds2=confounds2[full_confounds]
            #Removing NAs in the first row because it doesn't let the model run otherwise 
            #MAYBE NOT THE MOST SCIENTIFIC WAY TO DO THIS
            confounds2.loc[0,:]=confounds2.loc[1,:]
            confounds1[i]=confounds2
        models_confounds[ii][:]=confounds1

    ################### events
    events_sub= ['onset','duration',"trial_type"]
    # Subset model events with selection
    for ii in range(len(models_events)):
        events1=models_events[ii][:]
        for i in range(len(events1)):
            events2=events1[i].copy()
            ##########add total to decks beccause it is only listed in the win/loss part of the dataframe
            for index in range(40):
                events2.loc[index,"total"] =  events2.loc[index+40,"total"]
            #separate deck rows
            events2 = events2[events2["trial_type"] == "decks"]
            #sort by response"
            events2 = events2.sort_values("RT")
            #rename onset
            events2["onset"] = events2["RT"]
            #add new columns, separate choices and categorize them
            events2["risk_cat"] = events2.apply(label_risk, axis = 1)
            events2["loss_cat"] = events2.apply(label_loss, axis = 1)
            events2["total_cat"] = events2.apply(label_total, axis = 1)
            
            #events2["trial_type"] = events2["loss_cat"]+ "_" + events2["risk_cat"] + "_" +  events2["total_cat"]
            events2["trial_type"] = events2["risk_cat"] + "_" +  events2["total_cat"]
            #subset needed columns
            events2=events2[events_sub]
            # filter nans
            events2 = events2[events2["trial_type"].notnull()]
            events1[i]=events2
            
        models_events[ii][:]=events1
        

    return models, models_run_imgs, models_events, models_confounds

def make_it_dm(models_events,models_confounds):
    #manually making a desing matrix for padding with 0s
    #setup for design matrix
    t_r = 1.0  # repetition time is 1 second
    n_scans = 600  # the acquisition comprises 600 scans
    frame_times_p = (
        np.arange(n_scans) * t_r
    )  # here are the corresponding frame times
        #1st p 2st run
    design_matrix = manualdm(frame_times = frame_times_p, 
                    events=models_events, 
                    hrf_model='glover', 
                    drift_model='cosine', 
                    high_pass=0.01, 
                    drift_order=1, 
                    fir_delays=None, 
                    add_regs=models_confounds, 
                    #add_reg_names=None, 
                    min_onset=-24, 
                    oversampling=50)
    
    return design_matrix

def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros
        

def pad_dm(design_matrix):
    #pad the design matrix with zeros in the appropriate columns
    trial_types_list = ["a_neg","a_pos","s_neg","s_pos"]
    for type_index in range(len(trial_types_list)):
        if design_matrix.columns[type_index] != trial_types_list[type_index]:
            design_matrix.insert(type_index,trial_types_list[type_index],zerolistmaker(len(design_matrix)))
            
    return design_matrix

def  main():
    models, models_run_imgs, models_events, models_confounds = set_it_up_no_model()
    #make design_matrix list of 6 participants, 4 runs each
    p_design_matrix_list = []
    for k in range(len(models_events)):
        r_design_matrix_list = []
        for g in range(len(models_events[k])):
            design_matrix = make_it_dm(models_events = models_events[k][g], models_confounds = models_confounds[k][g])
            padded_matrix = pad_dm(design_matrix)
    
            r_design_matrix_list.append(padded_matrix)
            print(f"Participant {k+1}, Design matrix {g + 1} / 6x4 done ")
            print(datetime.datetime.now())

        p_design_matrix_list.append(r_design_matrix_list)
    
    models_list = []
    for i in range(len(models)):
        models[i].fit(models_run_imgs[i],design_matrices = p_design_matrix_list[i])
        models_list.append(models[i])
        print(f"model {i + 1} / {len(models)} done ")
        print(datetime.datetime.now())


    #save
    f = open('./models/first_level_all_hyp2_man_dm.pkl', 'wb') 
    pickle.dump([models_list, 
                     models_run_imgs,
                     models_events, 
                     models_confounds], f)
    f.close()


if __name__ == "__main__":
    main()