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


#functions
def label_loss(row):
   if row["loss"] == -50:
      return "N"
   if row["loss"] == -250:
      return "L"
   if row["loss"] == -1250:
      return "L"

def label_total(row):
    if row["total"]  < 0:
      return "Negative"
    else :
      return "Positive"
    
def concat_events(list_of_runs):
    #concatine events and confounds
    event_list = []
    for ii in list_of_runs:
        event_list.append(ii)
    
    #recalc onset
    for i, element in enumerate(event_list):
        element.index = ((element.index + 80*i).to_list())
        element["onset"] = ((element["onset"] + 600*i).to_list())
    
    concat_list = pd.concat(event_list)
    return concat_list

def concat_confounds(list_of_runs):
    #concatine events and confounds
    event_list = []
    for ii in list_of_runs:
        event_list.append(ii)
    
    #recalc onset
    for i, element in enumerate(event_list):
        element.index = ((element.index + 600*i).to_list())
        
    concat_list = pd.concat(event_list)
    return concat_list

#main
def main():
    ####################### get eveerything
    derivatives_dir = os.path.join("/work","BIDS_2024E","derivatives")
    data_dir= os.path.join("/work","BIDS_2024E")
    # Name for experiment in the BIDS directory
    task_label = 'boldiowa'
    #Run the function that can gather all the needed info from a BIDS folder
    models, models_run_imgs, models_events, models_confounds = \
        first_level_from_bids(
            data_dir, task_label, derivatives_folder=derivatives_dir, n_jobs=-1, verbose=0, minimize_memory = False,
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
            #add new columns, separate losses and categorize them
            events2 = events2[events2["trial_type"].isin(["loss","neutral"])]
            events2["loss_cat"] = events2.apply(label_loss, axis = 1)
            events2["total_cat"] = events2.apply(label_total, axis = 1)
            
            events2["trial_type"] = events2["loss_cat"] + "_" +  events2["total_cat"]
            events2=events2[events_sub]
            events1[i]=events2
            
        models_events[ii][:]=events1
        

    ########################checkpoint
    print("preproc done")
    ######################## concatinate because of not all events happen across runs
    #concatinate scan image for participants
    concat_image_list = []
    concat_e_list = []
    concat_c_list = []
    for each_image in models_run_imgs:
        concat_img = nilearn.image.concat_imgs(each_image)
        concat_image_list.append(concat_img)
        
        image_ind = models_run_imgs.index(each_image)
        print(f"concat image {image_ind + 1} / {len(models_run_imgs)} done ")
        print(datetime.datetime.now())

    
    for each_events in models_events:
        concat_e = concat_events(each_events)
        concat_e_list.append(concat_e)

    for each_confounds in models_confounds:
        concat_c = concat_confounds(each_confounds)
        concat_c = concat_c.fillna(0)
        concat_c_list.append(concat_c)
        
    ########################checkpoint
    print("concat done")

    # Get data and model 
    models_list = []
    for i in range(len(models)):
        models[i].fit(concat_image_list[i],concat_e_list[i],concat_c_list[i])
        models_list.append(models[i])

        print(f"model {i} / {len(models)} done ")
        print(datetime.datetime.now())
    
    
    ########################checkpoint
    print("modeling done")
    #save model

    f = open('./models/first_level_all_quick.pkl', 'wb') 
    pickle.dump([models_list, 
                 concat_image_list,
                 concat_e_list, 
                 concat_c_list], f)
    f.close()
if __name__ == '__main__':
    main()


