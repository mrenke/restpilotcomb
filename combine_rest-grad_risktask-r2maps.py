



import cortex
from nilearn import image
import numpy as np
import os.path as op

sub = '01'
removedLabels = '' # '' = _no118  for mrenke (sub-01 in )

# sub in pycortex
subject = 'fsaverage'
xfm = 'identity.bold'

bids_folder_r2 = '/Users/mrenke/data/ds-tmspilot' #risk-task
bids_folder_grad = '/Users/mrenke/data/resting-state_ZI-2019-pilot' #rest

base = 'encoding_model.smoothed'
ses_r2 = 2
ses_grad = 1

#%% grad data
subject = 'fsaverage'
import nibabel as nib

ims = [[None]*2]*2
for grad_n_i, grad_n in enumerate([1,2]):
    for hemi_i, hemi in enumerate(['L', 'R']):  
        file = op.join(bids_folder_grad, 'derivatives', 'gradients', f'sub-{sub}', f'ses-{ses_grad}',
            f'sub-{sub}_ses-{ses_grad}_task-rest_space-fsaverage_hemi-{hemi}_grad{grad_n}_noParcel{removedLabels}.surf.gii')

        ims[grad_n_i][hemi_i] = nib.load(file)

grad1 = cortex.dataset.Vertex(np.concatenate([ims[0][0].agg_data(), ims[0][1].agg_data()], axis=0), subject=subject,cmap='viridis_r' )
grad2 = cortex.dataset.Vertex(np.concatenate([ims[1][0].agg_data(), ims[1][1].agg_data()], axis=0), subject=subject, cmap='viridis_r')

ds1 = cortex.Dataset(grad1=grad1)
ds2 = cortex.Dataset(grad2=grad2)

cortex.webshow(ds1) 
cortex.webshow(ds2) 

# %%

#%% r2 data

r2_data =  op.join(bids_folder_r2, 'derivatives', base , f'sub-{sub}', f'ses-{ses_r2}', 'func', 
    f'sub-{sub}_ses-{ses_r2}_desc-r2.optim_space-T1w_pars.nii.gz')

r2_data = image.load_img(r2_data).get_data().T

alpha = r2_data
alpha  = (alpha > .075).astype(np.float)


r2_data_thr = r2_data.copy()
r2_data_thr[r2_data < 0.05] = np.nan

r2_surf = cortex.Volume(r2_data, subject, xfm)
r2_surf_thr = cortex.Volume(r2_data_thr, subject, xfm)


#%% prepare "fsaverage5-subject" in mambaforge/sharepycortex/db/...
from cortex import freesurfer

bids_folder = '/Users/mrenke/data/ds-stressrisk' # for freesurfer/fsaverage5
ses = 1

subject = 'fsaverage5'

freesurfer.import_subj(subject,freesurfer_subject_dir=op.join(bids_folder, 'derivatives', 'freesurfer'))

# %%
