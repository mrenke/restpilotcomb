import cortex
from nilearn import image
import numpy as np
import os.path as op
import nibabel as nib

def main():
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

    ## read in all the files in the right format
    grad_n = 1
    hemi = 'L'
    file = op.join(bids_folder_grad, 'derivatives', 'gradients', f'sub-{sub}', f'ses-{ses_grad}',
        f'sub-{sub}_ses-{ses_grad}_task-rest_space-fsaverage_hemi-{hemi}_grad{grad_n}_noParcel{removedLabels}.surf.gii')
    im1L = nib.load(file)


    hemi = 'L'
    file = op.join(bids_folder_grad, 'derivatives', 'gradients', f'sub-{sub}', f'ses-{ses_grad}',
        f'sub-{sub}_ses-{ses_grad}_task-rest_space-fsaverage_hemi-{hemi}_grad{grad_n}_noParcel{removedLabels}.surf.gii')
    im1R = nib.load(file)


    grad_n = 2
    hemi = 'L'
    file = op.join(bids_folder_grad, 'derivatives', 'gradients', f'sub-{sub}', f'ses-{ses_grad}',
        f'sub-{sub}_ses-{ses_grad}_task-rest_space-fsaverage_hemi-{hemi}_grad{grad_n}_noParcel{removedLabels}.surf.gii')
    im2L = nib.load(file)


    hemi = 'L'
    file = op.join(bids_folder_grad, 'derivatives', 'gradients', f'sub-{sub}', f'ses-{ses_grad}',
        f'sub-{sub}_ses-{ses_grad}_task-rest_space-fsaverage_hemi-{hemi}_grad{grad_n}_noParcel{removedLabels}.surf.gii')
    im2R = nib.load(file)


    ## 
    grad1 = cortex.dataset.Vertex(np.concatenate([im1L.agg_data(), im1R.agg_data()], axis=0), subject=subject,cmap='viridis_r' )
    grad2 = cortex.dataset.Vertex(np.concatenate([im2L.agg_data(), im2R.agg_data()], axis=0), subject=subject, cmap='viridis_r')

    ds1 = cortex.Dataset(grad1=grad1)
    ds2 = cortex.Dataset(grad2=grad2)

    cortex.webshow(ds1) 
    #cortex.webshow(ds2) 

if __name__ == '__main__':
    #parser = argparse.ArgumentParser()

    main()