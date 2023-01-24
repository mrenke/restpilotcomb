# find out why pycortex view (from combine_rest-grad...py) does not show rest_grad1 properly (shows real-grad2 for both try_load_from_file-grad1 & grad2)
#%%
import os.path as op
import numpy as np
import nibabel as nib
from brainspace.datasets import load_fsa5
from brainspace.plotting import plot_hemispheres

surf_lh, surf_rh = load_fsa5()

bids_folder = '/Users/mrenke/data/resting-state_ZI-2019-pilot'

sub = '01'
ses = 1
removedLabels = ''

target_dir = op.join(bids_folder, 'derivatives', 'gradients', f'sub-{sub}', f'ses-{ses}')

#%%
grad_fa = []
for n_grad in [1,2]:
    grad = np.load(op.join(target_dir, f'grad{n_grad}_noParcel{removedLabels}.npy'))
    grad_fa.append(grad)    

plot_hemispheres(surf_lh, surf_rh, array_name=grad_fa, size=(1200, 400), cmap='viridis_r',
                 color_bar=True, label_text=['Grad1', 'Grad2'], zoom=1.5)

#    this is correct !


#%%

grad_fi =  [] # fm - from-image should be ndarray (20484,)

for n_grad in [1,2]:
    ar = []
    for i, hemi in enumerate(['L', 'R']):    

        fn =  op.join(target_dir, f'sub-{sub}_ses-{ses}_task-rest_space-fsaverage5_hemi-{hemi}_grad{n_grad}_noParcel{removedLabels}.surf.gii')
        im = nib.load(fn)
        ar.append( im.agg_data())

    ar = np.concatenate((ar[0],ar[1]))
    grad_fi.append(ar)    
    

plot_hemispheres(surf_lh, surf_rh, array_name=grad_fi, size=(1200, 400), cmap='viridis_r',
                 color_bar=True, label_text=['Grad1', 'Grad2'], zoom=1.5)

#    this is ALSO correct !

#%% --> needs to be after fsaverage5*.sur.gii files 

# --> fsaverage*.sur.gii files  ?
# inspect 

grad_fi_av =  [] # fm - from-image should be ndarray (20484,)

for n_grad in [1,2]:
    ar = []
    for i, hemi in enumerate(['L', 'R']):    

        fn =  op.join(target_dir, f'sub-{sub}_ses-{ses}_task-rest_space-fsaverage_hemi-{hemi}_grad{n_grad}_noParcel{removedLabels}.surf.gii')
        im = nib.load(fn)
        ar.append( im.agg_data())

    ar = np.concatenate((ar[0],ar[1]))
    grad_fi_av.append(ar)    
    
# %%
ses_grad = ses
bids_folder_grad = bids_folder

ims = [[None]*2]*2
grad_fi_av_s =  []
for grad_n_i, grad_n in enumerate([1,2]):
    ar = []
    for hemi_i, hemi in enumerate(['L', 'R']):  
        file = op.join(bids_folder_grad, 'derivatives', 'gradients', f'sub-{sub}', f'ses-{ses_grad}',
            f'sub-{sub}_ses-{ses_grad}_task-rest_space-fsaverage_hemi-{hemi}_grad{grad_n}_noParcel{removedLabels}.surf.gii')
        
        im = nib.load(file)
        ar.append( im.agg_data())
        ims[grad_n_i][hemi_i] = im

    ar = np.concatenate((ar[0],ar[1]))
    grad_fi_av_s.append(ar) 

# problem: ims[0][0].agg_data() == ims[1][0].agg_data()  == grad_fi_av[1]

# %%
file = op.join(bids_folder_grad, 'derivatives', 'gradients', f'sub-{sub}', f'ses-{ses_grad}',
            f'sub-{sub}_ses-{ses_grad}_task-rest_space-fsaverage_hemi-{hemi}_grad{grad_n}_noParcel{removedLabels}.surf.gii')
        
im = nib.load(file)

grad1 = cortex.dataset.Vertex(np.concatenate([ims[0][0].agg_data(), ims[0][1].agg_data()], axis=0), subject=subject,cmap='viridis_r' )
grad2 = cortex.dataset.Vertex(np.concatenate([ims[1][0].agg_data(), ims[1][1].agg_data()], axis=0), subject=subject, cmap='viridis_r')

#%%
ims = [[None]*2]*2
grad_n = 1
hemi = 'L'
file = op.join(bids_folder_grad, 'derivatives', 'gradients', f'sub-{sub}', f'ses-{ses_grad}',
    f'sub-{sub}_ses-{ses_grad}_task-rest_space-fsaverage_hemi-{hemi}_grad{grad_n}_noParcel{removedLabels}.surf.gii')
im = nib.load(file)
ims[0][0] = im

hemi = 'L'
file = op.join(bids_folder_grad, 'derivatives', 'gradients', f'sub-{sub}', f'ses-{ses_grad}',
    f'sub-{sub}_ses-{ses_grad}_task-rest_space-fsaverage_hemi-{hemi}_grad{grad_n}_noParcel{removedLabels}.surf.gii')
im = nib.load(file)
ims[0][1] = im

grad_n = 2
hemi = 'L'
file = op.join(bids_folder_grad, 'derivatives', 'gradients', f'sub-{sub}', f'ses-{ses_grad}',
    f'sub-{sub}_ses-{ses_grad}_task-rest_space-fsaverage_hemi-{hemi}_grad{grad_n}_noParcel{removedLabels}.surf.gii')
im = nib.load(file)
ims[1][0] = im

hemi = 'L'
file = op.join(bids_folder_grad, 'derivatives', 'gradients', f'sub-{sub}', f'ses-{ses_grad}',
    f'sub-{sub}_ses-{ses_grad}_task-rest_space-fsaverage_hemi-{hemi}_grad{grad_n}_noParcel{removedLabels}.surf.gii')
im = nib.load(file)
ims[1][1] = im

# %%

grad_n = 1
hemi = 'L'
file = op.join(bids_folder_grad, 'derivatives', 'gradients', f'sub-{sub}', f'ses-{ses_grad}',
    f'sub-{sub}_ses-{ses_grad}_task-rest_space-fsaverage_hemi-{hemi}_grad{grad_n}_noParcel{removedLabels}.surf.gii')
im1L = nib.load(file)
hemi = 'R'
file = op.join(bids_folder_grad, 'derivatives', 'gradients', f'sub-{sub}', f'ses-{ses_grad}',
    f'sub-{sub}_ses-{ses_grad}_task-rest_space-fsaverage_hemi-{hemi}_grad{grad_n}_noParcel{removedLabels}.surf.gii')
im1R = nib.load(file)


grad_n = 2
hemi = 'L'
file = op.join(bids_folder_grad, 'derivatives', 'gradients', f'sub-{sub}', f'ses-{ses_grad}',
    f'sub-{sub}_ses-{ses_grad}_task-rest_space-fsaverage_hemi-{hemi}_grad{grad_n}_noParcel{removedLabels}.surf.gii')
im2L = nib.load(file)
hemi = 'R'
file = op.join(bids_folder_grad, 'derivatives', 'gradients', f'sub-{sub}', f'ses-{ses_grad}',
    f'sub-{sub}_ses-{ses_grad}_task-rest_space-fsaverage_hemi-{hemi}_grad{grad_n}_noParcel{removedLabels}.surf.gii')
im2R = nib.load(file)



grad1 = cortex.dataset.Vertex(np.concatenate([im1L.agg_data(), im1R.agg_data()], axis=0), subject=subject,cmap='viridis_r' )
grad2 = cortex.dataset.Vertex(np.concatenate([im2L.agg_data(), im2R.agg_data()], axis=0), subject=subject, cmap='viridis_r')


# thsi workds, finally !!!!!