# brainspace_analysis_01

# nurmefields env

# run without parcellation

#%% create gradients from resting state dataset

import numpy as np
import nibabel as nib
import os.path as op

bids_folder = '/Users/mrenke/data/resting-state_ZI-2019-pilot'
func_folder = op.join(bids_folder, 'derivatives', 'fmriprep_surface', 'derivatives/sub-01/ses-1/func/')

sub = '01'
ses = 1

removedLabels = ''

target_dir = op.join(bids_folder, 'derivatives', 'gradients', f'sub-{sub}', f'ses-{ses}')

if not op.exists(target_dir):
    os.makedirs(target_dir)

# %% 1. load in data as timeseries
filename = op.join(func_folder, 'sub-01_ses-1_task-rest_hemi-{}_space-fsaverage5_bold.func.gii')
timeseries = [None] * 2
for i, h in enumerate(['L', 'R']):
    timeseries[i] = nib.load(filename.format(h)).agg_data()
timeseries = np.vstack(timeseries)

#%% 2. load confounds and Do the confound regression
import pandas as pd
from nilearn import signal

conf_file = 'sub-01_ses-1_task-rest_desc-confounds_timeseries.tsv'
fmriprep_confounds_include = ['global_signal', 'dvars', 'framewise_displacement', 'trans_x',
                                  'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                                  'a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03', 'cosine00', 'cosine01', 'cosine02'
                                  ]
fmriprep_confounds = op.join(func_folder, conf_file)
fmriprep_confounds = pd.read_table(fmriprep_confounds)[fmriprep_confounds_include] 
fmriprep_confounds= fmriprep_confounds.fillna(method='bfill')


clean_ts = signal.clean(timeseries.T, confounds=fmriprep_confounds).T

#%% 3. extract the cleaned timeseries 
import numpy as np
from nilearn import datasets
from brainspace.utils.parcellation import reduce_by_labels

# Fetch surface atlas
atlas = datasets.fetch_atlas_surf_destrieux()

# Remove non-cortex regions
regions = atlas['labels'].copy()
masked_regions = [b'Medial_wall', b'Unknown']
masked_labels = [regions.index(r) for r in masked_regions]
for r in masked_regions:
    regions.remove(r)

# Build Destrieux parcellation and mask
labeling = np.concatenate([atlas['map_left'], atlas['map_right']])
mask = ~np.isin(labeling, masked_labels)

# Distinct labels for left and right hemispheres
lab_lh = atlas['map_left']
labeling[lab_lh.size:] += lab_lh.max() + 1

# grad1 range was weird: plt.hist(grad_noParcel[0]) --> solution: remove regions with weird eigenvector value

mask = ~np.isin(labeling, masked_labels)
mask[labeling == 118] = False 

seed_ts_noParcel = clean_ts[mask]

#%% --> Correlation Matrix & Gradients

from nilearn.connectome import ConnectivityMeasure
from brainspace.gradient import GradientMaps
from brainspace.utils.parcellation import map_to_labels

correlation_measure_noParcel = ConnectivityMeasure(kind='correlation')
correlation_matrix_noParcel = correlation_measure_noParcel.fit_transform([seed_ts_noParcel.T])[0]

gm_noParcel = GradientMaps(n_components=2, random_state=0)
gm_noParcel.fit(correlation_matrix_noParcel)

# Map gradients to original parcels
labeling_noParcel = np.arange(0,len(labeling),1,dtype = int)

grad_noParcel = [None] * 2
for i, g in enumerate(gm_noParcel.gradients_.T):
    grad_noParcel[i] = map_to_labels(g, labeling_noParcel, mask=mask, fill=np.nan)

#%% save gradients as np.array
import os

for i, n_grad  in enumerate([1,2]):
    np.save(op.join(target_dir,f'grad{n_grad}_noParcel{removedLabels}.npy'), grad_noParcel[i])


# %% visualize

from brainspace.datasets import load_fsa5
from brainspace.plotting import plot_hemispheres

surf_lh, surf_rh = load_fsa5()

plot_hemispheres(surf_lh, surf_rh, array_name=grad_noParcel, size=(1200, 400), cmap='viridis_r',
                 color_bar=True, label_text=['Grad1', 'Grad2'], zoom=1.5)

#%% create .surf.gii files with grad in fsaverage5 space

for n_grad in [1,2]:
    grad = np.load(op.join(target_dir, f'grad{n_grad}_noParcel{removedLabels}.npy'))
    grad = np.split(grad,2) # for i, hemi in enumerate(['L', 'R']): --> left first?!

    for i, hemi in enumerate(['L', 'R']):    

        gii_im_datar = nib.gifti.gifti.GiftiDataArray(data=grad[i])
        gii_im = nib.gifti.gifti.GiftiImage(darrays= [gii_im_datar])

        out_file = op.join(target_dir, f'sub-{sub}_ses-{ses}_task-rest_space-fsaverage5_hemi-{hemi}_grad{n_grad}_noParcel{removedLabels}.surf.gii')
        gii_im.to_filename(out_file) # https://nipy.org/nibabel/reference/nibabel.spatialimages.html


#%% convert fsaverage5 to fsaverage (for later visualization)

from nipype.interfaces.freesurfer import SurfaceTransform
#removedLabels = '_no120'

target_space = 'fsaverage'

for n_grad in [1,2]:

    for i, hemi in enumerate(['L', 'R']):   

        sxfm = SurfaceTransform(subjects_dir='/Users/mrenke/data/ds-stressrisk/derivatives/freesurfer')

        grad_sub_dir = op.join(bids_folder, 'derivatives', 'gradients', f'sub-{sub}', f'ses-{ses}')

        in_file = op.join(grad_sub_dir, f'sub-{sub}_ses-{ses}_task-rest_space-fsaverage5_hemi-{hemi}_grad{n_grad}_noParcel{removedLabels}.surf.gii')
        out_file = op.join(grad_sub_dir, f'sub-{sub}_ses-{ses}_task-rest_space-{target_space}_hemi-{hemi}_grad{n_grad}_noParcel{removedLabels}.surf.gii')

        sxfm.inputs.source_file = in_file
        sxfm.inputs.out_file = out_file

        sxfm.inputs.source_subject = 'fsaverage5'
        sxfm.inputs.target_subject = target_space

        if hemi == 'L':
            sxfm.inputs.hemi = 'lh'
        elif hemi == 'R':
            sxfm.inputs.hemi = 'rh'

        r = sxfm.run()

# %% find our which areas (belonging to a label) make the gradient weird --> remove via mask
import matplotlib.pyplot as plt
labeling[ grad_noParcel[0] == max(grad_noParcel[0])]
labeling[ grad_noParcel[0] == min(grad_noParcel[0])]

# %%--------------------------------------with parcellation-----------------------------------------------


# %% 4. Calculate functional connectivity matrix & visualize

from nilearn.connectome import ConnectivityMeasure

correlation_measure = ConnectivityMeasure(kind='correlation')
correlation_matrix = correlation_measure.fit_transform([seed_ts.T])[0]

from nilearn import plotting

mat_mask = np.where(np.std(correlation_matrix, axis=1) > 0.2)[0] # # Reduce matrix size, only for visualization purposes

c = correlation_matrix[mat_mask][:, mat_mask]

# Create corresponding region names
regions_list = ['%s_%s' % (h, r.decode()) for h in ['L', 'R'] for r in regions]
masked_regions = [regions_list[i] for i in mat_mask]


corr_plot = plotting.plot_matrix(c, figure=(15, 15), labels=masked_regions,
                                 vmax=0.8, vmin=-0.8, reorder=True)

# %% 5. Run gradient analysis and visualize

from brainspace.gradient import GradientMaps

gm = GradientMaps(n_components=2, random_state=0)
gm.fit(correlation_matrix)

from brainspace.datasets import load_fsa5
from brainspace.plotting import plot_hemispheres
from brainspace.utils.parcellation import map_to_labels

# Map gradients to original parcels
grad = [None] * 2
for i, g in enumerate(gm.gradients_.T):
    grad[i] = map_to_labels(g, labeling, mask=mask, fill=np.nan)

for i in range(len(grad)):
    grad[i] = -grad[i]

# Load fsaverage5 surfaces
surf_lh, surf_rh = load_fsa5()

plot_hemispheres(surf_lh, surf_rh, array_name=grad, size=(1200, 400), cmap='viridis_r',
                 color_bar=True, label_text=['Grad1', 'Grad2'], zoom=1.5)

