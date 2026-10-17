"""Localisation error of an LCMV beamformer on evoked fields, with the
forward model from RHINO or from FreeSurfer.

The human evoked field recording and the dipole fits it is compared with
come from the beamformer comparison paper (Jaiswal et al., 2020,
https://doi.org/10.1016/j.neuroimage.2020.116797; data and scripts at
https://zenodo.org/record/3233557). With do_rhino the surfaces, the
coregistration and the forward model are computed by RHINO on the MNI grid;
otherwise MNE uses the precomputed FreeSurfer BEM and transform. The
beamformer itself is MNE's, on the evoked response of each stimulus, and
the peak of its power is compared with the fitted dipole.
"""

import mne
import numpy as np
from nilearn.image import index_img
from nilearn.plotting import plot_stat_map
from scipy.io import loadmat

from osl_ephys import rhino
from osl_ephys.session import Session

do_rhino = True

data_dir = "BeamComp_DataRepo"
raw_file = f"{data_dir}/MEG/Human_EF/multimodal_raw.fif"
subjects_dir, subject = f"{data_dir}/MRI", "BeamCompMRI"
smri_file = f"{subjects_dir}/{subject}/mri/T1.nii.gz"
trans_file = f"{subjects_dir}/{subject}/mri/brain-neuromag/sets/BeamCompMRI-amit-131118-MNEicp-trans.fif"
mri_file = f"{subjects_dir}/{subject}/mri/T1.mgz"
brain_surface = f"{subjects_dir}/{subject}/bem/watershed/{subject}_brain_surface"
dipole_fits = "multimodal_biomag_Xfit_results.mat"  # from the zenodo scripts

# Stimulus codes and the fitted dipole of each (from the paper's scripts)
event_id = {"VEF_UR": 1, "VEF_LR": 2, "AEF_Re": 3}
fitted = loadmat(dipole_fits)["multimodal_biomag_Xfit_diploc"][:, 3:6]
fitted = dict(zip(event_id, fitted))
windows = {  # control and active windows (s) by modality
    "VEF": ([-0.200, -0.050], [0.050, 0.200]),
    "AEF": ([-0.150, -0.020], [0.020, 0.150]),
}


def reject_by_variance(epochs, low=2.0, high=98.0):
    """Drop the trials whose maximum channel variance is outside the percentiles."""
    var = np.array([np.var(e, axis=1).max() for e in epochs.get_data()])
    bad = np.where((var < np.percentile(var, low)) | (var > np.percentile(var, high)))[
        0
    ]
    epochs.drop(bad, reason="variance")
    return epochs


raw = mne.io.read_raw_fif(raw_file, preload=True)
if raw.info["projs"]:
    raw.drop_channels(["MEG 0442"])
events = mne.find_events(
    raw, stim_channel="STI 014", min_duration=0.001, shortest_event=1
)
events = events[events[:, 1] == 0]
raw.pick("meg")
raw.filter(2, 95)
epochs = mne.Epochs(raw, events, event_id, -0.5, 0.5, baseline=(-0.5, 0), preload=True)

if do_rhino:
    outdir = "data"
    session = Session(
        outdir=outdir,
        id=subject,
        surfaces_dir=f"{outdir}/{subject}/surfaces",
        preproc_file=raw_file,
    )
    rhino.extract_surfaces(smri_file, session.surfaces_dir, include_nose=True)
    rhino.extract_fiducials_and_headshape_from_fif(session)
    rhino.coregister_head_and_mri(session, use_nose=True, use_headshape=True)
    rhino.plot_coregistration(session, filename=f"{session.coreg_dir}/coreg.png")
    rhino.forward_model(session, model="Single Layer", gridstep=8, mindist=4.0)
    fwd = mne.read_forward_solution(session.fwd_model_file)
    background = session.coreg.mri_file
else:
    model = mne.make_bem_model(
        subject, ico=4, conductivity=(0.33,), subjects_dir=subjects_dir
    )
    bem = mne.make_bem_solution(model)
    src = mne.setup_volume_source_space(
        subject,
        pos=5.0,
        mri=mri_file,
        surface=brain_surface,
        mindist=2.5,
        exclude=10.0,
        subjects_dir=subjects_dir,
    )
    fwd = mne.make_forward_solution(
        epochs.info,
        trans=trans_file,
        src=src,
        bem=bem,
        meg=True,
        eeg=False,
        mindist=2.5,
    )
    background = mri_file

print("Leadfield: %d sensors x %d dipoles" % fwd["sol"]["data"].shape)

# The rank of maxfiltered data is in its processing history
rank = None
if epochs.info["proc_history"]:
    rank = int(epochs.info["proc_history"][0]["max_info"]["sss_info"]["nfree"])

for stimulus in event_id:
    control, active = windows[stimulus[:3]]
    stim_epochs = reject_by_variance(epochs[stimulus])
    noise_cov = mne.compute_covariance(stim_epochs, tmin=control[0], tmax=control[1])
    data_cov = mne.compute_covariance(stim_epochs, tmin=active[0], tmax=active[1])
    evoked = stim_epochs.average().crop(*active)

    filters = mne.beamformer.make_lcmv(
        stim_epochs.info,
        fwd,
        data_cov,
        reg=0.05,
        noise_cov=noise_cov,
        pick_ori="max-power",
        rank=rank,
        weight_norm="nai",
        reduce_rank=True,
    )
    stc = abs(mne.beamformer.apply_lcmv(evoked, filters))
    power = (stc**2).sum()
    peak, _ = power.get_peak()
    estimated = fwd["src"][0]["rr"][peak] * 1000
    error = np.linalg.norm(fitted[stimulus] - estimated)
    n_active = (power.data > power.data.max() / 2).sum()
    print(f"{stimulus}: fitted {fitted[stimulus]}, estimated {np.round(estimated, 1)}")
    print(
        f"  localisation error {error:.1f} mm, point spread volume {n_active * 5.0**3:.0f} mm3"
    )

    _, t_peak = stc.get_peak()
    img = stc.as_volume(fwd["src"], dest="mri", mri_resolution=False)
    plot_stat_map(
        index_img(img, int(round((t_peak - stc.tmin) / stc.tstep))),
        background,
        threshold=stc.data.max() / 2,
        title=f"{stimulus}: error {error:.1f} mm",
        output_file=f"{stimulus}_power.png",
    )
