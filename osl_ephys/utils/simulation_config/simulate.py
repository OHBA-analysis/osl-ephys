import os
import mne
import sails
import numpy as np


def simulate_data(model, num_samples=1000, num_realisations=1, use_cov=True):
    num_sources = model.nsignals

    # Preallocate output
    Y = np.zeros((num_sources, num_samples, num_realisations))

    for ep in range(num_realisations):

        # Create driving noise signal
        Y[:, :, ep] = np.random.randn(num_sources, num_samples)

        if use_cov:
            C = np.linalg.cholesky(model.resid_cov)
            Y[:, :, ep] = Y[:, :, ep].T.dot(C).T

        # Main Loop
        for t in range(model.order, num_samples):
            for p in range(1, model.order):
                Y[:, t, ep] -= -model.parameters[:, :, p].dot(Y[:, t-p, ep])
    return Y


def simulate_raw_from_template(sim_samples, bad_segs=None):

    basedir = os.path.dirname(os.path.realpath(__file__))
    info = mne.io.read_info(os.path.join(basedir, 'megin_template_info.fif'))

    Y = np.zeros((306, sim_samples))
    for mod in ['mag', 'grad']:
        red_model = sails.AbstractLinearModel()
        fname = 'reduced_mvar_params_{0}.npy'.format(mod)
        red_model.parameters = np.load(os.path.join(basedir, fname))
        fname = 'reduced_mvar_residcov_{0}.npy'.format(mod)
        red_model.resid_cov = np.load(os.path.join(basedir, fname))
        red_model.delay_vect = np.arange(20)
        fname = 'reduced_mvar_pcacomp_{0}.npy'.format(mod)
        pcacomp = np.load(os.path.join(basedir, fname))

        Xsim = simulate_data(red_model, num_samples=sim_samples) * 2e-12
        Xsim = pcacomp.T.dot(Xsim[:,:,0])[:,:,None]  # back to full space


        Y[mne.pick_types(info, meg=mod), :] = Xsim[:, :, 0]


    sim = mne.io.RawArray(Y, info)
    sim.info['sfreq'] = 150

    if bad_segs is not None:
        for mod in ['mag', 'grad']:
            mne.pick_types

    return sim


def simulate_rest_mvar(raw, sim_samples,
                       mvar_pca=32, mvar_order=12,
                       picks=None, modalities=None, drop_dig=False):
    """Best used on low sample rate data <200Hz. fiff only for now."""

    if modalities is None:
        modalities = ['mag', 'grad']

    # Fit model and simulate data
    Y = np.zeros((raw.info['nchan'], sim_samples))
    for mod in modalities:
        X = raw.get_data(picks=mod)
        X = X[:, 5000:45000] * 1e12

        red_model, full_model, pca = sails.modelfit.pca_reduced_fit(X, np.arange(mvar_order), mvar_pca)

        scale = X.std() / 1e12
        Xsim = simulate_data(red_model, num_samples=sim_samples) * scale
        Xsim = pca.components.T.dot(Xsim[:,:,0])[:,:,None]  # back to full space

        Y[mne.pick_types(raw.info, meg=mod), :] = Xsim[:, :, 0]

    # Create data info for simulated object
    info = mne.io.anonymize_info(raw.info.copy())
    info['description'] = 'osl-ephys Simulated Dataset'
    info['experimentor'] = 'osl-ephys'
    info['proj_name'] = 'osl_simulate'
    info['subject_info'] = {'id': 0, 'first_name': 'osl-ephys', 'last_name': 'Simulated Data'}
    if drop_dig:
        info.pop('dig')

    if picks is None:
        picks = {'meg': True, 'eeg': False,
               'eog': False, 'ecg': False,
               'stim': False, 'misc': False}

    info = mne.pick_info(info, mne.pick_types(info, **pks))

    sim = mne.io.RawArray(Y, info)

    return sim
