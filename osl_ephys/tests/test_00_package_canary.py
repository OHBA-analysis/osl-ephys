import os
import unittest

class TestModuleStructure(unittest.TestCase):

    def test_module_structure(self):

        try:
            from  .. import utils
        except ImportError:
            raise Exception("Unable to import 'utils'")

        try:
            from  .. import maxfilter
        except ImportError:
            raise Exception("Unable to import 'maxfilter'")

        try:
            from  .. import preprocessing
        except ImportError:
            raise Exception("Unable to import 'preprocessing'")

        try:
            from  .. import report
        except ImportError:
            raise Exception("Unable to import 'report'")

        try:
            from  .. import source_recon
        except ImportError:
            raise Exception("Unable to import 'source_recon'")


class TestPackageData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.testdir = os.path.dirname(os.path.realpath(__file__))
        cls.osldir = os.path.abspath(os.path.join(cls.testdir, '..'))

    def test_simulatons_data(self):

        template = os.path.join(self.osldir, 'utils', 'simulation_config', 'megin_template_info.fif')
        assert(os.path.exists(template))

        for ff in ['reduced_mvar_params_mag.npy', 'reduced_mvar_residcov_mag.npy',
                   'reduced_mvar_pcacomp_mag.npy', 'reduced_mvar_params_grad.npy',
                   'reduced_mvar_residcov_grad.npy', 'reduced_mvar_pcacomp_grad.npy']:

            template = os.path.join(self.osldir, 'utils', 'simulation_config', ff)
            assert(os.path.exists(template))

    def test_channel_data(self):

        template = os.path.join(self.osldir, 'utils', 'neuromag306_info.yml')
        assert(os.path.exists(template))

    def test_parcellation_data(self):

        to_check = ['WTA_fMRI_parcellation_ds2mm.nii.gz',
                    'WTA_fMRI_parcellation_ds8mm.nii.gz',
                    'dk_cortical.nii.gz',
                    'dk_full.nii.gz',
                    'fMRI_parcellation_ds2mm.nii.gz',
                    'fMRI_parcellation_ds8mm.nii.gz',
                    'fmri_d100_parcellation_with_PCC_reduced_2mm.nii.gz',
                    'fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz',
                    'fmri_d100_parcellation_with_PCC_tighterMay15_v2_2mm.nii.gz',
                    'fmri_d100_parcellation_with_PCC_tighterMay15_v2_6mm_exclusive.nii.gz',
                    'fmri_d100_parcellation_with_PCC_tighterMay15_v2_8mm.nii.gz',
                    'giles_39_binary.nii.gz']

        for ff in to_check:
            template = os.path.join(self.osldir, 'source_recon', 'files', ff)
            assert(os.path.exists(template))
