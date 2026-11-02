import unittest

import os
import glob
import shutil
import tempfile
import numpy as np
from pathlib import Path


class TestFileInputs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Temp dir for some dummy input files
        cls.test_dir = tempfile.mkdtemp()

        # Define some dummy files - including some missing data and mistakes
        cls.fnames = [
            "sub-001_task-rest_meg-raw.fif",
            "sub-001_task-read_meg-raw.fif",
            "sub-002_task-rest_meg-raw.fif",
            "sub-002_task-read_meg-raw.fif",
            "sub-003_task-rest_meg-raw.fif",
            "sub-004_task-read_meg-raw.fif",
            "sub-006_task-RSET_meg-raw.fif",
            "sub-006_task-rest_meg-raw.fif",
        ]

        # Make some files
        cls.file_inputs = []
        for fname in cls.fnames:
            name = os.path.join(cls.test_dir, fname)
            Path(name).touch()  # Create empty file
            cls.file_inputs.append(name)

        # Now list files with no mistakes
        cls.fnames_test = [
            "sub-001_task-rest_meg-raw.fif",
            "sub-001_task-read_meg-raw.fif",
            "sub-002_task-rest_meg-raw.fif",
            "sub-002_task-read_meg-raw.fif",
            "sub-003_task-rest_meg-raw.fif",
            "sub-003_task-read_meg-raw.fif",
            "sub-004_task-rest_meg-raw.fif",
            "sub-004_task-read_meg-raw.fif",
            "sub-005_task-rest_meg-raw.fif",
            "sub-005_task-read_meg-raw.fif",
            "sub-006_task-rest_meg-raw.fif",
            "sub-006_task-rest_meg-raw.fif",
        ]

        # Make some files
        cls.file_inputs_test = []
        for fname in cls.fnames_test:
            name = os.path.join(cls.test_dir, fname)
            cls.file_inputs_test.append(name)
        return cls

    @classmethod
    def tearDownClass(cls):
        # Auto-run at the end - remove dir
        shutil.rmtree(cls.test_dir)

    def test_find_files(self):
        # This is basically testing that the temp-files exist properly
        import glob

        fnames = glob.glob(os.path.join(self.test_dir, "sub-*_task-*_meg-raw.fif"))
        assert len(fnames) == 8

    def test_process_file_inputs(self):
        from .. import io

        # Tests with 'correct' file inputs
        infiles, outnames, goods = io.process_file_inputs(self.file_inputs)

        # infiles should be same as input file list
        self.assertListEqual(infiles, self.file_inputs)

        # Output names should be inputs with file extension stripped
        outnames_test = [out[:-4] for out in self.fnames]
        self.assertListEqual(outnames, outnames_test)

        # All files exist, so should be good
        assert np.sum(goods) == len(self.fnames)

    def test_process_file_inputs_with_missing(self):
        from .. import io

        # Tests with 'correct' file inputs
        infiles, outnames, goods = io.process_file_inputs(self.file_inputs_test)

        # infiles should be same as input file list
        self.assertListEqual(infiles, self.file_inputs_test)

        # Output names should be inputs with file extension stripped
        outnames_test = [out[:-4] for out in self.fnames_test]
        self.assertListEqual(outnames, outnames_test)

        # Four files are missing
        assert np.sum(goods) == len(self.fnames_test) - 4

    def test_process_file_inputs_with_regular_expression(self):
        from .. import io

        # Tests with regular expression for all subjects
        reg = os.path.join(self.test_dir, "sub-*_task-*_meg-raw.fif")
        infiles, outnames, goods = io.process_file_inputs(reg)
        assert len(infiles) == 8

        # Tests with regular expression for all subjects 'read' dataset
        reg = os.path.join(self.test_dir, "sub-*_task-read_meg-raw.fif")
        infiles, outnames, goods = io.process_file_inputs(reg)
        assert len(infiles) == 3

        # Tests with regular expression for subb-006
        reg = os.path.join(self.test_dir, "sub-006_task-*_meg-raw.fif")
        infiles, outnames, goods = io.process_file_inputs(reg)
        assert len(infiles) == 2

    def test_process_file_inputs_with_single_path(self):
        from .. import io

        # BROKEN - we only touched a these files so they don't throw a
        # UnicodeDecodeError and trigger glob, need to touch or make a fake
        # binary file?

        # Single input that does exist
        infiles, outnames, goods = io.process_file_inputs(self.file_inputs_test[2])
        # assert(np.sum(goods) == 1)

        # Single input that does not exist
        infiles, outnames, goods = io.process_file_inputs(self.file_inputs_test[10])
        # assert(np.sum(goods) == 0)


class TestStudyClass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Temp dir for some dummy input files
        cls.test_dir = tempfile.mkdtemp()

        # Define some dummy files - including some missing data and mistakes
        cls.fnames = [
            "sub-001_task-rest_meg-raw.fif",
            "sub-001_task-read_meg-raw.fif",
            "sub-002_task-rest_meg-raw.fif",
            "sub-002_task-read_meg-raw.fif",
            "sub-003_task-rest_meg-raw.fif",
            "sub-004_task-read_meg-raw.fif",
            "sub-006_task-RSET_meg-raw.fif",
            "sub-006_task-rest_meg-raw.fif",
        ]

        # Make some files
        cls.file_inputs = []
        for fname in cls.fnames:
            name = os.path.join(cls.test_dir, fname)
            Path(name).touch()  # Create empty file
            cls.file_inputs.append(name)

        # Now list files with no mistakes
        cls.fnames_test = [
            "sub-001_task-rest_meg-raw.fif",
            "sub-001_task-read_meg-raw.fif",
            "sub-002_task-rest_meg-raw.fif",
            "sub-002_task-read_meg-raw.fif",
            "sub-003_task-rest_meg-raw.fif",
            "sub-003_task-read_meg-raw.fif",
            "sub-004_task-rest_meg-raw.fif",
            "sub-004_task-read_meg-raw.fif",
            "sub-005_task-rest_meg-raw.fif",
            "sub-005_task-read_meg-raw.fif",
            "sub-006_task-rest_meg-raw.fif",
            "sub-006_task-rest_meg-raw.fif",
        ]

        # Make some files
        cls.file_inputs_test = []
        for fname in cls.fnames_test:
            name = os.path.join(cls.test_dir, fname)
            cls.file_inputs_test.append(name)
        return cls

    @classmethod
    def tearDownClass(cls):
        # Auto-run at the end - remove dir
        shutil.rmtree(cls.test_dir)

    def test_simple_study(self):
        from ..io import Study

        fbase = os.path.join(self.test_dir, "{subj}_task-{task}_meg-{preproc}.fif")

        st = Study(fbase)
        assert len(st.match_files) == len(self.fnames)

        assert len(st.get(task="rest")) == 4
        assert len(st.get(task="read")) == 3
        assert len(st.get(task="RSET")) == 1
        assert len(st.get(task="counting")) == 0

        assert len(st.get(preproc="raw")) == len(self.fnames)
        assert len(st.get(preproc="sss")) == 0
