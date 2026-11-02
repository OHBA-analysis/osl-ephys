"""Classes for loading SPM files."""

from dataclasses import dataclass

import os
import numpy as np
from scipy.io import loadmat

from ._data import Data, Channel, chan_types, Montage
from ._events import Trial
from ._spmmeeg_utils import check_lowered_string


class SPMMEEG:
    def __init__(self, filename):
        self.filename = filename
        D = loadmat(filename, simplify_cells=True)["D"]
        self._D = D
        self.type = D["type"]
        self.nsamples = D["Nsamples"]
        self.nchannels = len(self._D["channels"])
        self.fsample = D["Fsample"]
        self.time_onset = D["timeOnset"]
        self.channels = [Channel.from_dict(channel) for channel in D["channels"]]
        self._find_dat_file()
        self.data = Data(**D["data"])
        self.fname = D["fname"]
        self.fullpath = filename
        self.path = D["path"]
        self.sensors = D["sensors"]
        self.fiducials = D["fiducials"]
        self.transform = D["transform"]
        self.condlist = D["condlist"]
        self.history = D["history"]
        self.other = D["other"]

        ## Setup trial information - events relate to all individual triggers
        # in the dataset, trials relate to user specified data segmenting.

        # Always store trials in list
        if self.type == "continuous":
            # D['trials'] is a dictionary of event information across scan
            self.trials = [Trial(**D["trials"], sample_frequency=self.fsample)]
            self.events = self.trials[0].events
        elif self.type == "single":  # Why are epochs of type 'single'
            # D['trials'] is a list of dictionaries of event information per trial
            self.trials = [
                Trial(**event, sample_frequency=self.fsample) for event in D["trials"]
            ]
            self.events = [ev for t in self.trials for ev in t.events]
        self.ntrials = len(self.trials)

        # Build some metadata lists - copying spm naming conventions
        if self.type == "single":
            self.conditions = [t.label for t in self.trials]
            self.condition_values = []
            try:
                for t in self.trials:
                    self.condition_values.append(int(_get_trial_trigger_value(t)))
            except ValueError:
                print("Could not find an integer trigger value in trial")
                print(t)
                raise
            self.condlist = np.unique(self.conditions)
        else:
            self.conditions = None
            self.condition_values = None
            self.condlist = None

        ## Setup montage structures
        self.montage = {}

        if isinstance(D["montage"]["M"], dict):
            # We have a single montage - store in list of len-1 for consistency
            self._montage_cache = [D["montage"]["M"]]
        else:
            # We have a list of montage - store it
            self._montage_cache = D["montage"]["M"]

        # Create montage objects
        for idx in range(len(self._montage_cache)):
            self.montage[idx] = Montage(**self._montage_cache[idx])
        self.current_montage = D["montage"]["Mind"]

        ## Setup sample indexing
        self.time = (
            np.linspace(0, self.nsamples / self.fsample, self.nsamples)
            - self.time_onset
        )

        self.index = np.ones(self.nsamples, dtype=bool)
        self.good_index = np.zeros(self.nsamples, dtype=int)

        self.mark_artefacts_as_bad()
        self.reindex_good_samples()

        if self.type == "continuous":
            # only reindex continuous data - assume this is fine in already epoched data
            self.reindex_event_samples()

        self.trial_definition = None

    def get_data(self, montage=None):
        """Return memorymapped data and optionally apply a montage."""
        if montage is None:
            return self.data.data
        else:
            return self.montage[montage].apply(self.data.data)

    def epoch_data(self, data):
        trial_def = self.trial_definition
        if trial_def is None:
            raise ValueError("No trials has been defined.")

        trials = self.trials
        events = check_lowered_string(self.trials.types, trial_def.event_type)

        starts = np.round(trials.good_samples) - trial_def.pre_stim * self.fsample
        starts = starts[events]

        ends = np.round(trials.good_end_samples) + trial_def.post_stim * self.fsample
        ends = ends[events]

        valid = (starts > 0) & (ends < min(self.index.sum(), data.shape[0]))

        epochs = []
        for start, end in zip(starts[valid], ends[valid]):
            epochs.append(data[start:end])

        return np.array(epochs)

    def define_trial(self, event_type, pre_stim, post_stim):
        self.trial_definition = TrialParameters(event_type, pre_stim, post_stim)

    def mark_artefacts_as_bad(self):
        # import pdb; pdb.set_trace()
        artefacts = check_lowered_string([e.type for e in self.events], "artefact")
        starts = [t.samples for t in self.trials]
        starts = np.concatenate(starts)[artefacts]
        ends = [t.end_samples for t in self.trials]
        ends = np.concatenate(ends)[artefacts]

        for start, end in zip(starts, ends):
            self.index[start:end] = False

    def _channel_property(self, property_):
        return np.array([getattr(channel, property_) for channel in self.channels])

    def full_index(self, channel_type):
        return np.ix_(self.index, self.channel_selection(channel_type))

    def reindex_good_samples(self):
        self.good_index = np.zeros_like(self.index) - 1
        self.good_index[self.index] = np.arange(self.index.sum())
        self.good_index = np.minimum(self.good_index, self.nsamples)

    def reindex_event_samples(self):
        for tr in self.trials:
            tr.good_samples = self.good_index[tr.samples]
            tr.good_end_samples = self.good_index[
                np.minimum(tr.end_samples, self.nsamples - 1)
            ]
        # self.trials.good_samples = self.good_index[self.trials.samples]

        # TODO: Sort out this hacky solution to excess samples.
        #  It might actually not be hacky, bad segments *may* extend past nsamples.
        # self.trials.good_end_samples = self.good_index[
        #    np.minimum(self.trials.end_samples, self.nsamples - 1)
        # ]

    def print_info(self):
        print("SPM M/EEG data object - Loaded by OSL")
        print("Type: {0}".format(self.type))
        print("Transform: {0}".format(self.transform))
        print("{0} conditions".format(len(self.condlist)))
        print("{0} channels".format(self.nchannels))
        print("{0} samples/trial".format(self.nsamples))
        print("{0} trials".format(self.ntrials))
        print("Sampling frequency {0}Hz".format(self.fsample))
        print("Loaded from : {0}".format(self.fullpath))

        if len(self.montage) > 0:
            print("\nMontages available : {0}".format(len(self.montage)))
            for ind, mon in self.montage.items():
                print("\t{0} : {1}".format(ind, mon.name))

        print(
            "\nUse syntax 'X = D.get_data(montage_index)[channels, samples, trials]' to get data"
        )

    def _find_dat_file(self):
        matname = self.filename
        datname = self._D["data"]["fname"]

        dat = None
        if os.path.exists(datname):
            dat = datname
        else:
            datname2 = os.path.join(os.path.dirname(matname), os.path.basename(datname))
            print(datname2)
            if os.path.exists(datname2):
                dat = datname2
        if dat is None:
            raise FileNotFoundError(
                "Associated 'dat' file not found ({0})".format(datname)
            )
        else:
            self._D["data"]["fname"] = dat

    # SPM Style Helpers

    @property
    def size(self):
        return self.data.shape

    @property
    def chantype(self):
        return self._channel_property("type")

    def indchantype(self, channel_type):
        return np.where(np.isin(self.chantype, chan_types[channel_type]))[0]

    def indsample(self, t):
        # get index of sample given time in seconds
        return np.argmin(np.abs(self.time - t))

    def indtrial(self, cond):
        if cond not in self.condlist:
            raise ValueError(
                "Condition '{0}' not in dataset (available conditions = {1})".format(
                    cond, self.condlist
                )
            )
        return np.where([c == cond for c in self.conditions])[0]

    # Properties

    @property
    def n_good_samples(self):
        return self.index.sum()


def _get_trial_trigger_value(t):
    """Return value of first STI event in trial."""
    ind = [ev.type.find("STI") for ev in t.events][0]
    return t.values[ind]


@dataclass
class TrialParameters:
    event_type: str
    pre_stim: float
    post_stim: float
