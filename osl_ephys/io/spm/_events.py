"""Classes relating to the format and storage of MEEG events and trials."""

from ._spmmeeg_utils import empty_to_zero

from logging import getLogger

import numpy as np

_logger = getLogger("py_spm")


class Trial:
    ## This works for continuous data but not epoched.
    # Epoched info is stored in list with dict per trial
    # Need to account for both
    # Should populate events all the time anyway
    # Need to populate trial info when present
    def __init__(self, label, events, onset, bad, tag, repl, sample_frequency=None):
        self.label = label
        self.events = [Event.from_dict(event_dict) for event_dict in events]
        self.onset = onset
        self.bad = bad
        self.tag = tag
        self.repl = repl

        if sample_frequency is not None:
            self.calculate_samples(sample_frequency)

    def calculate_samples(self, sample_frequency):
        for event in self.events:
            event.sample = np.floor(event.time * sample_frequency).astype(int)
            event.end_sample = np.floor(event.end_time * sample_frequency).astype(int)

    def _event_property(self, property_):
        return np.array([getattr(event, property_) for event in self.events])

    def _set_event_property(self, property_, values):
        if len(self.events) != len(values):
            _logger.warning(
                f"{len(self.events)} events, but {len(values)} values given."
            )
        for event, value in zip(self.events, values):
            setattr(event, property_, value)

    @property
    def types(self):
        return self._event_property("type")

    @property
    def values(self):
        return self._event_property("value")

    @property
    def durations(self):
        return self._event_property("duration")

    @property
    def times(self):
        return self._event_property("time")

    @property
    def offsets(self):
        return self._event_property("offset")

    @property
    def end_times(self):
        return self._event_property("end_time")

    @property
    def samples(self):
        return self._event_property("sample")

    @property
    def end_samples(self):
        return self._event_property("end_sample")

    @property
    def good_samples(self):
        return self._event_property("good_sample")

    @good_samples.setter
    def good_samples(self, values):
        self._set_event_property("good_sample", values)

    @property
    def good_end_samples(self):
        return self._event_property("good_end_sample")

    @good_end_samples.setter
    def good_end_samples(self, values):
        return self._set_event_property("good_end_sample", values)

    @property
    def trial_starts(self):
        return self._event_property("trial_start")

    @trial_starts.setter
    def trial_starts(self, values):
        return self._set_event_property("trial_start", values)


class Event:
    def __init__(self, type_, value, duration, time, offset):
        self.type = type_
        self.value = value
        self.duration = empty_to_zero(duration)

        if "artefact" in self.type.lower():
            self.duration *= 1000

        self.time = time
        self.offset = offset
        self.end_time = time + self.duration / 1000

        self.sample = -1
        self.end_sample = -1

        self.good_sample = -1
        self.good_end_sample = -1

        self.trial_start = -1
        self.trial_end = -1

    @classmethod
    def from_dict(cls, event_dict):
        if "type" in event_dict:
            event_dict["type_"] = event_dict.pop("type")
        return cls(**event_dict)

    def to_dict(self):
        return {
            "type_": self.type,
            "value": self.value,
            "duration": self.duration,
            "time": self.time,
            "offset": self.offset,
        }

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(type_='{self.type}', "
            f"value={self.value}, "
            f"duration={self.duration}, "
            f"time={self.time}, "
            f"offset={self.offset})"
        )
