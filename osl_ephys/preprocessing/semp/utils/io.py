import os
import pickle

import mne


def load_pkl(file_path):
    """Load data using pickle.

    Parameters
    ----------
    file_path : str
        File path containing the data to be loaded.

    Returns
    -------
    data : Any
        Loaded data.
    """

    with open(file_path, "rb") as input_path:
        data = pickle.load(input_path)
    return data

def save_pkl(data, file_path):
    """Save data using pickle.

    Parameters
    ----------
    data : Any
        Data object to be saved.
    file_path : str
        File path where the data will be saved.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "wb") as save_path:
        pickle.dump(data, save_path)


class _EEGDictMeta(type):
    """Metaclass plumbing so ``SingletonEEG(...)`` dispatches to ``.get()``.
    Internal only -- users go through ``SingletonEEG``, never this."""
    def __call__(cls, *args, **kwargs):
        return cls.get(*args, **kwargs)

class SingletonEEG(metaclass=_EEGDictMeta):
    """Singleton class for loading an example EEG file once and reusing it.
    This class is used to avoid loading the EEG file multiple times, which can be time-consuming.

    Example usage: some datasets lack a ``dev_head_t`` matrix (or a montage), so
    borrow it from a template recording once and reuse it for every subject::

        tmplt = SingletonEEG(path)                                   # loads once
        dataset['raw'].info['dev_head_t'] = tmplt.info['dev_head_t']

    (The next ``SingletonEEG(path)`` returns the cached Raw, not a reload.)
    """

    _raw = None
    _loaded_pth = None

    @classmethod
    def get(cls, file_path=None, safe_reload=True, preload=False):
        """Get the singleton instance of an example eeg file.

        Parameters:
        file_path (str): Path to the EEG file to load.
        keys (list): List of keys to extract from the EEG file. any key should be a property extractable via raw.key, e.g. ['info', 'first_samp', 'ch_names', '_data', 'annotations'].
        info_keys (list): List of keys to extract from the EEG info.
        safe_reload (bool): If True, when loading this dict again, it will check if the file path and keys are either none or the same as the one loaded before. If not, would raise an error.

        Returns:
        ExampleEEGSingletonLoader: The singleton instance of a dictionary, containing all keys from the eeg raw class.
        """
        if cls._raw is None:
            if file_path is None:
                raise ValueError("SingletonEEG needs a file_path on first load.")
            cls._raw = mne.io.read_raw(file_path, preload=preload)   # format sniffed by MNE
            cls._loaded_pth = file_path
        elif safe_reload and file_path is not None:
            if file_path != cls._loaded_pth:
                raise ValueError(f"File path {file_path} does not match the previously loaded file path {cls._loaded_pth}. Use the same file path or None to reload the singleton.")

        return cls._raw