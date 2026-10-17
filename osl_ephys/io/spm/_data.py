"""Classes relating to the format and storage of MEEG data and sensors."""

import numpy as np
from ._spmmeeg_utils import empty_to_none

KNOWN_DTYPEIDS = {16: np.dtype("<f4")}  # FLOAT32-LE


class Data:
    def __init__(
        self, fname, dim, dtype, be, offset, pos, scl_slope, scl_inter, permission
    ):
        self.fname = fname
        self.shape = np.array(dim)
        self.dtype = dtype
        self.be = be
        self.offset = offset
        self.pos = np.array(pos)
        self.scl_slope = empty_to_none(scl_slope)
        self.scl_inter = empty_to_none(scl_inter)
        self.permission = permission

        # pythonise options
        self.dtype = KNOWN_DTYPEIDS.get(self.dtype, None)

        if self.permission == "rw":
            self.permission = "readwrite"

        # Load data as memorymap - note dimensions are flipped compared to what
        # we'd expect so transpose the result to get back to something
        # sensible.
        self.data = np.memmap(
            self.fname,
            dtype=self.dtype,
            mode=self.permission,
            shape=tuple(np.flipud(self.shape)),
        )
        if len(self.shape) == 2:
            self.data = np.transpose(self.data, (1, 0))
        else:
            self.data = np.transpose(self.data, (2, 1, 0))

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"fname='{self.fname}', "
            f"dim={self.shape.tolist()}, "
            f"dtype={self.dtype}, "
            f"be={self.be}, "
            f"offset={self.offset}, "
            f"pos={self.pos.tolist()}, "
            f"scl_slope={self.scl_slope}, "
            f"scl_inter={self.scl_inter}, "
            f"permission='{self.permission}')"
        )


chan_types = {
    "EOG": ["VEOG", "HEOG"],
    "ECG": ["EKG"],
    "REF": ["REFMAG", "REFGRAD", "REFPLANAR"],
    "MEG": ["MEGMAG", "MEGGRAD"],
    "MEGMAG": ["MEGMAG"],
    "MEGGRAD": ["MEGGRAD"],
    "MEGPLANAR": ["MEGPLANAR"],
    "MEGANY": ["MEG", "MEGMAG", "MEGGRAD", "MEGPLANAR"],
    "MEEG": ["EEG", "MEG", "MEGMAG", "MEGCOMB", "MEGGRAD", "MEGPLANAR"],
}


class Channel:
    def __init__(self, bad=None, label=None, type_=None, x=None, y=None, units=None):
        self.bad = bad
        self.label = label
        self.type = type_
        self.x = empty_to_none(x)
        self.y = empty_to_none(y)
        self.units = units

    @classmethod
    def from_dict(cls, channel_dict):
        if "X_plot2D" in channel_dict:
            channel_dict["x"] = channel_dict.pop("X_plot2D")
        if "Y_plot2D" in channel_dict:
            channel_dict["y"] = channel_dict.pop("Y_plot2D")

        return cls(
            bad=channel_dict.get("bad", None),
            label=channel_dict.get("label", None),
            type_=channel_dict.get("type", None),
            x=channel_dict.get("x", None),
            y=channel_dict.get("y", None),
            units=channel_dict.get("units", None),
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"bad={self.bad}, "
            f"label='{self.label}', "
            f"type_='{self.type}', "
            f"x={self.x}, "
            f"y={self.y}, "
            f"units='{self.units}')"
        )


class Montage:
    def __init__(
        self, name=None, tra=None, labelnew=None, labelorg=None, channels=None
    ):
        self.name = name
        self.tra = tra
        self.labelnew = labelnew
        self.labelorg = labelorg
        self.channels = channels

    def apply(self, data):
        print("Applying montage:  {0}".format(self.name))
        if data.ndim == 2:
            return self.tra.dot(data)
        elif data.ndim == 3:
            out = np.zeros((self.tra.shape[0], data.shape[1], data.shape[2]))
            for ii in range(data.shape[2]):
                out[:, :, ii] = self.tra.dot(data[:, :, ii])
            return out
