# MaxFilter at OHBA

`maxfilter.py` MaxFilters a set of recordings on a computer with a
MaxFilter licence. The same from the terminal, with the raw files listed
one per line in `files.txt`:

```
osl-ephys maxfilter files.txt data --maxpath /neuro/bin/util/maxfilter --scanner Neo --mode multistage --tsss --headpos --movecomp
```

The `scanner` option sets the calibration files of the OHBA scanner the
data was recorded with, which depends on the recording date:

- `Neo`: "TRIUX system at OHBA-3143" in the scanner name, recordings since
  2019.
- `VectorView2`: "VectorView" in the scanner name, December 2017 to 2019.
- `VectorView`: "VectorView" in the scanner name, before December 2017.

`check_scanner.py` prints the recording date and scanner name of each
file.
