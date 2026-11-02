# Conda environments

One environment file per platform, named as conda names them: `conda info |
grep platform` shows yours. Each installs the compiled dependencies (Qt,
VTK, ONNX Runtime, ...) from conda-forge and osl-ephys itself from PyPI, which
is the closest to what a conda package of osl-ephys would install.

- `linux-64.yml`: Linux on x86-64, with the software-rendering (OSMesa) VTK
  as the osl-dynamics conda package has, so the 3D plots render on servers
  without a display (set `PYVISTA_OFF_SCREEN=true` there).
- `linux-aarch64.yml`: Linux on ARM64 (no OSMesa VTK on conda-forge; without
  a display run under `xvfb-run`).
- `osx-64.yml`: Intel Macs.
- `osx-arm64.yml`: Apple silicon Macs.

There is no Windows file: osl-ephys runs under WSL there, with `linux-64.yml`,
as source reconstruction needs FSL.

If you do not already have conda, install
[Miniforge](https://conda-forge.org/download/), which provides `conda` and
`mamba`:

```
curl -LO "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
rm Miniforge3-$(uname)-$(uname -m).sh
```

Then create and activate the environment:

```
mamba env create -f envs/<platform>.yml
conda activate osle
```

See the [installation instructions](https://osl-ephys.readthedocs.io/en/latest/install.html)
for the alternatives.
