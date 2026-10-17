# osl-ephys v3 refactor plan

Written 2026-10-15 from a survey of osl-ephys (main, v2.4.0), osl-dynamics
(v3.6.2+12, commit b64c6eb0) and osl-files, plus the original todo list and the
open issues/PRs on GitHub. Tracked on the "osl-ephys major overhaul" project
board and draft PR #453 (`v3` -> `main`).

## 1. The big picture

osl-dynamics already contains a rewritten M/EEG pipeline in `osl_dynamics.meeg`
(~12k lines): `maxfilter`, `amm` (OPM), `preproc`, `session`, `rhino`,
`source_recon` (LCMV + bilateral + virtual electrodes), `parcellation`,
`parallel`, `report`, plus `data.sign_flipping` and `files` (pooch download from
osl-files). It is released on PyPI and conda-forge (`osl-dynamics-base` has no
TensorFlow), uses pyproject + setuptools_scm, trusted publishing, codespell and
black. Most items on the todo list are "make osl-ephys look like that".

osl-ephys today (24k lines) is organised around a YAML config API
(`run_proc_batch` / `run_src_batch` and the `run_mne_*` / `run_osl_*` wrappers),
dask for parallelism, jinja2 per-subject HTML reports, bundled parcellations and
simulation data, setup.py packaging, manual releases, and a CI job pinned to
Python 3.8 with actions v2. Issue #449 reports the tests currently fail.

What osl-ephys has that osl-dynamics does not:

- the YAML config / wrapper API and `osl_preproc`, `osl_func`, `osl_maxfilter`,
  `osl_ica_label`, `osl_ica_apply` CLIs
- `glm` (sensor GLM spectrum / epochs via sails + glmtools)
- `plot_ica` interactive ICA labelling GUI
- `minimum_norm` (volumetric MNE; PR #445 fixes it) and `freesurfer_utils`
- `utils.spmio` (SPM MEEG I/O), `utils.study` (`Study` file finder), OPM
  conversion utilities, `utils.simulate` (MVAR simulation from a MEGIN template)
- the preproc report's richer plots (flowchart, ICA, artefact scan, digitisation,
  head movement, events, custom figures)
- EEG sensor moving example, CTF with sMRI fiducials example

## 2. Decisions to make before any code moves

These change what the later phases look like, so settle them first.

D1. Relationship with osl-dynamics (the one that matters most)

  Working decision (2026-10-15): port the `osl_dynamics.meeg` code into
  osl-ephys so that osl-ephys is a standalone package with the same
  dependency set as osl-dynamics minus TensorFlow. Section 8 lists what is
  ported, module by module.

  osl-dynamics stays as it is: it keeps its own `meeg` copy and does not
  import from osl-ephys (decided 2026-10-15). Whether that changes later is
  a separate decision for osl-dynamics; nothing in v3 assumes it. Porting
  rather than depending keeps osl-ephys installable on its own.

  Consequences:
  - The code exists in both packages. The `meeg` subpackage changed in 30
    commits in the last 90 days, so the port is done from a pinned
    osl-dynamics commit and recorded (section 8), and a fix made in one
    package after the port is offered to the other as a PR.
  - Licences already match: osl-ephys has been Apache-2.0 since 2024-10-09
    (the `license='MIT'` in main's `setup.py` was stale metadata; `v3` has it
    right). Phase 1 aligns the rest with osl-dynamics: ship the full Apache
    text rather than the short header, add a NOTICE file (copyright line, OHBA
    attribution, the osl-files note), and set `license = "Apache-2.0"` with
    `license-files = ["LICENSE", "NOTICE"]` in `pyproject.toml`. Ported code
    then needs no extra attribution beyond the shared NOTICE.
  - The tests osl-ephys writes in phase 7 are the first tests this code has
    had; they can be offered to osl-dynamics too.

D2. Python floor: `>=3.10` (same as osl-dynamics; drops the 3.7-3.9
    classifiers and allows `X | None` annotations). Dependencies are the
    osl-dynamics set minus TensorFlow, plus what the osl-ephys-only modules
    need (section 8).

D3. Parallelism: drop dask and distributed, use the `multiprocessing` runner
    design (per-session log file, `summary.csv`, worker recycling, QC report
    hook). Config API keeps a single `n_workers` argument; `dask_client` goes
    with a clear error message for one release.

D4. Keep the YAML config API as the public face of osl-ephys, and make it
    one config for the whole pipeline. Today preprocessing and source
    reconstruction have separate config formats, separate `find_func`/
    `load_config` implementations and separate batch runners
    (`run_proc_batch`, `run_src_batch`). In v3 a single config can hold any
    mix of steps (preprocessing, surface extraction and coregistration,
    forward model and source reconstruction, parcellation, sign flipping) and
    one runner
    executes it per session. Running only the preprocessing section, or only
    the source-recon sections, is the same config with sections left out.
    It is what the workshops and the toolbox paper teach, and it is the thing
    osl-dynamics does not offer for M/EEG.

D5. Reports: replace the per-subject jinja2 pages with the osl-dynamics style
    (each step writes `summary.json` + WebP plots to a QC directory, one table
    report across sessions). Keep the osl-ephys-specific preproc plots as QC
    plots. `coreg.html` stays available as an option (the ported
    `plot_coregistration` writes it when asked); the PNG multi-view is the
    default, which is what issue #437 asks for.

D6. Version: the next release is v3.0.0 with a v2 to v3 migration page. v2.4.x
    stays installable for the toolbox paper and existing workshop material.

D7. Settled: the five PRs and three of the stale branches are merged into
    `v3`; `viz` is the one left to port by hand or delete (section 5).

D9-D18, answered 2026-10-16 (details in section 11):
    D9  `run_batch` keeps `overwrite`, `random_seed`, `verbose`, `extra_funcs`,
        `covs` (per-session covariates for the GLM steps); `logsdir`/`reportdir`
        become `log_dir`/`qc_dir` (`qc_dir=None`: no report); `strictrun`,
        `ftype` and `skip_save` go.
    D10 Output layout is the osl-dynamics one (`outdir/preproc/<id>_preproc-raw.fif`,
        `outdir/<id>/{coreg,src}`), so derivatives are interchangeable.
    D11 Batches run interactively on a many-core machine: the multiprocessing
        runner on one node, no job-array option.
    D12 Session IDs come from an explicit `id` column, the raw filename stem
        as the fallback.
    D13 One `osl-ephys` command with subcommands; the old script names go
        (see D19).
    D14 Sign flipping: port the osl-dynamics method and test it; no
        benchmark against the `new-sign-flip` methods or an existing dataset.
    D15 FreeSurfer route kept and tested with a fast setting: MNE's
        `fsaverage` (surfaces and BEM included, no FreeSurfer binaries) in
        place of `recon_all`.
    D16 Epoched source reconstruction (`epoch_files`) kept.
    D17 Report: the table carries bad time %, bad channels, ICA components
        rejected, coregistration error, MNI registration score, empty parcels,
        plus head movement and EOG/ECG summary numbers; the v2 standalone
        subject pages are dropped.
    D18 SEMP is dealt with separately from the phases (own board item); its
        `osl_pathfinder` dependency is raised with its author, with the
        suggestion to rename it `osl-pathfinder` like the other OSL repos.
        Done 2026-10-17: osl-pathfinder 1.0.0 is on PyPI (Apache-2.0) and is
        a dependency of osl-ephys (pyproject and the channel recipe); the
        SEMP package imports and its tests pass. Left for the SEMP item:
        its two tutorials on the v2 API and its docstring warnings.
        osl-signflip (SungJun Cho, the upstream of the ported sign-flipping
        solvers, MIT) is to become a dependency too (user, 2026-10-17): it
        has no release yet (no tag, not on PyPI), so once it is released
        the `sign_flip`/`fix_sign_ambiguity` steps call it and the copied
        solvers in `group/sign_flipping.py` go.
    D21 (2026-10-17, supersedes D20; D8 reason) osl-ephys is pip-only: no
        conda package, no OHBA channel. Every dependency ships wheels, so
        the channel only added headless VTK on Linux, which one documented
        pip line (`vtk-osmesa` from wheels.vtk.org) covers; against that it
        cost an anaconda.org organisation and token, five recipes, a
        five-platform workflow and, once osl-dynamics joined, a shared
        recipes repository. Removed: `recipe/` and
        `.github/workflows/conda.yml` (both in git history at 968f43f).
        The `ohba-analysis-group` organisation, its dev builds and the
        `ANACONDA_API_TOKEN` secret stay for now (user, 2026-10-17); nothing
        publishes to the channel any more. Install docs: pip into
        a conda environment (no venv option, user 2026-10-17), `osl-ephys check`, headless section.
    D20 (2026-10-17, superseded by D21) The conda package is distributed from an OHBA channel on
        anaconda.org, not from conda-forge: conda-forge cannot carry glmtools
        (no package) or sails (1.5.18 there, without the GLM periodogram),
        and a conda-forge package cannot depend on pip. The channel
        (`ohba-analysis-group`, created 2026-10-17 on anaconda.org after `ohba-analysis` was created on anaconda.com by mistake and its name stayed reserved; `ohba` belongs to an unused
        personal account from 2019) carries osl-ephys and, built from their
        PyPI sdists by the same workflow, glmtools, anamnesis and sails, so
        one command installs everything:
        `mamba create -n osle -c ohba-analysis-group -c conda-forge osl-ephys`.
        osl-dynamics on the channel is the user's own, separate task;
        conda-forge and mne-installers are dropped. The channel can be
        closed later (install docs first; existing environments keep
        working).
    D19 (2026-10-16) No backwards compatibility with v2: no re-export shims
        at old import paths, no wrappers for the v2 entry points
        (`run_proc_chain` etc. are gone), no old script names, no step-name
        aliases, no renamed- or ignored-option handling, no `FutureWarning`
        for old data file names (the osl-files names are the only names).
        No migration page either (user, 2026-10-17): the changelog of the
        release describes what changed.
    Also: the browser review page is the primary ICA review tool, the
    matplotlib GUI stays as an alternative; GLM config steps kept; FSL tests
    skipped in CI (no self-hosted runner); all eleven tutorials kept; no
    workshop deadline; glmtools will not be put on conda-forge by us; v3 is
    the only line from here (no v2.4.x branch).

D8. No feature is removed from osl-ephys unless there is a reason that is
    written down. The default disposition of every public function, class,
    wrapper, CLI and config option is "keep": either unchanged, or with a
    ported equivalent that does at least what the old one did. The feature
    inventory (section 9) lists every public name on `v3` with its
    disposition, and every "drop" carries its reason. A phase is not done
    while an entry in its scope still says "undecided".

## 3. Phases

Ordered so that each phase lands on a green CI and the dangerous ones (4, 5)
come after the scaffolding exists to test them.

### Phase 1: Packaging, tooling, CI  (todo: dependencies, codespell/black, release process)

Status 2026-10-16: done. Tests pass on Python 3.10 and 3.13 on Ubuntu and
macOS, black and codespell pass, the PyPI trusted publisher and the GitHub
`pypi` environment exist, and the `v3` docs build succeeds on ReadTheDocs
(152 Sphinx warnings, all pre-existing rst issues in docstrings and
tutorials; cleaned up in phase 8).

- `pyproject.toml` with setuptools_scm; delete `setup.py`, `setup.cfg`,
  `requirements.txt`, `Makefile`; `__version__` from `importlib.metadata`.
  Version then lives only in git tags (fixes the four-file bump in
  `release_notes.md`; `CITATION.cff` version updated at release time only).
  The files already disagree: `setup.py` says 2.5.dev0, `__init__.py` says
  2.4.dev0, `conf.py` and `CITATION.cff` say 2.4.0.
- Quick correctness fixes to land before the rewrite so they are not lost:
  issue #446 (`set_format` undefined in `utils/logger.py:90`, `click` undefined
  in `preproc/batch.py:1094`), #440 (`detect_zeros` should skip EEG
  picks), and the inline TODO/XXX notes in `utils/file_handling.py:124`,
  `utils/spmio/spmmeeg.py:163`, `preproc/plot_ica.py` (five places),
  `source_recon/freesurfer_utils.py:89`, `source_recon/parcellation.py:205`,
  each either fixed or turned into a board item.
- Dependency audit (what is actually imported):
  - remove: `dask`, `distributed` (D3), `opencv-python` (one
    `morphologyEx` call in `rhino/surfaces.py`, replace with
    `scipy.ndimage`; issue #439), `open3d` soft import (issues #439, #193),
    `neurokit2`, `panel`, `h5io` (listed but never imported)
  - keep: `sails`, `glmtools` (GLM module; required, not an extra; `glmtools`
    needs a conda-forge feedstock before phase 9), `fslpy`, `parse`,
    `numba`, `jinja2` (until D5)
  - add: `osl-dynamics` (D1a) or `pooch` (D1b/c)
  - `sails` unpinned. Its `glm_irasa` raises inside sails itself from 1.8 to
    1.13, so osl-ephys' `glm_irasa` wrapper is dropped rather than pinning
    sails to 1.7 (decided 2026-10-16; the user does not expect a sails fix).
    `glm_periodogram` works on every version.
- `.github/workflows/`: `tests.yml` (push + PR, matrix Python 3.10-3.13,
  ubuntu + macos, `pip install -e .[dev]`, pytest), `codespell.yml`,
  `black.yml`, `release.yml`. Delete `osl-cicd-actions.yml`.

  Release process (own board item). Today: ten manual steps in
  `release_notes.md`, version strings in four files, `python setup.py sdist`
  and `twine upload` by hand. Target, copied from osl-dynamics:
  1. Version from the git tag via setuptools_scm. No version string anywhere
     in the repo; `CITATION.cff` is edited at release time only.
  2. `release.yml` as in osl-dynamics: on `release: published`, build the
     sdist and wheel, `twine check`, upload with
     `pypa/gh-action-pypi-publish` using trusted publishing (`id-token:
     write`, environment `pypi`, `skip-existing: true` so a failed upload can
     be re-run).
  3. One-time setup: a `pypi` environment on the GitHub repo, and a trusted
     publisher on pypi.org for the osl-ephys project (owner `OHBA-analysis`,
     repo `osl-ephys`, workflow `release.yml`, environment `pypi`). No API
     tokens.
  4. README "Releases" section copied from osl-dynamics: check the latest
     main built on RTD, create a GitHub release with tag `vX.Y.Z`, write the
     notes, publish, check the Actions run. `release_notes.md` deleted.
  5. Prove the workflow with a release candidate (`v3.0.0rc1`, marked
     pre-release) before `v3.0.0`.

  ReadTheDocs (own board item). Today: `.readthedocs.yaml` at the root on
  Python 3.8 installing `requirements.txt`, docs under `doc/source`, the
  GitHub avatar as logo, no sitemap or `llms.txt`. Target, copied from
  osl-dynamics:
  1. `doc/.readthedocs.yaml` (moved from the root 2026-10-17 to match
     osl-dynamics; needs the configuration-file path set to
     `doc/.readthedocs.yaml` in the project's Admin > Settings on
     readthedocs.org, which the user does): Python 3.12, `post_checkout` fetching tags so setuptools_scm
     can version the build, `doc/requirements.txt` that installs the package.
  2. Flatten `doc/source` to `doc/`; `conf.py` copied: pydata theme, OHBA
     logo from osl-files, autoapi with the `_templates/python` templates,
     sphinx-gallery `tutorials` -> `tutorials_build`, sphinx-sitemap with
     `html_baseurl`, `llms.txt` through `html_extra_path`, `github_url`,
     `show_nav_level` 2, no section navigation, no "show source".
  3. `index.rst` header with the OHBA and OxCIN logos from osl-files and a
     feature list that matches v3 (no "Dask", no "HTML reports").
  4. `install.rst` in the osl-dynamics style: conda-forge first, pip, BMRC,
     development install, removal. The `envs/*.yml` lockfiles go.
  5. README trimmed to description, badges, conda install, docs link,
     Releases, citation. RTD project settings already match (default branch
     `main`, version `latest`); only the config file path changes.
- One-off black reformat with black's defaults (line length 88, as
  osl-dynamics; decided 2026-10-15) in its own commit, listed in
  `.git-blame-ignore-revs`. Codespell config in `pyproject.toml`.
- (done 2026-10-17) An `AUTHORS` file at the repository root listing the contributors (the
  module headers no longer carry author lists; `git shortlog -sn` is the
  starting point), referenced from `NOTICE` and `CITATION.cff`. Added
  2026-10-16, not yet done.
- Licence files aligned with osl-dynamics: full Apache-2.0 text in `LICENSE`,
  a `NOTICE` file, `license`/`license-files` in `pyproject.toml`, the
  classifier. `CITATION.cff` version (currently `2.5.0.dev1+semp.2`) set only
  at release time.
- README: short description, badges, conda install first, "Releases" section
  copied from osl-dynamics; `release_notes.md` deleted; CONTRIBUTING trimmed
  (drop the git school, point at osl-dynamics' conventions).
- Remove import-time side effects in `osl_ephys/__init__.py` (NUMEXPR env var,
  logger level changes), the README-as-`__doc__` hack in every subpackage, and
  the module-level `matplotlib.use('Agg')` in `source_recon/wrappers.py` and
  `rhino/surfaces.py` (issues #450, #431). Set Agg only inside worker processes.

### Phase 2: Data to osl-files  (todo: test data, remove data, logos)

Status 2026-10-16: `osl_ephys.files` ported (shared cache with osl-dynamics,
`osl-ephys-download-data` CLI, `test_data` and `simulation` submodules);
`find_file` resolves parcellations and masks through osl-files (the alias
table for old names was removed under D19); the eleven duplicate
parcellations and the simulation config are removed from the package; the
Neuromag-306 template info (0.08 MB once the acquisition dump and HPI data
were stripped, headshape thinned and jittered) and the MVAR parameters are
in osl-files under `scanner/` and `simulation/`. All bundled data is gone from the package (`source_recon/files/` removed,
3.3 MB of package left): five files added to osl-files under the naming
convention, six kept under their original names in `parcellation/legacy/`,
three dropped with the reason in the table below and the inventory.
Found while porting: `fetch_file` raised "not a data file" for anything
added to osl-files since the registry was cached (up to an hour) instead of
refreshing the registry first; fixed in the osl-ephys copy.

Reference files (8.4 MB in `source_recon/files`, 6.9 MB in
`utils/simulation_config`) move out; fetched on demand through a ported
`osl_ephys.files` (pooch, same cache directory as osl-dynamics).

Measured mapping (2026-10-16). Each bundled file was matched to osl-files by
content: same parcel count, then the mean distance between matched parcel
centroids in MNI mm, which survives the regridding osl-files did. Under
2 mm is the same parcellation.

| osl-ephys file | osl-files match | Distance | Disposition |
| --- | --- | --- | --- |
| Glasser52_binary_space-MNI152NLin6_res-8x8x8 | atlas-Glasser_nparc-52_space-MNI_res-8x8x8 | 0.0 mm | alias |
| Glasser50_space-MNI152NLin6_res-8x8x8 | atlas-Glasser_nparc-50_space-MNI_res-8x8x8 | 0.7 mm | alias |
| dk_cortical | atlas-DK_nparc-68_space-MNI_res-8x8x8 | 0.0 mm | alias |
| fMRI_parcellation_ds2mm | atlas-Giles_nparc-38_space-MNI_res-2x2x2 | 0.0 mm | alias |
| fMRI_parcellation_ds8mm, fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm | atlas-Giles_nparc-38_space-MNI_res-8x8x8 | 0.0 mm | alias |
| fmri_d100_parcellation_with_PCC_tighterMay15_v2_8mm | atlas-Giles_nparc-39_space-MNI_res-8x8x8 | 0.8 mm | alias |
| fmri_d100_parcellation_with_PCC_tighterMay15_v2_2mm | atlas-Giles_nparc-39 (only 8 mm in osl-files) | 1.3 mm | added as atlas-Giles_nparc-39_space-MNI_res-2x2x2 |
| aal_cortical_merged_8mm_stacked | atlas-AAL_nparc-78_space-MNI_res-8x8x8 | 2.8 mm | alias (osl-dynamics already maps it) |
| MNI152_T1_2mm_brain, MNI152_T1_8mm_brain | mask/MNI152_T1_2mm_brain, 8mm_brain | 0.0 mm | alias |
| Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm_4d (+ _ds8) | atlas-Schaefer_nparc-100 | 0.7 mm | ds8 aliases the 8 mm file; the 2 mm one added as res-2x2x2 |
| fmri_d100_parcellation_with_PCC_tighterMay15_v2_6mm_exclusive, giles_39_binary, fmri_d100_parcellation_with_PCC_reduced_2mm, WTA_fMRI_parcellation_ds2mm, WTA_fMRI_parcellation_ds8mm, dk_full | variants with no maintained counterpart | | kept under their original names in osl-files `parcellation/legacy/` (decided 2026-10-16: no data is lost); aliases point there |
| HarvOxf-sub-Schaefer100-combined-2mm_4d (+ _ds8) | none (114 parcels) | | added as atlas-SchaeferHarvOxf_nparc-114_space-MNI_res-2x2x2 and 8x8x8, with a docs entry and the Harvard-Oxford licence note |
| HarvardOxford-sub-prob-bin-2mm | none (21 parcels) | | added as atlas-HarvOxfSub_nparc-21_space-MNI_res-2x2x2 |
| reduced_hcp-mmp_2mm, _8mm | none (230 parcels) | | dropped (decided 2026-10-16): referenced nowhere in the code, docs or examples; the maintained Glasser files cover the atlas |
| Glasser52_binary_space-MNI152NLin6_res-8x8x8_10mm | Glasser-52 at 10 mm | | dropped (decided 2026-10-16): referenced nowhere; the 8 mm and 2 mm files are in osl-files |
| *.xml label files | docs/parcellations in osl-files | | labels go with their parcellation's docs entry |

- Rule for the move: nothing is copied into osl-files that is already
  there. For each osl-ephys file, first check whether osl-files has the same
  parcellation (same atlas, parcel count and resolution; compare the volumes,
  not the checksums, because osl-files regridded everything); if so, osl-ephys
  simply points at the osl-files name. Only files with no counterpart are
  added, and they are renamed to the osl-files convention
  (`atlas-<Name>_nparc-<N>_space-MNI_res-<r>x<r>x<r>.nii.gz`, masks as
  `MNI152_T1_<r>mm_brain.nii.gz`) with a licence file in `LICENSES/` and a
  line in `docs/parcellations`. The old-name to new-name table lives only
  in the migration page (D19); the content matches are in section 2.
- `utils/simulation_config/*` moved: the template info to osl-files
  `scanner/neuromag306_template_info.fif`, the MVAR parameters to
  `simulation/neuromag306_mvar_*.npy` (done). `neuromag306_info.yml` and the
  `trees/` are tiny and stay in the package.
- Add osl-ephys and its `osl-ephys-download-data` CLI to the "Packages that
  use these files" table in the osl-files README.
- Test recordings in osl-files `test_data/`: section 10, done first. Summary:
  - non-identifiable: `mne.io.anonymize_info`, no sMRI (coreg tests use
    `surface/mni152` from osl-files, which also means no FSL is needed in CI),
    headshape/fiducials kept because coregistration tests need them
  - small: crop to tens of seconds, downsample, keep full channel set for
    Elekta (maxfilter / ICA-label paths depend on it)
  - features: a bad channel, a bad segment, ECG + EOG channels, a stim channel
    with events, head position channels (chpi) for the maxfilter code paths,
    CTF as a `.ds` directory plus `.pos`/`.elc`, multi-axis OPM (issue #427),
    EEG with a montage and a reference
  - osl-files is a plain git repo fetched over raw.githubusercontent, so keep
    each dataset to a few MB; larger tutorial datasets stay on OSF

### Phase 3: One config, one runner  (todo: refactor parallel processing, remove dask; unify preproc and source recon)

Status 2026-10-16: core landed on `v3`. `osl_ephys.pipeline` holds the
runner, registry, config, manifest, `run_chain`, `run_batch` and the
`osl-ephys` command (`run`, `download-data`, `check`); the v2 entry points
(`run_proc_chain`, `run_proc_batch`, `run_src_chain`, `run_src_batch`,
`osl_preproc`) are wrappers that warn; dask is gone. The source-recon
sections call the v2 wrappers until phase 4; the reports are the v2 ones
until phase 5. Left in phase 3: `examples/parallelisation/` rewrite (done
with the examples in phase 8), and `extract_surfaces` marked per-subject
once phase 4 ports it.

Detailed steps and the target layout: section 11.

API proposal for discussion with Mats: issue #454 (what a batch script, the
CLI, interactive use and custom steps look like; sections are `preproc` / `coreg` (surfaces, fiducials,
coregistration) / `source_recon` (forward model, beamformer) /
`parcellation` / `group`; open questions on manifest columns, HPC, in-memory data and `osl_func`).

The unified config (D4):

```yaml
preproc:
  - filter: {l_freq: 1, h_freq: 45}
  - bad_segments: {picks: mag}
  - ica_label: {picks: meg}
coreg:
  - extract_surfaces: {include_nose: false}
  - extract_fiducials_and_headshape: {}
  - coregister: {use_nose: false, allow_mri_scaling: false}
source_recon:
  - forward_model: {model: Single Layer, gridstep: 8}
  - lcmv_beamformer: {chantypes: [mag, grad], rank: {meg: 60}}
parcellation:
  - parcellate: {parcellation_file: atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz,
                 orthogonalisation: symmetric}
```

- One step registry replaces the two `find_func` implementations: every
  wrapper registers under a section name, and custom functions are passed in
  the same way for every section (today `extra_funcs` only exists for
  preprocessing). A step's wrapper receives the session dict and the loaded
  data object and returns them, so sections chain without going through the
  filesystem unless the user asks for intermediate files.
- One session manifest (list of dicts, or a CSV, with `id`, `raw_file`,
  `smri_file`, `subject`, optional `pos_file`/`elc_file`) feeds every section;
  the `Session` object from phase 4 is built from it. `utils.file_handling`'s
  glob-based input handling and `Study` become helpers for building the
  manifest, not part of the runner.
- Steps declare whether they are per subject or per session when they are
  registered: `extract_surfaces` (in `coreg`) runs once per subject before
  the per-session loop, grouped by the `subject` column; everything else
  runs per session. A config with only some sections present skips the others and
  checks (via `check_up_to_date`) that the earlier outputs it needs exist.
- Group-level steps (sign flipping template selection, `fix_sign_ambiguity`)
  run after the per-session loop as a `group` section, which also answers
  issue #383.
- `run_batch(config, manifest, outdir, n_workers=N)` is the single entry
  point, with `run_chain` for one session; `run_proc_batch` and
  `run_src_batch` remain for one release as thin wrappers that build the
  equivalent config and warn.
- `osl_preproc` CLI becomes `osl-ephys run config.yml manifest.csv outdir`
  (the old name removed, D19).

Parallelism underneath it:


- One runner (ported from `osl_dynamics.meeg.parallel.run`)
  used by both `run_proc_batch` and `run_src_batch`: items are session dicts,
  the runner handles per-session logging, failure isolation, `summary.csv`,
  and calls report generation at the end.
- `run_proc_batch(... n_workers=N)` replaces `dask_client=True` plus a
  user-created `Client`. `utils.parallel.dask_parallel_bag`, `test_parallel.py`
  and `examples/parallelisation/` rewritten.
- `utils.logger` replaced by the per-session logger the runner expects
  (issue #323).

### Phase 4: Source reconstruction and parcellation  (todo: match osl-dynamics)

Status 2026-10-16: core landed on `v3`. `source_recon/{session,rhino,
beamforming,parcellation,sign_flipping}.py` are the ported osl-dynamics
modules (commit b64c6eb0), `source_recon/steps.py` registers the config
steps on them (`extract_surfaces` per subject,
`extract_fiducials_and_headshape` and its `_from_pos`/`_from_elc`
variants, `remove_stray_headshape_points`,
`save_mni_fiducials`, `coregister`, `forward_model`, `beamform`,
`parcellate`, `beamform_and_parcellate`, `fix_sign_ambiguity`, and a
`sign_flip` group step that picks the template), with v2 options that no
longer apply ignored with a warning. A session with no `smri_file` uses the
MNI152 surfaces. The end-to-end test runs preprocessing, coregistration,
forward model, LCMV, parcellation and sign flipping on two sessions of the
Elekta recording in CI. Still v2 and registered as such: `minimum_norm`,
`minimum_norm_and_parcellate`, `make_watershed_bem`, `extract_rhino_files`.
Left in phase 4: those four onto `Session` with the fsaverage test, the
kept helpers from the `_v2_` modules (inventory lines), deleting the `_v2_`
modules and `wrappers.py`, the migration notes.

Detailed steps and the target layout: section 11.

- Port (see the inventory above) `Session` (fixed output layout: `{outdir}/{id}/{bem,coreg,src}`),
  `check_up_to_date`, `save_params` JSON next to each output, the consolidated
  `rhino` API (surfaces, fiducials from fif/pos/elc, coregistration, forward
  model), `source_recon.lcmv_beamformer` (with bilateral pairs, which
  supersedes PR #426 and issue #103), `parcellation` (Parcellation class,
  `parcellate_lcmv`, symmetric orthogonalisation, `save_as_fif`), and the
  osl-dynamics sign flipping (compare with the `new-sign-flip` branch first).
- Keep and port onto the new Session: `minimum_norm` (merge PR #445 first),
  `freesurfer_utils`, `nii.py`, `extract_rhino_files`, EEG sensor moving.
- Rewrite `source_recon/wrappers.py` so each YAML step is a thin call into the
  new functions, registered in the phase 3 step registry under the `coreg`,
  `source_recon` and `parcellation` sections; old
  config keys map to new ones with warnings where cheap. Output layout
  changes, so the migration page documents old vs new paths.
- Closes along the way: #441/#442 (transform_vtk_mesh), #409 (coreg float),
  #418 (nilearn private function), #393 (fix_sign_ambiguities), #437
  (coreg.html), #439/#193 (open3d, opencv), #200 (MNE option exists), #383
  (group steps after individual steps become a documented pattern).
- Retire what the new code replaces (`rhino/utils.py`, `rhino/coreg.py`,
  `rhino/surfaces.py`, `rhino/polhemus.py`, `rhino/forward_model.py`,
  `beamforming.py`, most of `parcellation.py`) only once every public name in
  them has a disposition in the feature inventory (section 9). Functions the
  osl-dynamics code has no equivalent for (e.g. `delete_headshape_points`,
  `plot_polhemus_points`, `fsleyes` helpers, `get_leadfields`,
  `voxel_timeseries`, `transform_recon_timeseries`, `recon_timeseries2niftii`)
  are moved, not dropped.

### Phase 5: Reporting  (done 2026-10-17)

Detailed steps and the target layout: section 11.

- Port `osl_dynamics.meeg.report`: each step writes `summary.json` and WebP
  plots into `qc_dir/<step>/<id>/`; `generate_report` builds one sortable table. The preproc plots
  that only osl-ephys has (flowchart, ICA components, artefact scan,
  digitisation, head movement, events, custom figures) become QC plot
  functions in `preproc`. `report/templates/*.html`, `preproc_report.py`
  (1745 lines) and `src_report.py` go.
- `osl_ica_label` keeps working from the saved fif + ICA files; its GUI
  (`plot_ica.py`) is untouched by this phase.
- Triaged 2026-10-17 (to be closed when v3 is released, per the rule of
  not touching issues before then): #411 and #413 (report robustness) are
  closed by design, the report reads what each session saved and a failed
  QC plot is a warning in the session log; #135: the report takes a list of
  ICA objects (ica.webp, ica_2.webp, ...), the file overwrite is fixed in
  phase 6 when the ICA step names its file by its picks; #212 (navigating
  the report): the table's filter box and sorting; #134 (scan notes): any
  extra manifest column (e.g. `notes`) is a column of the table; #170
  (ERF when epoching): the evoked responses plot (GFP per condition and
  channel type) and the epoch counts; #251 (surfaces plot slices): the
  slices go through the centre of the inner skull.

### Phase 6: Preprocessing cleanup

Detailed steps and the target layout: section 11.

- Merge PR #444 (GESD) and fix #440 (maxfilt zeros fallback) before touching
  `osl_wrappers.py`, so the fixes are not lost in the rewrite.
- Port `osl_dynamics.meeg.preproc` and add wrappers for its functions
  (`detect_bad_segments`, `detect_bad_channels`, `ica_label` via mne-icalabel /
  MEGnet) alongside the existing `run_osl_*`; issue #427 (multi-axis OPM) and
  #373 (rectified signal warning) addressed here.
- `maxfilter`: adopt the osl-dynamics rewrite (chunking, CBU three-stage,
  split-file chains; fixes #192) behind the existing `osl_maxfilter` CLI.
- Merge the two ICA review/apply CLI pairs into one (decision in the
  feature inventory, section 9).
- `glm`: keep as is; only the sails/glmtools pin changes.
- `utils.spmio`, `utils.study`, `utils.opm`: keep; `utils.simulate` reads the
  template from osl-files.

### Phase 7: Tests and GitHub Actions  (done 2026-10-17)

Detailed steps and the target layout: section 11.

- Session-scoped fixtures that fetch the osl-files test data once per CI run
  (cache `~/.cache/osl-files` with `actions/cache`).
- Test the things that break silently today: config loading and version
  checks, every `run_mne_*`/`run_osl_*` wrapper on each modality's test data,
  the chain and batch on two sessions with `n_workers=2`, the parallel runner's
  failure isolation and summary, parcellation maths (orthogonalisation,
  parcel weights), sign flipping on synthetic data, coreg + forward + LCMV +
  parcellate end to end on MNI152 surfaces, report generation, every CLI's
  `--help`.
- FSL-dependent code (`extract_surfaces`, FNIRT) marked `fsl` and skipped
  unless `FSLDIR` is set. MaxFilter can only be tested at the command-building
  level.
- `osl_ephys.check_installation()` / `osl-ephys-check` CLI (issue #425)
  replaces `utils.package.run_package_tests`.

### Phase 8: Examples and docs  (done 2026-10-17)

Detailed steps and the target layout: section 11.

- `examples/{elekta,ctf,eeg,opm,parallelisation}` rewritten against the v3
  API, mirroring the osl-dynamics `meg_preproc` script layout (config block at
  the top, `process_session`, `parallel.run`). `examples/toolbox-paper` and
  `examples/oxford` are frozen with a note that they target v2.4.
- Tutorials: keep the OSF datasets for the long ones; the short ones can run
  on the osl-files test data so the docs build exercises the code.
- Docs: install page in the osl-dynamics style (conda-forge first), migration
  page, `llms.txt`, FAQ refresh; issues #352, #361, #386.

### Phase 9: the OHBA conda channel (D20, 2026-10-17; dropped by D21 the same day: pip-only)

Detailed steps and the target layout: section 11.

- `glmtools` stays a required pip dependency (decided 2026-10-15) and we will
  not submit a conda-forge recipe for it (2026-10-16): the conda package
  omits it and the GLM module imports it lazily, see section 11.
- Recipe modelled on `osl-dynamics-feedstock/recipe/recipe.yaml`: `noarch:
  python`, entry points for the five CLIs, import tests per subpackage,
  `vtk * *osmesa*` on linux-64 so the 3D coreg plot renders headless (this is
  the "make sure the coreg plotting works" item; verify with a headless test
  that calls `plot_coregistration` with `pyvista.OFF_SCREEN = True`).
- mne-installers: a PR adding `osl-ephys =X.Y.Z` (and `osl-dynamics`, which
  is not there either) to `recipes/mne-python/construct.yaml`. Requires the
  conda-forge package first and that every dependency resolves with the
  installer's pinned MNE and Qt stack; check `pyside6`/`pyvistaqt` versions
  against theirs before opening the PR.

### Phase 10: Release v3.0.0

Detailed steps and the target layout: section 11.

- Changelog from the project board (no migration page, user 2026-10-17:
  the release notes carry the config key renames, output layout, dask
  removal, dropped parcellation names and dropped `coreg.html`); tag
  `v3.0.0`, the release workflow publishes to PyPI.

### Port inventory: what comes from osl-dynamics

Everything below is in osl-dynamics at commit b64c6eb0 (2026-10-12). "Port"
means copy into osl-ephys with the import paths changed and the module
docstrings pointing at osl-ephys; "replace" names the osl-ephys code it makes
redundant. Phase numbers say when.

| osl-dynamics | Lines | Into osl-ephys | Replaces | Phase |
| --- | --- | --- | --- | --- |
| `files/_fetch.py`, `files/__init__.py`, `files/functions.py`, `files/{parcellation,mask,surface,mni152_surfaces,scanner,scene}` | ~400 | `osl_ephys/files/` with an `osl-ephys-download-data` CLI; same cache directory so both packages share downloads | `source_recon/files/`, `utils/simulation_config/`, `parcellation.find_file` | 2 |
| `utils/logger.py` (`MEEGSessionLogger`, `set_logging_level`) | 130 | `utils/logger.py` | current `utils/logger.py` (`log_or_print`, `set_up`) | 3 |
| `meeg/parallel.py` (`run`, `log`, `warning`, thread limiting, `summary.csv`) | 283 | `utils/parallel.py` | `dask_parallel_bag`, the dask paths in both batch functions | 3 |
| `meeg/session.py` (`Session`, `SurfaceFilenames`, `CoregFilenames`, `check_up_to_date`, `save_params`) | 322 | `source_recon/session.py` | `rhino.get_surfaces_filenames`, `get_coreg_filenames`, the ad hoc path building in `wrappers.py` | 4 |
| `meeg/rhino.py` (surfaces, fiducials from fif/pos/elc, stray point removal, coregistration + QC, BEM repair, forward model, MNI grid, FNIRT) | 3860 | `source_recon/rhino.py` (one module; the `rhino/` package goes) | `rhino/utils.py`, `coreg.py`, `surfaces.py`, `polhemus.py`, `forward_model.py`, `fsl_utils.py` | 4 |
| `meeg/source_recon.py` (LCMV, bilateral pairs, virtual electrodes, QC plot) | 1760 | `source_recon/beamforming.py` | current `beamforming.py` and the #426 implementation merged into v3 | 4 |
| `meeg/parcellation.py` (`Parcellation`, `parcellate`, `parcellate_lcmv`, orthogonalisation, `save_as_fif`, `convert_to_mne_raw`, QC plots) | 1118 | `source_recon/parcellation.py` | current `parcellation.py` | 4 |
| `data/sign_flipping.py` + the two helpers it uses from `data/processing.py` (`standardize`, `time_embed`) | 430 + 60 | `source_recon/sign_flipping.py` | current `sign_flipping.py` (compare with the `new-sign-flip` branch first) | 4 |
| `utils/plotting.py` `plot_psd_topo` and `plot_brain_surface`, `analysis/power.py` pieces used by `parcellation.save_qc_plots` | ~300 of 2742 | `utils/plotting.py` (only the functions needed) | nothing | 4 |
| `meeg/preproc.py` (`detect_bad_segments`, `detect_bad_channels`, GESD, `ica_label`, `ica_ecg_eog_correlation`, `decimate_headshape_points`, QC plots) | 1323 | `preproc/functions.py`, wrapped by `run_osl_*` steps | `osl_wrappers.gesd`, `bad_segments`, `bad_channels`, `detect_artefacts`; `ica_label` gains the mne-icalabel route (#427) | 6 |
| `meeg/maxfilter.py` (chunking, multistage, CBU three-stage, split chains, bad-channel merging) | 1482 | `maxfilter/maxfilter.py` behind the `osl_maxfilter` CLI (osl-dynamics has no CLI; keep ours) | current `maxfilter.py` (fixes #192) | 6 |
| `meeg/amm.py` (`apply_amm`, OPM interference rejection) | 1196 | `preproc/amm.py` + a `run_osl_amm` step | nothing (new capability) | 6 |
| `meeg/report.py` (`generate_report`, `TABS`, `CAPTIONS`) | 598 | `report/report.py` | `preproc_report.py`, `src_report.py`, `report/templates/` | 5 |
| `pyproject.toml` structure, `.github/workflows/{release,codespell,black}.yml`, `doc/requirements.txt`, README "Releases" section, `conf.py` logo and autoapi settings | | same paths | `setup.py`, `setup.cfg`, `requirements.txt`, `Makefile`, `release_notes.md`, `osl-cicd-actions.yml` | 1 |
| `examples/meg_preproc/` layout (config block, `process_session`, `parallel.run`) and tutorials 0-1, 0-3, 0-4, 0-5 | | `examples/` and `doc/source/tutorials/` adapted to the config API | current examples | 8 |
| `__init__.py` (version from `importlib.metadata`, one `logging.basicConfig`) | 20 | `osl_ephys/__init__.py` | the NUMEXPR and logger-level side effects | 1 |

Not ported, because osl-ephys has nothing to gain or already has it:
`config_api` (osl-dynamics' YAML pipeline is for dynamics models, not M/EEG
steps; osl-ephys keeps its own config API and extends it in phase 3),
`models`, `inference`, `analysis` (apart from the plotting pieces above),
`simulation`, `data.base`.

Kept from osl-ephys because osl-dynamics has no equivalent: the config API
and all CLIs, `glm`, `plot_ica` GUI and `osl_ica_label`/`osl_ica_apply`,
`minimum_norm`, `freesurfer_utils`, `nii.py`, `utils.spmio`, `utils.study`,
`utils.opm`, `utils.simulate`, the SEMP code from #452, and the preproc QC
plots that osl-dynamics lacks (flowchart, artefact scan, digitisation, head
movement, events, custom figures).

Dependencies after the port (osl-dynamics' list minus TensorFlow, minus
what only the dynamics models and analyses use, plus osl-ephys-only needs):

| Keep from osl-dynamics | Used by |
| --- | --- |
| numpy, scipy, matplotlib, pandas, pyyaml, packaging | everywhere |
| mne, mne-icalabel[onnx] | preprocessing, rhino, beamforming |
| nibabel, nilearn, fslpy, scikit-image, scikit-learn, numba | rhino, parcellation |
| pyvista, pyvistaqt, pyside6, trame, trame-vtk, trame-vuetify, ipyevents, ipywidgets | coregistration 3D plot: MNE's off-screen renderer is its notebook backend, which needs ipyevents and ipywidgets (found in phase 4) |
| pooch | files |
| h5io | LCMV filters file (`filters-lcmv.h5`) |

| Add for osl-ephys-only code | Used by |
| --- | --- |
| sails, glmtools | glm (both required; glmtools gets a conda-forge feedstock in phase 9) |
| jinja2 | until phase 5 removes the templates |
| parse | `utils.study` |

| Drop from osl-dynamics' list | Reason |
| --- | --- |
| tensorflow, tensorflow-probability, tf-keras | models only |
| mat73, osfclient, nest-asyncio2, tqdm, pqdm, threadpoolctl, seaborn, h5py | dynamics data loading, model training |

| Drop from osl-ephys' current list | Reason |
| --- | --- |
| dask, distributed | replaced by the multiprocessing runner |
| opencv-python, open3d | one morphology call (scipy.ndimage) and a soft import |
| neurokit2, panel, tabulate | never imported, or only by the reports that phase 5 removes |

## 4. Suggested order and milestones

1. Decide D1-D7 (this week).
2. Phase 1 on a `v3` branch with CI green on the existing tests (fix #449 to
   get there). Merge. Tag nothing yet.
3. Phase 2 (data) and phase 7's fixture scaffolding together, because the new
   tests need the test data and the data move needs tests to prove the aliases.
4. Phase 3 (config + runner, first with only the `preproc` section wired up),
   then phase 4 (source recon sections) with its tests, then phase 5
   (reports), then phase 6 (preproc). Each as its own PR against `v3`.
5. Phases 8 and 9 in parallel once the API is stable.
6. Phase 10.

Everything up to step 2 is low risk and useful even if the later phases slip.

## 5. Merge the open PRs into the v3 branch

The five open PRs are merged into the `v3` branch before the rewrite starts,
so that their fixes are ported along with everything else rather than lost.
Order them smallest first so conflicts stay local:

| PR | Size | Notes |
| --- | --- | --- |
| #442 transform_vtk_mesh fix | small | Closes #441. The function is replaced in phase 4, but the fix keeps v3 working until then. |
| #444 GESD fixes | small, no checks ran | Goes into `osl_wrappers.py` ahead of the phase 6 rewrite. |
| #445 volumetric minimum norm fix | small, checks failing | Fix the checks on the branch first; phase 4 keeps minimum norm. |
| #426 bilateral beamformer | +1745 / -844, checks failing | Merge for its tests and QC plots; the implementation is replaced by the osl-dynamics bilateral pairs in phase 4. Decide whether that is worth the conflicts or whether a pointer is enough. |
| #452 SEMP (EEG-fMRI toolbox, manual ICA browser, bug fixes) | +9749 / -99, 56 files, checks pass | Largest and last. Decide whether SEMP is in scope for v3; if not, cherry-pick the bug fixes and the ICA browser. |

None of the five has had a review yet, so each merge is also its review.

The seven stale branches, settled 2026-10-15 (D7):

| Branch | State | Outcome |
| --- | --- | --- |
| `bilateral_beamformer`, `semp`, `vol-mne` | fully contained in the merged PRs #426, #452, #445 | nothing to do; left in place until the PRs close |
| `leadfields` | one commit after its merged PR #290 (3D leadfield option) | merged into `v3` (32a6360) |
| `ohba-examples` | one commit, the Oxford SSP pipeline examples | merged into `v3` (96d43e2) |
| `new-sign-flip` | two commits by scho97, +825 lines: new sign-flipping methods in `sign_flipping.py` and `wrappers.py` | merged into `v3` (55227d7); phase 4 compares these methods with the osl-dynamics implementation before choosing |
| `viz` | one 2022 commit against the old `osl/` package path (sensor-space plots) | deleted 2026-10-16 (user's decision); nothing ported |

## 6. Risks

- Scope: phases 4 and 5 remove roughly 9k lines and change output paths;
  workshop material and existing derivatives directories depend on the old
  layout. The migration page and a v2.4.x maintenance tag are the mitigation.
- Two copies of the meeg code until osl-dynamics switches to importing from
  osl-ephys. Record the ported commit, and treat any fix to one copy as a PR
  to the other until then.
- osl-files size: test recordings in a git repo are permanent history. Agree
  a per-dataset budget before adding anything.
- `glmtools` on conda-forge is outside the team's control unless someone on the
  team becomes its maintainer.

## 7. Tracking on the project board

No new GitHub issues for tracking. The board (project 4, "osl-ephys major
overhaul") holds:

- the 5 open PRs, in "In progress", because they are merged into `v3`
- one draft item "v3 design decisions" for D1-D7
- one draft item per phase
- one draft item for merging the PRs and settling the stale branches
- one draft item "Revisit the open issues after v3"

The 48 open issues are deliberately not on the board. Many will be closed or
made irrelevant by the rewrite, so they are triaged once v3 exists; the
"closes #..." notes in the phases above are hints for that pass, not
commitments.

The script that does this is ~/Desktop/osl-ephys-populate-board.sh
(project id PVT_kwDOAOityM4BnKN4, Status field PVTSSF_lADOAOityM4BnKN4zhlwG7M,
options Todo=f75ad846, In progress=47fc9ee4, Done=98236657). Suggested
Priority options once someone creates them: P0 (blocks a release), P1 (in the
v3 scope), P2 (nice to have).

## 9. Feature inventory (D8)

Every public name on `v3` (commit 01e83fe), grouped by module, with its v3
disposition. "Keep" = unchanged or moved; "Port" = replaced by the ported
osl-dynamics equivalent, which must cover the same use; "Drop" = removed, with
the reason. Update this table as phases land; a phase is not done while a
line in its scope says "undecided".

| Module | Public names | Disposition |
| --- | --- | --- |
| CLIs | `osl_maxfilter`, `osl_ica_label`, `osl_ica_apply`, `osl_preproc`, `osl_func`, `osl-ica-review`, `osl-ica-apply` | Done (2026-10-17): every command is an `osl-ephys` subcommand (`run`, `maxfilter`, `ica review`, `ica apply`, `func`, `download-data`, `check`) and the old script names are gone (D19). The two ICA review/apply pairs (`osl_ica_label`/`osl_ica_apply` and the SEMP `osl-ica-review`/`osl-ica-apply` from #452) are merged into one pair (decided 2026-10-15): one CLI that opens the review, one that applies, covering both the fif-based and the manual-label workflows. |
| `preprocessing.batch` | `run_proc_chain`, `run_proc_batch`, `load_config`, `find_func`, `check_config_versions`, `get_config_from_fif`, `append_preproc_info`, `import_data`, `read_dataset`, `write_dataset`, `plot_preproc_flowchart`, `print_custom_func_info` | Keep; `run_proc_batch` becomes a wrapper over `run_batch` (phase 3), `find_func`/`load_config` become the shared registry. Config provenance in the fif (`get_config_from_fif`, `append_preproc_info`) is kept. |
| `preprocessing.mne_wrappers` | 23 `run_mne_*` steps | Keep all. |
| `preprocessing.osl_wrappers` | `gesd`, `detect_artefacts`, `detect_maxfilt_zeros`, `bad_segments`, `bad_channels`, `drop_bad_epochs`, `detect_bad_channels_psd`, `run_osl_read_dataset`, `run_osl_bad_segments`, `run_osl_bad_channels`, `run_osl_drop_bad_epochs`, `run_osl_zscore_present_data`, `run_osl_glm_*` (4), `run_osl_manual_ica`, `run_osl_apply_ica` | Keep all wrappers. `gesd`/`bad_segments`/`bad_channels` are Port (osl-dynamics `detect_bad_segments`/`detect_bad_channels`, same GESD, with the #444 fixes carried over); `detect_maxfilt_zeros`, `drop_bad_epochs`, `detect_bad_channels_psd`, zscore and GLM steps have no osl-dynamics equivalent and are Keep. |
| `preprocessing.ica_label`, `plot_ica`, `manual_ica.*` | `ica_label`, `apply`, `plot_ica`, `osl_MNEBrowseFigure`, `manual_ica`, `apply_ica`, review server | Done (2026-10-17): `preproc/ica/` holds the GUI (`gui.plot_ica`), the browser review package (`manual/`) and the shared `review_gui`/`serve_review`/`apply_review`; the automatic route is the `ica_label` step (mne-icalabel) and `ica_ecg_eog_correlation`. |
| `preprocessing.semp.*` | AAS/OBS/SSP epoching, metrics, plots, tracer | Keep (merged from #452); scope of maintenance to be agreed with its author. |
| `maxfilter` | `run_maxfilter`, `run_multistage_maxfilter`, `run_cbu_3stage_maxfilter`, `run_maxfilter_batch`, `quick_load_dig`, `fit_cbu_origin`, `main` | Port (osl-dynamics `run_maxfilter` covers standard, multistage, CBU 3-stage, chunking, split chains) behind the same CLI; `run_maxfilter_batch` keeps its file-list interface. |
| `glm.*` | `glm_spectrum`, `glm_epochs`, group variants, permutation classes, readers, all `plot_*` | Keep all (osl-dynamics has no sensor GLM). Drop `glm_irasa` and the `irasa` option of `run_osl_glm_fit` (decided 2026-10-16): it was a thin wrapper around `sails.stft.glm_irasa`, which raises inside sails itself from 1.8 onwards and is unmaintained; keeping it meant pinning sails to 1.7. |
| `report.preproc_report` | `gen_report_from_fif`, `gen_html_*`, `plot_flowchart`, `plot_rawdata`, `plot_channel_time_series`, `plot_sensors`, `plot_channel_dists`, `plot_spectra`, `plot_freqbands`, `plot_digitisation_2d`, `plot_headmovement`, `plot_eog_summary`, `plot_ecg_summary`, `plot_bad_ica`, `plot_summary_bad_segs`, `plot_summary_bad_chans`, `plot_artefact_scan`, `print_scan_summary`, `plot_events`, `plot_custom_figures` | Done (2026-10-17) in `preproc/qc.py`: `plot_flowchart`, `plot_raw_data`, `plot_sum_square`, `plot_sensors`, `plot_channel_stds` (+ `plot_channel_dists`), `plot_spectra`, `plot_frequency_bands`, `plot_digitisation`, `plot_head_movement`, `plot_eog`, `plot_ecg`, `plot_bad_ica`, `plot_events`, `plot_artefact_scan`, `print_scan_summary`; custom figures are saved by `save_qc_plots`. `plot_summary_bad_segs`/`plot_summary_bad_chans` are the sortable table columns. `gen_html_*` and the per-subject pages dropped (the table report). `gen_report_from_fif` kept. |
| `report.src_report` | `gen_html_*`, `plot_config`, `plot_parcellation`, `plot_sign_flipping_results`, `add_to_data`, `update_config` | Done (2026-10-17): `plot_sign_flipping_results` is the group step's sign_flip.webp (before/after correlation per session); `plot_parcellation` is `parcellation.plot_parcellation`; `plot_config` dropped (the preprocessing flowchart already draws the config, and the source-recon config is in the logs); `gen_html_*`/`add_to_data`/`update_config` dropped (summary.json per step and the table report). |
| `source_recon.batch` | `run_src_chain`, `run_src_batch`, `load_config`, `find_func` | Keep `run_src_chain`/`run_src_batch` as warning wrappers over the pipeline and `load_config` (read by the v2 report until phase 5); `find_func` replaced by the registry. Done. |
| `source_recon.wrappers` | all 19 config step names | Kept as steps (done; the v2 aliases removed under D19): `extract_fiducials_and_headshape` (was `extract_polhemus_from_info`/`extract_fiducials_from_fif`) and `_from_pos`/`_from_elc`, `remove_stray_headshape_points`, `save_mni_fiducials`, `extract_surfaces` (was `compute_surfaces`, per subject), `make_watershed_bem`, `coregister`, `forward_model`, `beamform`, `minimum_norm`, `parcellate` (with `source_method`), `beamform_and_parcellate`/`minimum_norm_and_parcellate` (combined steps), `fix_sign_ambiguity`, and `sign_flip` as the group step. `find_template_subject` is the template choice inside `sign_flip`. `rescale_sensor_positions` removed (user, 2026-10-17): it scaled the electrode positions of an EEG montage to the head, overwriting the preprocessed file; `examples/eeg` coregisters the montage's electrodes as headshape points as they are (the user, 2026-10-17, does not want `allow_mri_scaling` there either). `extract_rhino_files`: dropped, it migrated pre-2.x derivatives into the v2 layout and report. |
| `source_recon.rhino.coreg` | `coreg`, `coreg_metrics`, `coreg_display`, `bem_display`, `apply_head2mni_xform`, `get_coreg_filenames` | Ported: `coregister_head_and_mri`, `coreg_error`/`coreg_rms` (+ `coreg.json`), `plot_coregistration`, `CoregFilenames`. `bem_display` is covered by the BEM check in `forward_model` and `repair_bem_surfaces`; `apply_head2mni_xform` by the MNI grid the forward model is built on (`_get_source_coords_mni`). Done. |
| `source_recon.rhino.surfaces` | `compute_surfaces`, `surfaces_display`, `plot_surfaces`, `check_if_already_computed`, `get_surfaces_filenames` | Ported: `extract_surfaces`, `plot_surfaces`, `SurfaceFilenames`, `check_up_to_date`; `surfaces_display` = `fsleyes` on the surface files. Done. |
| `source_recon.rhino.forward_model` | `forward_model`, `make_fwd_solution`, `setup_volume_source_space` | Ported (`forward_model` on the MNI grid). Done. |
| `source_recon.rhino.polhemus` | `get_polhemus_from_info`, `extract_polhemus_from_info`, `save_mni_fiducials`, `plot_polhemus_points`, `delete_headshape_points`, `remove_stray_headshape_points`, `extract_polhemus_from_pos`, `extract_polhemus_from_elc`, `rescale_sensor_positions` | Ported the extraction and removal functions; `save_mni_fiducials` is the `mni_fiducials` argument of the coregistration; `plot_polhemus_points` is part of `plot_coregistration`; `delete_headshape_points` (interactive) kept in `rhino.py` on `Session` (a tutorial teaches it); `rescale_sensor_positions` removed (see wrappers). Done. |
| `source_recon.rhino.fsl_utils` | `setup_fsl`, `check_fsl`, `fsleyes`, `fsleyes_overlay` | Kept in `rhino.py`. Done. |
| `source_recon.rhino.utils` | 35 names | Ported inside `rhino.py` as private helpers (geometry, FLIRT/MNE transforms, meshes, `system_call`, `save_or_show_renderer`); public in `nii.py`: `get_gridstep`, `get_sform`, `get_orient`, `niimask2indexpointcloud`, `niimask2mmpointcloud`, `xform_points`, `closest_node`, `get_vol_info_from_nii`, `timeseries2nii`; headshape helpers (`downsample_headshape`, `rotate_pointcloud`, `grid_average_downsample`, `replace_headshape`, `save_polhemus_fif`) are the ported `decimate_headshape_points` in preprocessing; `recon_timeseries2niftii` is `apply_lcmv_beamformer(reference_brain="mni")` + `timeseries2nii`; `extract_rhino_files` dropped (see wrappers). Done. |
| `source_recon.beamforming` | `make_lcmv`, `apply_lcmv`, `apply_lcmv_raw`, `load_lcmv`, `get_recon_timeseries`, `transform_recon_timeseries`, `get_lcmv_weights`, `get_leadfields`, `plot_dipole_locations`, `voxel_timeseries`, `make_plots`, `get_beamforming_filenames` | Ported: `lcmv_beamformer` (filters in `filters-lcmv.h5`), `apply_lcmv_beamformer` (= `get_recon_timeseries`, `transform_recon_timeseries`, `voxel_timeseries` with `reference_brain`), `virtual_electrodes`, `plot_bilateral_pairs`; added on `Session`: `lcmv_weights`, `leadfields` (3D leadfields from the `leadfields` branch). `plot_dipole_locations` dropped: the dipole grid is `model-mni-grid.nii.gz` (view with `fsleyes`) and the bilateral pairs have `plot_bilateral_pairs` (reason recorded). Done. |
| `source_recon.minimum_norm` | `create_inverse_operator`, `apply_inverse_operator_surf`, `apply_inverse_operator_vol`, `calc_noise_cov`, `morph_surface`, `get_mne_filenames` | Kept in `freesurfer.py` on `Session`: `minimum_norm`, `apply_inverse` (with the morph), `apply_inverse_vol`, `calc_noise_cov`, `FreeSurferFilenames`. Done. |
| `source_recon.freesurfer_utils` | `setup_freesurfer`, `check_freesurfer`, `recon_all`, `make_watershed_bem`, `make_fsaverage_src`, filename helpers | Kept in `freesurfer.py` (`fsaverage_source_space`, `fetch_fsaverage` added). Done. |
| `source_recon.parcellation` | `load_parcellation`, `load_parcellation_description`, `guess_parcellation`, `vol_parcellate_timeseries`, `surf_parcellate_timeseries`, `resample_parcellation`, `local_orthogonalise`, `symmetric_orthogonalise`, `parcel_centers`, `plot_parcellation`, `plot_psd`, `plot_correlation`, `convert2niftii`, `convert2mne_raw`, `convert2mne_epochs`, `spatial_dist_adjacency`, `parcel_vector_to_voxel_grid`, `convert2source_estimate`, `plot_source_topo`, `assign_voxels_to_binary_parcellation` | Ported: `Parcellation`, `parcellate` (= `vol_parcellate_timeseries`), `parcellate_lcmv`, `_symmetric_orthogonalisation`, `convert_to_mne_raw` (= `convert2mne_raw`), `parcel_vector_to_nifti` (= `convert2niftii`), `plot_parcellation`, `plot_psds`, `save_qc_plots`. Kept from v2 in the same module: `load_parcellation`, `load_parcellation_description`, `guess_parcellation`, `surf_parcellate_timeseries` (on an explicit source space and labels), `resample_parcellation`, `local_orthogonalise`, `parcel_centers`, `plot_psd`, `plot_correlation`, `convert2niftii`, `convert2mne_epochs`, `spatial_dist_adjacency`, `convert2source_estimate` (on an explicit source space), `plot_source_topo`, `assign_voxels_to_binary_parcellation`. Done. |
| `source_recon.sign_flipping` | `find_flips`, `find_template_subject`, `load_covariances`, `covariance_matrix_correlation`, `randomly_flip`, `apply_flips_to_covariance`, `apply_flips`, `time_embed`, `std_data`, and from the `new-sign-flip` branch `find_flips_spectral`, `find_flips_belief_propagation`, `find_flips_sdp` | Ported: `calc_cov`, `calc_corr`, `find_flips` (coordinate ascent), `sign_flip`, `sign_flip_mne_raw`, `apply_flips_to_cov`; `time_embed`/`standardize` in `utils.processing`. Kept from the branch in the same module: the three solvers with `covariance_matrix_correlation` and `apply_flips_to_covariance` (D14: they import and run; the `method` option of `fix_sign_ambiguity` and `sign_flip` selects them). `load_covariances` and `randomly_flip` were internals of the v2 loop; `find_template_subject` is the template choice in the `sign_flip` group step. Done. |
| `source_recon.nii`, `source_recon.utils` | `convert_4dparc_to_3d`, `convert_3dparc_to_4d`, `spatially_downsample`, `append_4d_parcellation`, `find_file` | Keep (`find_file` becomes the osl-files lookup). |
| `utils.file_handling`, `utils.study` | `process_file_inputs`, `find_run_id`, `validate_outdir`, `get_rawdir`, `add_subdir`, `sanitise_filepath`, `osl_print`, `Study` | Keep; they build the session manifest. |
| `utils.logger` | `set_up`, `set_level`, `get_level`, `log_or_print` | Port (`MEEGSessionLogger`, `set_logging_level`); `log_or_print` kept as a thin shim since wrappers call it everywhere. |
| `utils.parallel` | `dask_parallel_bag` | Drop: replaced by the multiprocessing runner (D3). |
| `utils.simulate`, `utils.simulation_config` | `simulate_data`, `simulate_raw_from_template`, `simulate_rest_mvar` | Keep; template and MVAR parameters come from osl-files. The duplicate `simulation_config/simulate.py` module is Drop (same functions as `utils.simulate`). |
| `utils.spmio` | `SPMMEEG`, `Data`, `Channel`, `Montage`, `Trial`, `Event`, helpers | Keep. |
| `utils.opm` | `convert_notts`, `correct_mri` | Keep. |
| `utils.misc`, `utils.package`, `utils.version_utils`, `utils.run_func` | `ensure_dir`, `proc_userargs`, `mean_psd_in_band`, `set_random_seed`, `soft_import`, `run_package_tests`, `check_version`, `osl_func` main | Keep; `run_package_tests` becomes the `osl-ephys-check` CLI (#425). |
| Package data | `source_recon/files/*`, `utils/simulation_config/*`, `neuromag306_info.yml`, `trees/*.tree` | Moved to osl-files (phase 2, done) with old-name aliases; `trees` and the yml stay in the package. Drop: `reduced_hcp-mmp_2mm/8mm` (230-parcel HCP-MMP variant) and `Glasser52_binary_space-MNI152NLin6_res-8x8x8_10mm`, because nothing in the code, docs or examples referenced them and the maintained Glasser files cover the atlas. Everything else is in osl-files, the unmaintained variants under `parcellation/legacy/` with their original names. |

## 10. Test data for osl-files (do this first)

Pulled out of phase 2 because the tests in every later phase need it, and
because osl-files is public: nothing is pushed until the checklist below is
met for each dataset.

### Source recordings

Surveyed 2026-10-15 in `~/projects/ohba/canonical_hmm_networks/BIDS`
(private; never copied anywhere as is).

| Modality | Source | Scanner, rate, length | Channels | Head geometry in file | Identifying content found |
| --- | --- | --- | --- | --- | --- |
| Elekta | sub-01 rest, eyes closed (403 MB fif) | VectorView, 1000 Hz, 300 s | 204 grad, 102 mag, 2 EOG, 1 ECG, 1 EMG, 19 misc, 3 stim; HPI measured | 3 fiducials, 4 HPI coils, 291 headshape points | subject id, first and last name, experimenter, free-text description with names, measurement date, sMRI alongside |
| CTF | sub-02 rest, eyes closed (606 MB .ds + .pos) | CTF-275, 1200 Hz, 305 s | 273 mag, 28 ref, 64 EEG, 42 misc, 1 stim | 3 fiducials, 64 EEG positions; 605-point .pos headshape | `his_id`; .ds sidecar files (`.infods`, `.acq`, `.xml`) carry their own metadata; date already blanked |
| OPM (a) | sub-03 movie run (533 MB fif) | triaxial OPM, 1200 Hz, 638 s | 174 mag (`I6 [X]` naming), 6 bad channels listed | 3 fiducials, 174 sensor points as dig | none in the header; sMRI alongside |
| OPM (b) | sub-05 braille run (441 MB fif) | triaxial OPM, 375 Hz, 1340 s | 192 mag (`A1X` naming), 27 stim; 14766 event annotations | 3 fiducials, 4783 headshape points | none in the header; sMRI alongside |
| EEG | sub-04 rest (BrainVision, 2500 Hz, 1097 s) + `loc.mat` | 62 EEG | 62 ch | electrode positions and head points in the localiser .mat | measurement date; 498 marker annotations incl. free-text comments; the file layout matches the LEMON dataset, whose redistribution terms must be checked |

### What goes to osl-files

Directory `test_data/<modality>/` in osl-files, which the registry workflow
registers automatically. One README per dataset: scanner, site, what was
done to the data, and the terms it is shared under.

| Dataset | Files | Target size |
| --- | --- | --- |
| `test_data/elekta/` | `rest-raw.fif` (fif, float32) | 5 MB: 30 s at 250 Hz, all 306 MEG + EOG/ECG/EMG + one stim channel; misc channels dropped |
| `test_data/ctf/` | `rest-raw.fif` converted from the .ds (keeps the CTF head frame, the 28 reference channels and the 64 EEG), plus a decimated `headshape.pos` | 5 MB: 30 s at 250 Hz. A trimmed `.ds` can be added later if the `.ds` reading path needs its own test (MNE has no .ds writer, so it means a small res4/meg4 writer) |
| `test_data/opm/` | `task-raw.fif` from one of the two OPM runs | 5 MB: 30 s at 250 Hz, all mags, 2 stim channels, events kept |
| `test_data/eeg/` | `rest.vhdr/.vmrk/.eeg` (BrainVision via `mne.export`), electrode positions as a fif montage or the decimated localiser | 3 MB: 60 s at 250 Hz |

Total under 20 MB, against 28 MB of data in osl-files today. Git history is
permanent, so each file is written once and only replaced for a real reason.

Features each file must contain so the tests have something to find: a bad
channel, a high-variance segment, ECG/EOG channels where the scanner has
them, a stim channel with at least a few events, fiducials and a headshape
for coregistration, HPI coil positions in the Elekta file (so the
MaxFilter-related code paths can run without MaxFilter), the CTF reference
channels (`ref_meg` handling), both OPM axis namings if both runs are kept.

### Non-identifiability checklist (every dataset, before pushing)

1. `raw.anonymize()` plus explicit clearing of `subject_info`,
   `experimenter`, `description`, `proj_name`, `his_id`, device serials;
   `meas_date` set to a fixed fake date, not merely shifted.
2. Annotations: keep stimulus codes, drop free-text comments.
3. Headshape: decimate to about 100 points and remove everything below the
   nasion-to-ear plane (no nose or face), using the decimation already in the
   codebase. Fiducials and HPI coils kept.
4. No structural MRI. Coregistration tests use `surface/mni152` from
   osl-files with `allow_mri_scaling`.
5. CTF sidecars (`.infods`, `.acq`, `.xml`, `.hc`) are not shipped; the fif
   conversion carries only what MNE reads.
6. Signal: decided 2026-10-15, synthetic signal over real geometry. The
   real file structure, channel set, sensor positions, fiducials, HPI coils
   and (decimated) headshape are kept; every data channel is overwritten
   with simulated signal: a few dipoles projected through a sphere-model
   forward solution computed from the real sensor geometry, plus 1/f sensor
   noise, synthetic ECG and EOG channels with a scaled blink leaking into
   frontal channels (so ICA has a component to find), a known bad channel,
   a known high-variance segment and a stim channel with known event
   times. Nothing of the participant's time series survives, and the tests
   have ground truth for every artefact.
7. Redistribution rights per dataset, written in its README: who owns the
   recording, that public sharing of a derived segment is permitted, and the
   licence (CC0 or CC-BY for the derived file). The EEG run needs this
   checked first because of the LEMON match. Geometry (fiducials, headshape,
   electrode positions) is derived from the participant even under (B), so
   this check applies to (B) too.
8. A second person inspects the final files (`mne.io.read_raw` + `info`
   dump, annotations, dig) before the PR to osl-files is opened.
9. Neuromag files carry the HPI measurement epochs inside `info`
   (`hpi_meas[i]['hpi_coils'][j]['epoch']`): about 5 MB of the original MEG
   data, which survives `anonymize()` and a data overwrite. Found 2026-10-15
   when the Elekta test file came out 5 MB larger than its data; the
   generator now clears them. Check for this in any other info-carrying
   file that is published.

### How it is made

A script per modality in `osl-files/tools/make_testdata/`, taking the
private source path as an argument, so the files are reproducible and the
anonymisation steps are code, not memory. The script prints the `info` dump
of what it wrote, which goes into the PR description.

### Decisions

- Signal: synthetic over real geometry (decided 2026-10-15, see item 6).
- Geometry: fine to release (decided 2026-10-15). The kept headshape points
  are a random subset, jittered by about a millimetre; fiducials, HPI coils
  and electrode positions are exact because coregistration needs them.
- OPM run: sub-05 only; the pre-coregistered sub-03 is not needed (decided
  2026-10-15).
- CTF: converted fif plus the thinned `.pos`; no trimmed `.ds`.
- EEG: BrainVision (`.vhdr/.vmrk/.eeg`) via `mne.export` with pybv, so the
  `.vhdr` reading path is covered (decided 2026-10-15).
- Size: 20 s at 250 Hz, float32 fif, 60 s for EEG; about 21 MB in total.
- Review: the audit in `make_osl_test_data/audit.py` stands in for a second
  reviewer; the geometry is cleared for release by the data owners. Pushed
  directly to osl-files main in one commit.

### Status: done (pushed to osl-files main, 2026-10-15)

| File | Size | Content |
| --- | --- | --- |
| `test_data/elekta/rest-raw.fif` | 6.3 MB | 20 s at 250 Hz; 204 grad, 102 mag, 2 EOG, ECG, EMG, STI101; 3 fiducials, 4 HPI, 100 headshape |
| `test_data/ctf/rest-raw.fif` + `headshape.pos` | 7.4 MB | 20 s; 273 mag, 28 ref, 64 EEG, UPPT002; CTF compensation kept; 100-point .pos |
| `test_data/opm/task-raw.fif` | 3.9 MB | 20 s; 192 OPM channels (QuSpin triaxial), 2 stim; 3 fiducials, 100 headshape |
| `test_data/eeg/rest.vhdr/.vmrk/.eeg` + `rest-dig.fif` | 3.7 MB | 60 s; 62 EEG, events as markers; montage (62 electrodes + fiducials) in the dig fif |

21.4 MB in total; `test_data/README.md` (generated) documents the ground
truth and the provenance. Every file passed `make_osl_test_data/audit.py`,
which checks the header fields, the dates, the HPI epochs, the acquisition
parameter block, every printable string in the file bytes against tokens
from the original headers, the correlation of every channel with the
original after a 1 Hz high-pass (max 0.10), and that no headshape point is
an original coordinate (min 0.4 mm away). Two leaks were found and fixed on
the way: the HPI measurement epochs (raw MEG data, about 5 MB) and the
Neuromag acquisition parameter block (`info['acq_pars']`, scanner settings
with patient-name fields); neither is cleared by `anonymize()`.

Tests fetch these through the ported files module (phase 2) as
`test_data/<modality>/<file>`.

### Generator scripts

`~/projects/ohba/make_osl_test_data/` (OHBA-internal, not in osl-files):
`common.py` (anonymise, thin and jitter the headshape, build the sphere
forward model, synthesise the signal, inject the artefacts, write and
dump), one short script per modality that loads its private source, picks
channels, crops and calls the common steps, `readme.py` that regenerates
`test_data/README.md`, and `audit.py` that checks the built files against
the originals. They need mne, numpy and pybv, and the private source paths
as arguments.

## 11. Detailed plans for phases 3 to 10

Written 2026-10-16, after phases 1 and 2 landed. The package is restructured
where the ported code makes a cleaner shape than the v2 layout; names users
type (`osl_ephys.preproc`, `osl_ephys.source_recon`, `osl_ephys.glm`,
the config step names) are kept.

### Target package layout

Settled 2026-10-16 after the phase 4 layout review and done the same day
(commits 21f91c7, 8a15bbf and the section split): one subpackage per
config section (`maxfilter`, `preproc`, `coreg`, `source_recon`,
`parcellation`, `group`), each with its own `steps.py`, so a reader finds
a step's code from the section name. `rhino/` is the coregistration
toolbox and sits at the top level (it is OSL's own method and the thing
people import); the FreeSurfer route is one MNE-wrapping module that spans
three sections, so it stays a single file. Which section a step belongs
to is set in its `@step` decorator, not by its file: `rhino/steps.py`
registers `forward_model` under `source_recon`.
`preprocessing/` was renamed `preproc/` on 2026-10-16 to match its config
section. The step modules are imported lazily by the registry on the first lookup
(`registry.load_steps`), so the subpackages can import each other freely.
The shared step helpers are `Session.from_dataset`,
`registry.ignored_options` / `renamed_option` and
`freesurfer.is_freesurfer_route`. The FreeSurfer-route test needs MNE's
fsaverage, which is no longer on this machine (re-fetch with
`mne.datasets.fetch_fsaverage` before phase 7).

```
osl_ephys/
  __init__.py          version; re-exports run_batch, run_chain, Session
  session.py           ported meeg/session.py (Session, SurfaceFilenames, CoregFilenames, check_up_to_date, save_params); shared by rhino, source_recon, parcellation, group
  files/               (done) data from osl-files, neuromag306_info.yml, trees/
  pipeline/            (done) the config API and the runner
    config.py          load_config, validate, section order, old-format conversion, check_version
    registry.py        step registry: @step(section, per="session"|"subject"), find_function
    manifest.py        the session manifest and the output layout
    runner.py          the parallel runner: per-session logs, summary.csv
    chain.py           run_chain: one session through the config
    batch.py           run_batch: manifest x config over the runner, then the group section and the report
    cli.py             the `osl-ephys` command: run, download-data, check, func, ica, maxfilter
  io/                  readers and file-finding helpers
    spm/               spmio (SPMMEEG)
    opm.py             OPM file conversion
    study.py           Study (glob-based file matching)
    file_handling.py   kept helpers
  maxfilter/           phase 6: ported meeg/maxfilter.py
    maxfilter.py       command building, single/multistage/CBU runs, chunking
    steps.py           the `maxfilter` section: one per-session step
  preproc/       phase 6: the `preproc` section
    functions.py       ported meeg/preproc.py + kept osl-ephys functions
    steps.py           every `preproc` step, registered
    qc.py              (done) the preprocessing QC plots and summary.json for the report
    amm.py             ported meeg/amm.py
    ica/               plot_ica GUI, the manual-ICA review server, one apply function
    semp/              from #452
  rhino/               the `coreg` section (RHINO route), split from source_recon/rhino.py
    surfaces.py        extract_surfaces, plot_surfaces, repair_bem_surfaces, surfaces_display
    headshape.py       fiducials and headshape points from fif/pos/elc, stray-point removal, delete_headshape_points
    coregistration.py  coregister_head_and_mri, coreg_error/coreg_rms, plot_coregistration
    forward.py         forward_model on the MNI grid, the BEM check
    fsl.py             setup_fsl, check_fsl, fsleyes, fsleyes_overlay, FLIRT helpers
    _geometry.py       private geometry, transforms, meshes, renderer helpers
    steps.py           the `coreg` steps and `forward_model` (registered under `source_recon`)
  source_recon/        the `source_recon` section
    beamforming/       split from beamforming.py
      lcmv.py          LCMV filters, apply, lcmv_weights, leadfields, virtual electrodes
      bilateral.py     bilateral pairs
      grid.py          the MNI grid of the forward model (_get_mni_grid, _get_source_coords_mni, _get_gridstep)
    freesurfer.py      the FreeSurfer route (recon_all, watershed BEM, coregister, forward model, minimum norm, parcellate); its steps register under coreg / source_recon / parcellation
    steps.py           beamform, minimum_norm, and recon_all / make_watershed_bem (registered under coreg)
  parcellation/        the `parcellation` section
    parcellation.py    parcellate (pca/centroid), orthogonalisation, convert_to_mne_raw, save_as_fif
    plotting.py        parcellation plots (markers, glass brain, surfaces)
    nii.py             nifti point-cloud helpers (from source_recon/nii.py)
    steps.py           parcellate (with source_method) and the _and_parcellate aliases
  group/               the `group` section
    sign_flipping.py   ported data/sign_flipping.py + the scho97 solvers + template selection
    steps.py           sign_flip, and fix_sign_ambiguity (registered under parcellation)
  report.py            (done) the QC report: generate_report, TABS, CAPTIONS, save_figure, save_summary, headless
  glm/                 kept as is
  utils/               logger.py, plotting.py, processing.py, spectral.py, topoplots.py, misc.py, simulate.py
  tests/
```

Gone by the end: `preproc/batch.py` and `source_recon/batch.py`
(replaced by `pipeline/`; no wrappers at the old names, user 2026-10-16),
`preproc/mne_wrappers.py` and `osl_wrappers.py` (merged into
`steps.py`, same function names), `source_recon/rhino.py`,
`source_recon/session.py` (top-level `session.py`), `report/preproc_report.py`,
`src_report.py` and `templates/` (done: `report/` is the module `report.py`), `utils/parallel.py`, `utils/spmio/`,
`utils/opm.py`, `utils/study.py` and `utils/file_handling.py` (moved to
`io/`, done 2026-10-16; `opm.py` lost a dead `if False:` block of
unfinished plotting code). Already gone (2026-10-16): the per-subpackage README.md files
(docstrings now), `utils/package.py`, `utils/version_utils.py`
(`pipeline.config.check_version`), `utils/run_func.py`
(`osl-ephys func`), `utils/create_neuromag306_info.py`.

No re-export shims at the old import paths (user, 2026-10-16): v3 is a
clean break for imports.

One command, `osl-ephys`, with subcommands, replaces the seven scripts;
the old names are removed as each subcommand lands (D19):

| v3 command | Replaces |
| --- | --- |
| `osl-ephys run config.yml sessions.csv outdir` | `osl_preproc`, and `run_src_batch` scripts |
| `osl-ephys download-data` | (new) |
| `osl-ephys check` | `utils.package.run_package_tests` |
| `osl-ephys ica review`, `osl-ephys ica apply` | `osl_ica_label`/`osl_ica_apply` and `osl-ica-review`/`osl-ica-apply` (merged, decision 2026-10-15; done 2026-10-17) |
| `osl-ephys maxfilter files.txt outdir --tsss ...` | `osl_maxfilter` (removed 2026-10-17); builds a one-section config (`maxfilter:`) and runs it, so it is the same code path as `osl-ephys run` |
| `osl-ephys func` | `osl_func` |

### Phase 3: `pipeline/`: one config, one runner

Goal: `run_batch(config, sessions, outdir, n_workers)` runs any mix of
sections over a manifest with the osl-dynamics runner underneath; the
existing `preproc:` configs run unchanged. API as proposed in #454.

Steps, in order, each a commit on `v3` with tests:

1. `utils/logger.py`: port `MEEGSessionLogger` and `set_logging_level`;
   keep `log_or_print` as a shim that writes through the session logger when
   one is active and through the module logger otherwise (the wrappers call
   it everywhere).
2. `pipeline/runner.py`: port `meeg/parallel.py` as is, with the report
   hook made a callback argument rather than an import of `report` (the
   report module is replaced in phase 5). Test: a function that sleeps,
   one that raises, one that warns; check `summary.csv`, the `.log.err`
   rename, and that other items finish.
3. `pipeline/registry.py`: `@step(section, per="session")` decorator that
   records a function under its config name (`filter` for `run_mne_filter`,
   as `find_func` does today, both names accepted) and whether it runs per
   session or per subject; `find_function(name, section, extra_funcs)` with
   the same custom-function rules as today (`extra_funcs` list, module-level
   functions). Register the existing preprocessing wrappers under `preproc`
   by decorating them in place (no behaviour change).
4. `pipeline/config.py`: `load_config` (path, YAML string or dict, as today),
   section order `preproc, coreg, source_recon, parcellation, group`,
   validation (unknown section, unknown step, step in the wrong section,
   per-subject step with no `subject` column), and conversion of a v2
   `source_recon:` list into the three sections with a warning.
5. `pipeline/manifest.py`: `Manifest` from a list of dicts, a CSV or the
   old `(files, subjects, smri_files)` arguments; required columns per
   section (`raw_file` for `preproc`, `smri_file` for `extract_surfaces`,
   `preproc_file` or the standard location for everything after); derived
   paths follow the osl-dynamics layout (`derivatives/preproc/<id>_preproc-raw.fif`,
   `<id>/{bem,coreg,src}`, `surfaces/<subject>`). `utils.file_handling`
   and `Study` become ways to build one.
6. `pipeline/chain.py`: `run_chain(config, session, outdir, ...)` runs the
   sections in order for one session on a dataset dict (`raw`, `events`,
   `ica`, `epochs`, `session`), writing each section's output where the
   manifest says. Today's `run_proc_chain` body moves here; the
   source-recon sections call the existing `source_recon.wrappers` until
   phase 4 replaces them.
7. `pipeline/batch.py`: `run_batch(config, sessions, outdir, n_workers=1,
   log_dir=None, qc_dir=None, overwrite=False, random_seed="auto",
   verbose="INFO", extra_funcs=None, covs=None)`: per-subject steps first
   (grouped by `subject`), then the per-session loop through the runner,
   then `group` steps, then the report callback. `run_proc_batch` and
   `run_src_batch` stay as wrappers that build the equivalent call and
   warn; `dask_client` raises with the message to use `n_workers`;
   `strictrun`, `ftype` and `skip_save` are gone (D9). No job-array mode:
   batches run interactively on a many-core machine (D11).
8. `pipeline/cli.py`: `osl-ephys run`, `download-data`, `check` (D13);
   `osl_preproc` aliases `run`. `argparse` with subcommands.
9. Remove `dask` and `distributed` from the dependencies and `utils/parallel.py`;
   `test_parallel.py` becomes the runner test; `examples/parallelisation/`
   rewritten.

Done when: the four test recordings run through a `preproc:` config in
parallel with `n_workers=2`, a failing session is reported and the others
finish, and the old batch calls still work with a warning.

Risks: `run_proc_chain` is 250 lines of accumulated special cases (report
generation, epoching, ICA, file naming); move it whole first, split it
after the tests cover it.

### Phase 4: `source_recon/` on the osl-dynamics design

Goal: the ported rhino, beamformer, parcellation and sign flipping behind
the same config step names, on the `Session` file layout.

1. `session.py`: port as is. The manifest (phase 3) builds a `Session` per
   row; `head_model_id` comes from the `subject` column when surfaces are
   shared.
2. `rhino.py`: port `meeg/rhino.py` whole (3860 lines; it already is one
   module). Keep from the old `rhino/` package what it lacks, as public
   functions in the same module or in `utils`: `delete_headshape_points`
   (interactive), `plot_polhemus_points`, `save_mni_fiducials`,
   `rescale_sensor_positions`, the fsleyes helpers, `extract_rhino_files`,
   `recon_timeseries2niftii`/`timeseries2nii`, `get_vol_info_from_nii`.
   Each of the 35 old `rhino/utils.py` names gets its inventory line here.
3. `beamforming.py`: port `meeg/source_recon.py`; `filters-lcmv.h5` is the
   filter file. Keep `get_recon_timeseries`, `transform_recon_timeseries`,
   `get_lcmv_weights`, `get_leadfields` (from the `leadfields` branch),
   `plot_dipole_locations`, `voxel_timeseries` on top of the ported
   filters. The #426 bilateral implementation merged into `v3` is replaced
   by the ported one; its QC plot and tests are kept where they add
   something.
4. `parcellation.py`: port the osl-dynamics module; keep
   `surf_parcellate_timeseries`, `convert2source_estimate`,
   `convert2mne_epochs`, `local_orthogonalise`, `resample_parcellation`,
   `spatial_dist_adjacency`, `plot_correlation`, `plot_source_topo`,
   `guess_parcellation`, `load_parcellation_description`,
   `assign_voxels_to_binary_parcellation`.
5. `sign_flipping.py`: port `data/sign_flipping.py` plus `time_embed` and
   `standardize` and test it on synthetic data with a known flip vector
   (D14: no benchmark). The `new-sign-flip` methods merged from scho97 are
   kept as additional functions if they import and run on the test data,
   otherwise dropped with the reason in the inventory.
6. `minimum_norm.py` and `freesurfer.py`: onto `Session` (file names from
   `CoregFilenames`), otherwise unchanged. Tested with MNE's `fsaverage`
   (`mne.datasets.fetch_fsaverage`, which ships the surfaces, BEM and
   `aparc` annotation, so no FreeSurfer binaries are needed): coregister
   the Elekta test recording to fsaverage, surface forward model, minimum
   norm, surface parcellation (D15). `recon_all` and `make_watershed_bem`
   are tested at the command-building level only.
7. `steps.py`: every v2 config step name (`extract_fiducials_from_fif`,
   `compute_surfaces`, `coregister`, `forward_model`, `beamform`,
   `parcellate`, `beamform_and_parcellate`, `minimum_norm`, ...) as a thin
   registered step over the new functions; `compute_surfaces` registered
   `per="subject"`. Old keyword names map to new ones with a warning where
   they differ.
8. Delete the `rhino/` package, the old `beamforming.py` and
   `wrappers.py`, `coreg.html` becomes an option of `plot_coregistration`.
9. Migration page: old to new output paths, with a note that old
   derivatives directories are not read by v3.

Done when: coregistration, forward model, LCMV and parcellation run on the
Elekta, CTF and OPM test recordings against the MNI152 surfaces in CI
(no FSL), the EEG recording runs through coregistration with
`allow_mri_scaling`, and a FSL-marked test runs `extract_surfaces` on a
machine that has FSL.

Risks: the largest phase; the FreeSurfer route has no test data (no
surface files in osl-files), so it is kept working by inspection and a
skipped test until a small FreeSurfer example exists.

### Phase 5: `report/` in the table style (done 2026-10-17)

Goal: one `qc/report.html` across sessions, built from what each step
saved, replacing the per-subject jinja2 pages.

1. `report/report.py`: port `generate_report`, `TABS`, `CAPTIONS`; extend
   `TABS` with the preprocessing plots osl-ephys has. Each tab lists the
   WebP files a step saves under `qc/<step>/<id>/`.
2. `report/qc.py`: the osl-ephys plots that osl-dynamics lacks, as
   functions that take the dataset and write WebP files plus a
   `summary.json` of numbers: flowchart, raw data, channel time series,
   sensors, channel distributions, spectra, frequency bands, digitisation,
   head movement, EOG/ECG summaries, bad ICA components, bad segments and
   channels summaries, artefact scan, events, custom figures (every `plot_*`
   in the inventory).
3. `save_qc_plots` in each domain (preprocessing, coregistration,
   parcellation) calls those; the per-session dict of numbers goes into the
   table: percentage of time marked bad, bad channel count, ICA components
   rejected, head movement summary (from the MaxFilter head position when
   available), EOG and ECG event counts, coregistration error, MNI
   registration score, empty parcels (D17).
4. Image format, as osl-dynamics does it: QC plots are written straight to
   WebP by matplotlib (`savefig(path.webp, dpi=100, bbox_inches="tight",
   pil_kwargs={"quality": 80})`, about 1000 pixels wide, the width the
   report shows them at, several times smaller than PNG); plots that live
   with the derivatives (coregistration, surfaces, MNI registration) stay
   PNG there and are converted to WebP when the report copies them into the
   QC directory (Pillow, quality 80, in a thread pool, skipped when the copy
   is newer than the source, written to a temporary name and moved into
   place). A `report.save_figure(fig, path)` helper holds the savefig
   settings so every QC plot in osl-ephys uses the same ones; the pyvista
   coregistration screenshots go through Pillow the same way.
5. `gen_report_from_fif` stays as the way to regenerate a session's QC from
   a saved fif and its sidecars.
6. Delete `preproc_report.py`, `src_report.py`, `templates/`, `jinja2` and
   `tabulate` from the dependencies.
7. The `osl_ica_label` workflow (phase 6) reads what it needs from the saved
   fif and ICA files, not from report data.

Done when: a batch run of the test recordings produces a report whose
table sorts by each metric and whose session view shows every plot the v2
report had for the same data.

Done 2026-10-17: `report.py` (one module: generate_report, TABS, CAPTIONS,
METRICS, save_figure, save_summary, headless) and `preproc/qc.py` (every v2
`plot_*` as a function returning the figure, `scan_summary`,
`save_qc_plots` writing summary.json + WebP to qc/preproc/<id>/,
`gen_report_from_fif`, `plot_artefact_scan` kept as a standalone function).
The tabs: Preprocessing, Spectra, Artefacts, Acquisition, Surfaces, MNI
Registration, Coregistration, Parcellation, Sign flipping, plus a Custom
tab discovered from custom_*.webp files. The table columns: bad segments
(%), bad channels, ICA excluded, EOG/ECG event counts, head movement (mm,
from cHPI at 1 s steps when recorded), MNI registration/overlap/MI, coreg
rms, sign flip correlation. The group `sign_flip` step writes
qc/sign_flip/summary.json (template + before/after correlation per
session) and sign_flip.webp; parcellation writes qc/parc/<id>/summary.json
next to its plots. The surfaces and coregistration plots stay PNG with the
derivatives and are copied to WebP by the report. Saving QC plots runs
under the Agg backend (`report.headless`) so no window opens.
`preproc_report.py`, `src_report.py`, `templates/`, jinja2 and tabulate
are gone. `ica_label` regenerates the session's ICA plot under qc/ (phase
6 rewrites the script). Also in the table: bad segments per channel type,
epoch counts, the smallest number of dipoles in a parcel
(`parcellation.parcel_dipole_counts`, the empty-parcel metric of D17; an
empty parcel itself is an error from `parcellate`) and the manifest's own
extra columns.

### Phase 6: `preproc/` and `maxfilter/` (done 2026-10-17)

1. (done 2026-10-17) `preproc/functions.py`: port `meeg/preproc.py`. Map the old
   osl-ephys functions onto it: `gesd` and `bad_segments`/`bad_channels`
   become the ported `detect_bad_segments`/`detect_bad_channels` with the
   #444 fixes carried over (compare outputs on the test data before
   switching); `detect_maxfilt_zeros`, `drop_bad_epochs`,
   `detect_bad_channels_psd` and the zscore/GLM steps stay as they are.
2. (done 2026-10-17) `preproc/steps.py`: `mne_wrappers.py` and `osl_wrappers.py` merged,
   every function keeping its name; new steps `ica_label` (mne-icalabel
   route, #427 multi-axis OPM), `ica_ecg_eog_correlation`,
   `decimate_headshape_points`, `amm`. osl-dynamics' `plot_ica_components`
   is not ported: `preproc.qc.plot_bad_ica` shows the same components with
   their properties.
3. (done 2026-10-17) `preproc/amm.py`: port, with the `amm` step.
4. (done 2026-10-17) `preproc/ica/`: `gui.py` (the GUI, was `plot_ica.py`),
   `manual/` (the browser review package, was `manual_ica/`), `review.py`
   (`review_gui`, `serve_review`), `apply.py` (`apply_review`: the browser
   labels when `outdir/ica/<id>/label.txt` exists, else the GUI's
   `exclude`, written back to the preprocessed file and the QC ICA plot
   updated) and `files.py` (the run's layout). The CLI is `osl-ephys ica
   review outdir [ids] [--reject all|manual] [--browser]` and `osl-ephys
   ica apply outdir [ids]`; the four script names are removed (D19). The
   `manual_ica` and `apply_manual_ica` steps take the session id and the
   run's output directory (`outdir/ica/<id>/`) unless an extra_func set
   `subject`/`target_pth`. The GUI's Evoked branch called MNE internals
   that no longer exist (`_butterfly_onpick`, `plt_show`); it now uses
   MNE's `ICA.plot_sources` for Evoked input (the labelling GUI is for
   Raw and Epochs). The browser review page is the primary tool
   and the GUI the alternative; the review server's label format is the
   one both workflows write.
4b. `preproc/semp/` is outside this phase (D18, own board item): it
   stays as merged, its `osl_pathfinder` import is raised with its author
   (suggesting the name `osl-pathfinder`), and it is excluded from the docs
   build until that dependency is settled.
5. (done 2026-10-17) `maxfilter/`: port `meeg/maxfilter.py`. Maxfilter becomes a config
   section (decided 2026-10-16): `maxfilter: [{maxfilter: {scanner: Neo,
   mode: multistage, tsss: true, headpos: true, movecomp: true}}]`, a
   per-session step on `raw_file` writing `outdir/maxfilter/<id>_tsss.fif`
   (the ported keyword options map one-to-one onto the step options), and
   the `preproc` section reads that file when the section ran. A config
   with only a `maxfilter` section is the licensed-machine use. The
   `osl-ephys maxfilter` subcommand (and the `osl_maxfilter` alias) keeps
   the file-list-plus-flags interface by building that one-section config.
   Tests at the command-building level (no MaxFilter licence in CI).
6. `glm/`: unchanged.
7. (done 2026-10-17) Issues folded in: #373 (the `filter`, `notch_filter`
   and `savgol_filter` steps warn when the data is non-negative everywhere,
   as a rectified or power signal is), #440 done.

Done when: the preprocessing section of the config runs every step on the
test recordings, and the ICA review round-trips a label file through both
the GUI's save path and the server's.

### Phase 7: tests (done 2026-10-17)

Flaky: `test_batch_preproc.py::TestPreprocessingBatch::test_batch_mixes_existing_and_new_outputs` failed once in a full run on 2026-10-16 (when it still went through the v2 wrapper) and passed on rerun; rewritten on `run_batch` it passed five isolated runs and every full run since, so it is treated as gone.

Tests are written in each phase above; this phase is the top-up and the
CI shape.

1. `tests/conftest.py`: session-scoped fixtures for each test recording
   (fetched once; `OSL_DATA` cached between CI runs with `actions/cache`),
   a temporary derivatives directory, and a two-session manifest.
2. Coverage targets by module: config loading and validation, every
   registered step on the matching modality, chain and batch with
   `n_workers=2`, runner failure isolation, parcellation maths
   (orthogonalisation, parcel weights) against known inputs, sign flipping
   on synthetic data with a known flip vector, the coreg-to-parcellation
   path on MNI152 surfaces, report generation, every CLI `--help`, the
   alias table.
3. `pytest.mark.fsl` and `pytest.mark.freesurfer`, skipped unless
   `FSLDIR`/`FREESURFER_HOME` are set; a scheduled workflow on a self-hosted
   or larger runner can run them later.
4. `osl-ephys check` (#425): imports every subpackage, prints versions,
   fetches one test file, runs a 10-second preprocessing chain.

Done 2026-10-17: `conftest.py` (Agg backend, session-scoped `recordings`
and `two_sessions` fixtures, the `fsl` and `freesurfer` markers skipped
without `FSLDIR`/`FREESURFER_HOME`); `test_preproc_steps.py` (configs
exercising the preproc steps on each of the four recordings, the
unknown-option error, the rectified-signal warning);
`test_parcellation_maths.py` (PCA and centroid parcel time courses on the
parcellation's own voxels, symmetric orthogonalisation, dipole counts);
`test_sign_flipping.py` (coordinate ascent and the spectral and SDP
solvers recover known flips); `test_cli.py` (every subcommand's `--help`,
which caught that `func` had no parser); `test_fsl.py` (surface
extraction on the MNI152 brain, 3 minutes, passes locally; skipped in
CI); the two source-reconstruction tests carry the `fsl` marker too, as
the coregistration needs FLIRT even on the MNI152 surfaces; the osl-files data cached between CI runs. The FreeSurfer-route
test stays conditional on MNE's fsaverage being downloaded. 152 tests.

### Phase 8: examples and docs

1. `examples/`: one directory per scanner (`elekta`, `ctf`, `opm`, `eeg`)
   with the v3 scripts in the osl-dynamics `meg_preproc` shape (config at
   the top, manifest, `run_batch`), plus `parallelisation/` removed (the
   runner is the default) and `misc/` pruned to what still runs.
   `toolbox-paper/` and `oxford/` frozen with a v2.4 note.
2. Tutorials: the preprocessing, coregistration and source-recon tutorials
   rewritten on the v3 API; the short ones run on the osl-files test
   recordings so the gallery can execute them again; the long ones stay on
   OSF data and are not executed in the docs build.
3. The 152 Sphinx warnings: docstring rst fixes module by module.
4. `install.rst` conda-first (done), `documentation.rst` and `faq.rst`
   refreshed, a `migration.rst` page (config sections, output layout,
   removed `dask_client`, renamed parcellation files, `coreg.html`, the
   dropped `glm_irasa`), `llms.txt` kept current.
5. README quick example in the osl-dynamics style.

Done 2026-10-17: `examples/{elekta,ctf,eeg,opm}` rewritten as numbered
scripts sharing a `sessions.py` (manifest at the top, config, `run_batch`);
`examples/misc` pruned to four scripts on the v3 API (headshape deletion,
sMRI sform fix, FreeSurfer route, sign flipping of arrays);
`examples/parallelisation` removed (the runner is the default);
`examples/oxford/` (added 2026-10-17 from the `ohba-examples` branch, whose
content is identical to the frozen v2 copy) has the OHBA pipelines on the v3
API: `automated_ica`, `manual_ica` (`ica_autoreject: {apply: false}` then
`review_gui`), `ssp` (MNE projectors on the preprocessed file) and
`maxfilter` (`no_viz` dropped the same day: the plotting problems it
worked around are fixed) (scanner option, `check_scanner.py`);
`examples/elekta` uses generic `cal`/`ctc` files instead of `scanner: Neo`.
The `leadfields` branch (3D leadfields) was already ported as
`beamforming.leadfields(session, orientation=None)`.
`examples/v2/` dropped (user, 2026-10-17): the toolbox paper's scripts
are `examples/wakeman_henson/` on the v3 API (with the stats, which needed
the GLM steps registered for the `group` section and `_run_group` passing
the sessions' `<id>_glm.pkl` files as `dataset["glm"]`), the MNE-only
`wh_mne` walkthrough is its `freesurfer_route.py` (replacing
`misc/freesurfer_source_recon.py`), the beamformer comparison paper script
is `misc/beamformer_comparison.py` on the Session API, and the v2
`toolbox-paper` environment file is left in git history.
Previously `examples/v2/` held the frozen toolbox-paper, oxford, wh_mne and
beamformer-comparison scripts with a README. Tutorials: the three
preprocessing ones, `source-recon_coreg`, `source-recon_subject`,
`source-recon_deleting-headshape-points` and the osl-ephys calls of
`source-recon_batch` rewritten on the v3 API (Session, rhino functions,
run_batch, `osl-ephys ica review`, the QC report); they stay on the OSF
datasets and are rendered without execution (the test recordings are too
short and synthetic for what the tutorials show). The two SEMP tutorials
(`preprocessing_eeg-fmri`, `preprocessing_manual-ica`) still use
`run_proc_batch`/`find_func` and belong to the SEMP item. Docs: `documentation.rst` with the pipeline example (a migration page was
written and then removed on the user's request, 2026-10-17; the release
notes carry that content),
`faq.rst` rewritten, `llms.txt` and the README example. Sphinx warnings:
101 -> 14, the 12 left are docstrings in `preproc/semp` and
`preproc/ica/manual` (SEMP item); autoapi no longer documents the tests
or re-exported members, the templates and gallery README are excluded.
Issues: #352/#361 (tutorial problems) addressed by the rewrite; #386
fixed (coregistration no longer loads the headshape file when
`use_headshape: false`, and `coreg_error` returns NaN without one).

### Phase 9: the OHBA conda channel

Dropped 2026-10-17 (D21): osl-ephys is pip-only. The channel test below
passed end to end before the decision; the recipes and workflow are in git
history at 968f43f if a conda package is ever wanted again.

Revised 2026-10-17 (D20). Needs from the user: the anaconda.org
organisation (name to be chosen; `ohba` and `ohba-analysis` are both
unregistered on 2026-10-17) and its API token as the GitHub secret
`ANACONDA_API_TOKEN` in osl-ephys and osl-dynamics.

a. (done 2026-10-17) `recipe/osl-ephys/recipe.yaml`, built from the
   repository checkout with the version from `OSL_EPHYS_VERSION`, so the
   workflow does not wait for PyPI, with two outputs as in osl-dynamics:
   `osl-ephys-base` (noarch, the code) and `osl-ephys` (a metapackage built
   per platform; on Linux x86-64 it adds `vtk *osmesa*` for headless
   servers; a condition on a noarch package would be baked in for every
   platform, which broke the macOS install); `recipe/{glmtools,anamnesis,sails}`
   from the PyPI sdists (glmtools imports tabulate without declaring it;
   sails pins importlib-metadata <5 and reads requirements files its sdist
   lacks, so its recipe leaves the pin out, skips pip check and supplies
   empty files). All four build and pass their tests locally with
   rattler-build (`mamba create -n rb -c conda-forge rattler-build`;
   `OSL_EPHYS_VERSION=3.0.0.dev0 rattler-build build --recipe-dir recipe
   -c conda-forge --output-dir <scratch>`).
b. (done 2026-10-17) `.github/workflows/conda.yml`: on a published release,
   or by hand with a version, rattler-build the recipe directory on a Linux
   runner with `--skip-existing all` against the channel (the three
   dependencies are built once), upload the new packages, then install
   osl-ephys from the channel on Linux and macOS and run `osl-ephys check`.
   Needs the `ANACONDA_API_TOKEN` secret (an organisation token with API
   write and upload scopes). A development tag (`v*.dev*`) also runs it.
   Passed end to end on 2026-10-17 with `v3.0.0.dev7` (run 39068904695):
   the metapackage for five platforms, installs on Linux (Python 3.13) and
   macOS (3.14) with `osl-ephys check` passing. The development builds
   (osl-ephys 3.0.0.dev4 to dev7, osl-ephys-base 3.0.0.dev7) are still on
   the channel and should be deleted there before the release; the git
   tags are gone.
c. (done 2026-10-17) The install page and README use the channel.

The earlier conda-forge plan, not pursued (kept for reference):

1. glmtools: we will not submit a conda-forge recipe ourselves (decided
   2026-10-16), and conda-forge needs every run dependency on conda-forge.
   So the conda recipe omits glmtools, `osl_ephys.glm` imports glmtools
   lazily (inside the functions that use it, with a clear error naming
   `pip install glmtools`), and the install page says the GLM module needs
   that extra pip install in a conda environment. The pip package keeps
   glmtools as a required dependency. If glmtools ever reaches conda-forge,
   the recipe adds it.
2. `osl-ephys` recipe modelled on `osl-dynamics-feedstock`: `noarch`,
   the `osl-ephys` entry point, import tests per subpackage, `vtk *osmesa*`
   on linux-64 for headless coregistration plots; verify
   `plot_coregistration` with `PYVISTA_OFF_SCREEN` in the recipe test.
3. mne-installers PR adding `osl-ephys` (and osl-dynamics) after checking
   the installer's pinned MNE/Qt stack resolves with ours.

Done 2026-10-17: glmtools AND sails are imported lazily
(`utils.misc.lazy_import`; sails on conda-forge is 1.5.18, which has no
`glm_periodogram`, so it is pip-only for us too) with a `pip install
glmtools sails` error on use; `osl_ephys` imports without them (tested);
the install page says so. `recipe/recipe.yaml` is the conda-forge recipe
(rattler-build, modelled on osl-dynamics-feedstock): noarch, both entry
points, every other dependency from conda-forge, `vtk *osmesa*` on
linux-64, import tests per subpackage and an off-screen
`plot_coregistration` import. There is NO osl-ephys feedstock on
conda-forge yet (the install page's conda instructions are for after the
release): at release, submit `recipe/recipe.yaml` with the sdist's sha256
to conda-forge/staged-recipes (the first conda release of osl-ephys).
mne-installers afterwards: `recipes/mne-python/construct.yaml` pins (on
2026-10-17) mne 1.13.2 (pyside6 build), pyside6 6.11.2, pyvistaqt 0.13.1,
pyvista 0.49.1, trame 3.13.2 (<4), trame-vuetify 3.2.6, mne-icalabel
0.10.0, nilearn 0.14.1, numba 0.68.0, python 3.14; nothing in our run
dependencies pins against these, and CI now also tests Python 3.14. The
PR adds `osl-ephys =3.0.0` (and `osl-dynamics-base`) once the conda
package exists.

### Phase 10: release v3.0.0

1. Changelog from the board and the inventory's drop lines.
2. `v3.0.0rc1` as a GitHub pre-release: proves the release workflow and
   gives a week of testing on PyPI.
3. Delete `doc/v3_plan.md`, moving anything still useful into the
   release notes.
4. `v3.0.0` release; squash-merge PR #453 (user, 2026-10-17), which makes
   `.git-blame-ignore-revs` dead (its only entry was the v3 black reformat;
   deleted 2026-10-17); close the five v2 PRs with a pointer;
   revisit the 48 open issues (board item). No v2.4.x
   maintenance branch: v3 is the only line from here.
