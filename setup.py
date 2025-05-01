import pathlib

from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# Requirement categories
reqs = [
    "numpy",
    "scipy",
    "matplotlib",
    "mne",
    "scikit-learn",
    "fslpy",
    "sails",
    "tabulate",
    "pyyaml",
    "neurokit2",
    "jinja2",
    "glmtools",
    "numba",
    "nilearn",
    "dask",
    "distributed",
    "parse",
    "opencv-python",
    "panel",
    "h5io",
]
doc_reqs = ["sphinx", "numpydoc", "sphinx_gallery", "pydata-sphinx-theme"]
dev_reqs = ["setuptools", "pytest", "pytest-cov", "coverage", "flake8"]

name = "osl-ephys"

setup(
    name=name,
    version="2.2.dev0",
    description="OHBA Software Library for the analysis of electrophysiological data",
    long_description=README,
    long_description_content_type="text/markdown",
    author="OHBA Analysis Group",
    license="MIT",
    # Choose your license
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.12",
    install_requires=reqs,
    extras_require={
        "dev": dev_reqs,
        "doc": doc_reqs,
        "full": dev_reqs + doc_reqs,
    },
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "osl_maxfilter = osl_ephys.maxfilter.maxfilter:main",
            "osl_ica_label = osl_ephys.preprocessing.ica_label:main",
            "osl_ica_apply = osl_ephys.preprocessing.ica_label:apply",
            "osl_preproc = osl_ephys.preprocessing.batch:main",
            "osl_func = osl_ephys.utils.run_func:main",
        ]
    },
    packages=[
        "osl_ephys",
        "osl_ephys.tests",
        "osl_ephys.report",
        "osl_ephys.maxfilter",
        "osl_ephys.preprocessing",
        "osl_ephys.utils",
        "osl_ephys.utils.spmio",
        "osl_ephys.source_recon",
        "osl_ephys.source_recon.rhino",
        "osl_ephys.glm",
    ],
    package_data={
        "osl_ephys": [  # Simulations
            "utils/simulation_config/*npy",
            "utils/simulation_config/*fif",
            # Channel information
            "utils/neuromag306_info.yml",
            # Parcellation files
            "source_recon/files/*gz",
            # Report templates
            "report/templates/*",
            # READMEs
            "*/README.md",
        ]
    },
    command_options={
        "build_sphinx": {
            "project": ("setup.py", name),
            "version": ("setup.py", name),
            "release": ("setup.py", name),
        }
    },
)