Contributing to osl-ephys
==========================

osl-ephys accepts bug reports (as issues), bug fixes (as pull requests) and new features (as pull requests).


## Issues (Bug Reports)
If you have found a bug or problem with the code, please [open an issue](https://github.com/OHBA-analysis/osl-ephys/issues/new) on github. An EXCELLENT bug report contains the following ingredients:

* **Summary** A concise description of the the bug.
* **Versions and context** Which osl-ephys/Python versions (eg osl-ephys v0.0.1 on Python 3.9) and which operating system (eg Debian 11.2, Mac OSX 10.15 or Windows 10) are you using?
* **Steps to reproduce** How can we reproduce the issue? - this is very important. Please use (`) to format inline code and (```) to format code blocks
* **What is the expected correct behavior?** What should happen instead of this current behaviour?
* **Relevant logs and/or screenshots** Paste any relevant logs - please use code blocks (```) to format console output, logs, and code.
* **Possible fixes** If you can, link to the line of code that might be responsible for the problem, any suggestions for fixes are welcome!

Small and simple bugs might not need all this info but generally a more thorough description will help find a quick solution.

## Pull Requests (Bug Fixes and New Features)

If you want to contribute new code or documentation to osl-ephys - you'll need to open a 'pull request'. There are two ways to do this.

### Pull request from fork

If you have developed a new features or bug fix on your own copy (fork) of OSL and would like to contribute it back to osl-ephys. Then you can [create a pull request from your forked version of osl-ephys](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork).

Please create the pull request with the **base** branch set to **main** on osl-ephys and the **compare** branch set to your contribution. Please include an informative title and a clear description of what changes you have made and what the expected behaviour should be. An osl-ephys developer will review the code and potentially ask for some changes before the code can be merged into osl-ephys. In some cases a pull-request may be rejected.

### Pull request from osl-ephys branch

If you are an osl-ephys developer, or working with the osl-ephys team, you can make contributions by creating a new branch directly on the osl-ephys repository. Please create the branch relative to **main** and ensure that it stays up to date with subsequent changes to the **main** branch. Once you're ready to start sharing the branch. You can make a [pull request directly from the branch on osl-ephys](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request). If you would like to discuss and share changes but not merge them yet - please create a 'draft pull request'.

## Developer Notes

The core developer team is @matsvanes, @woolrich, @cgohil8 and @ajquinn.

### Installation

```
conda create -n osle python=3.12
conda activate osle
git clone git@github.com:OHBA-analysis/osl-ephys.git
cd osl-ephys
pip install -e ".[dev]"
```

### Tests

```
pytest osl_ephys/tests
```

### Code style

The code is formatted with [black](https://black.readthedocs.io) and spell-checked with [codespell](https://github.com/codespell-project/codespell); both run on every pull request:

```
black osl_ephys
codespell
```

### Documentation

```
pip install -r doc/requirements.txt
sphinx-build -b html doc build
```

### Releases

See the Releases section of the README.
