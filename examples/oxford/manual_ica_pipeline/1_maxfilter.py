"""Example script for maxfiltering raw data recorded at Oxford using the new scanner.

Note: this script needs to be run on a computer with a MaxFilter license.
"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from osl.maxfilter import run_maxfilter_batch

# Setup paths to raw (pre-maxfiltered) fif files
input_files = [
    "data/raw/file1.fif",
    "data/raw/file2.fif",
]

# Directory to save the maxfiltered data to
output_directory = "data/maxfilter"

# Run MaxFiltering
#
# Note:
# - We don't use the -trans option because it affects the placement of the head during coregistration.
# - See the /maxfilter directory for further info.
run_maxfilter_batch(
    input_files,
    output_directory,
    "--scanner Neo --mode multistage --tsss --headpos --movecomp",
)
