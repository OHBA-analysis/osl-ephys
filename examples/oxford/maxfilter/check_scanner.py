"""Example script for checking the data and which scanner data was recorded with.

This script has to be run on a computer with a MaxFilter license.
"""

from subprocess import PIPE, run

files = [
    "/ohba/pi/knobre/datasets/covid/rawbids/sub-004/meg/sub-004_task-restEO.fif",
    "/ohba/pi/knobre/datasets/meguk/raw/S6001_AR_B1.fif",
]

print()
for file in files:
    cmd = f"/neuro/bin/util/show_fiff -v -t 100:206 {file}"
    result = run(cmd.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    stdout = result.stdout.splitlines()
    date = str(stdout[0]).split("    ")[-1][2:-1]
    scanner = str(stdout[1]).split("    ")[-1][2:-1]

    print(file)
    print(f"date:    {date}")
    print(f"scanner: {scanner}")
    print()


print("See: https://github.com/OHBA-analysis/osl/tree/examples/examples/oxford/maxfilter for what to use for the --scanner argument.")
