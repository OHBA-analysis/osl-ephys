"""Print the recording date and scanner of each fif file, to choose the
scanner option of MaxFilter (see the README).

This script needs a computer with a MaxFilter licence, for show_fiff.
"""

from subprocess import PIPE, run

files = [
    "raw/sub-001_task-rest.fif",
    "raw/sub-002_task-rest.fif",
]

for file in files:
    cmd = f"/neuro/bin/util/show_fiff -v -t 100:206 {file}"
    result = run(cmd.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    lines = result.stdout.decode().splitlines()
    date = lines[0].split("    ")[-1]
    scanner = lines[1].split("    ")[-1]
    print(f"{file}\n  date:    {date}\n  scanner: {scanner}")
