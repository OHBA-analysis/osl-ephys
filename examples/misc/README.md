# Scripts for particular situations

- `delete_headshape_points.py`: remove the headshape points that spoil a
  coregistration, by hand, and rerun it.
- `fix_smri_files.py`: give a structural MRI a standard sform when RHINO
  complains about it.
- `sign_flipping_arrays.py`: sign flipping of parcel time courses held in
  arrays or MATLAB files, outside the pipeline.
- `beamformer_comparison.py`: the localisation error of an LCMV beamformer
  on evoked fields with a RHINO or a FreeSurfer forward model, on the data
  of the beamformer comparison paper.

The FreeSurfer route (surface source space, minimum norm) is shown on a
real dataset in `../wakeman_henson/2_coregister_freesurfer.py` and
`3_source_reconstruct_freesurfer.py`.
