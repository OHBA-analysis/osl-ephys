# CTF example

The headshape points and fiducials come from a Polhemus `.pos` file
(`pos_file` in the sessions). Without one, fiducials marked on the MRI can
be used instead: replace the two headshape steps of `2_coregister.py` by
`save_mni_fiducials` (a text file with the nasion, LPA and RPA in MNI mm)
and coregister with `use_headshape: false`.

1. `1_preprocess.py`: filtering, downsampling and artefact removal. CTF
   axial gradiometers are picked as `mag` with `ref_meg: false`.
2. `2_coregister.py`: surfaces and coregistration.
3. `3_source_reconstruct.py`: the forward model, beamforming and
   parcellation with the AAL parcellation.
4. `4_sign_flip.py`: the dipole sign ambiguity across sessions.
5. `5_save_npy.py`: the parcel time courses as numpy arrays, for
   osl-dynamics.
