"""
Challenge
=========

Which parts of the spectrum show a statistically significant group average for the first level effect of the VEOG channel?

Let's run the permutations for the age effect:
"""

group_contrast = 0  # Group mean
firstlevel_contrast = 3  # VEOG

P = osl_ephys.glm.MaxStatPermuteGLMSpectrum(gglmsp, group_contrast, firstlevel_contrast, nperms=50, nprocesses=1)

critical_value = P.perms.get_thresh(100 - 5)
print(critical_value)

plt.figure(figsize=(9, 9))
ax = plt.subplot(111)

osl_ephys.glm.plot_sensor_spectrum(gglmsp.f, gglmsp.model.tstats[group_contrast, firstlevel_contrast, :, :].T,
                                   gglmsp.info, base=0.5, ax=ax, sensor_proj=True)

xl = ax.get_xlim()

critical_value = P.perms.get_thresh(100 - 5)
ax.hlines(critical_value, xl[0], xl[1], 'k')
ax.hlines(-critical_value, xl[0], xl[1], 'k')

critical_value = P.perms.get_thresh(100 - 1)
ax.hlines(critical_value, xl[0], xl[1], 'r')
ax.hlines(-critical_value, xl[0], xl[1], 'r')

ax.set_ylim(-20, 20)

# There are very wide spread changes - the peak is in frontal sensors at frequencies below around 15Hz
