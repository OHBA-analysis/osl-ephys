"""
Frequency Spectrum Estimation - Simulations & General Linear Models
===================================================================

Frequency-domain analyses of oscillations in electrophysiological recordings of brain activity contain information about rhythms in the underlying neuronal activity. Many aspects of these rhythms are of interest to neuroscientists studying EEG and MEG time series. Many advanced methods for spectrum estimation have been developed in recent years, but the core approach has been the same for many years.

This tutorial will introduce the concept of power spectrum estimation using the standard approach in electrophysiology: Welch's Periodogram. We will describe this approach with simulations and explore a recent update that merges Welch's method with a General Linear Model to define a GLM-Spectrum.


Getting started
***************

We start with some preparation, let's import ``numpy``, ``scipy.signal`` and ``osl_ephys`` for analysis and ``matplotlib`` for some visulistions.

"""


import matplotlib
import numpy as np
import sails

import osl_ephys

font = {'size' : '22'}

matplotlib.rc('font', **font)
import matplotlib.pyplot as plt

#%%
# Spectrum Estimation
# *******************
#
# A frequency spectrum is a description of how the variance in a time domain signal can be distributed across a range of frequencies. This is a core analysis approach across science and engineering. The underlying mathematics is built upon the Fourier transform which is one of the fundamental algorithms in signal processing.
#
# The Fourier transform can then be used to compute a frequency spectrum to describe the signal. The profile of the frequency spectrum describes how energy in the time-domain signal is distrbuted across frequency.
#
# A huge number of resources provide an introductory of power spectra and the Fourier transform. Here are a few for wider reading (after the workshop!)
#
#  - `Fourier Transform Wikipedia Page <https://en.wikipedia.org/wiki/Fourier_transform>`_
#  - `Spectral Density Wikipedia Page <https://en.wikipedia.org/wiki/Spectral_density>`_
#  - `But what is the Fourier Transform? A visual introduction. (YouTube) <https://www.youtube.com/watch?v=spUNpyF58BY>`_
#  - `Welch's method for smooth spectral decomposition (YouTube) <https://www.youtube.com/watch?v=YK1F0-3VvQI>`_
#
#
# In EEG and MEG analysis, we're typically intrested in using these methods to identify and describe any oscillations that might be present in a time-series. This is performed by many papers in many different analysis but the main computation of the spectrum itself tends to be computed in a consistent way - Welch's Periodogram.
#
# Let's take a look at this standard approach using a simulated signal.
#
# We define a 10 second signal that is sampled at 128Hz. The signal will have one stationary (constant amplitude) oscillation at 10Hz and one oscillation with decreasing amplitude at 22Hz.
#


# Define some parameters
sample_rate = 128
seconds = 10
time = np.linspace(0, seconds, seconds*sample_rate)

# Stationary oscillation
f1 = np.sin(2*np.pi*10*time)
# Decreasing amplitude oscillatoin
f2_amp = np.linspace(1, 0.5, seconds*sample_rate)
f2 = f2_amp * np.sin(2*np.pi*22*time)

# Final signal
xx = f1 + f2

# Quick plot
plt.figure(figsize=(16, 9))
plt.plot(time, f1+7, 'g')
plt.plot(time, f2+4, 'm')
plt.plot(time, xx, 'k')
plt.xlabel('Time (seconds)')
# Some annotations
plt.text(5, 5.5, '+')
plt.text(5, 2.5, '=')

#%%
# The constant 10Hz oscillation in green mixes with the decaying 22Hz oscillation in magenta to create out final signal in black. We can see a complicated mix of oscillations at the start of the final signal which gradually becomes a single oscillation as the 22Hz amplitude decreases.
#
# Let's take a look at how a Fourier based frequency spectrum would describe this signal. We can compute Welch's periodogram using the ``sails.stft`` library. Many other implementations exist in libraries like ``mne`` and ``scipy.signal`` - we'll use ``sails`` as it contains some convenient options to help us visualise the analysis.


# Compute the spectrum with Welch's method
pdg = sails.stft.periodogram(xx, nperseg=sample_rate, fs=sample_rate)

#%%
# ``pdg`` is a very simple python object that contains our spectrum result. It hold four pieces of information.
#
# - ``pdg.config`` : the technical specification detailing how the spectrum was computed
# - ``pdg.f`` : the frequency values of the result
# - ``pdg.t`` : the time values of the sliding windows in the result (if they are present)
# - ``pdg.spectrum`` : the spectrum itself
#
# Let's take a look at the result.

# Simple plot
plt.figure(figsize=(7, 7))
plt.plot(pdg.f, pdg.spectrum)
plt.title("Welch's Periodogram")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.tight_layout()

#%%
# This is the classic first approach to spectrum analysis.
#
# We see two clear peaks, one large peak at 10Hz and a smaller peak at 22Hz. The 22Hz peak is smaller due to it's decreasing amplitude in the time-domain.
#
# Under the hood, Welch's method computes many Fourier transforms on sliding window data segments across the dataset - this is also known as a Short Time Fourier Transform (STFT).
#
# We can visualise the STFT by stopping the periodogram taking the final average. Let's set ``average=None`` in the periodogram before visualising the STFT and it's average across time.
#

# Compute the short-time Fourier transform (unaveraged periodogram)
pdg = sails.stft.periodogram(xx, nperseg=sample_rate, fs=sample_rate, average=None)

# Print out some helpful infoo
print('-'*20)
print('{} time segments'.format(len(pdg.t)))
print('{} frequency bins'.format(len(pdg.f)))
print('{}Hz frequency resolution'.format(np.diff(pdg.f)[0]))
print('-'*20)

# Simple visualisation
plt.figure(figsize=(18,9))
plt.subplot(121)
plt.pcolormesh(pdg.f, pdg.t, pdg.spectrum, cmap='hot_r')
plt.title("STFT")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Time (seconds)')
plt.subplot(122)
plt.plot(pdg.f, pdg.spectrum.mean(axis=0))
plt.title("Welch's method")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')

#%%
# The critical choice in computing the periodogram is the length of the sliding window. This is set by ``nperseg`` (Number-of-samples PER SEGment - following the ``scipy`` naming convention).
#
# We've set ``nperseg=sample_rate`` in this example, which is typically a sensible starting point. Let's try a shorter value and see what happens.


# Compute the short-time Fourier transform (unaveraged periodogram)
pdg = sails.stft.periodogram(xx, nperseg=sample_rate//4, fs=sample_rate, average=None)

# Print out some helpful infoo
print('-'*20)
print('{} time segments'.format(len(pdg.t)))
print('{} frequency bins'.format(len(pdg.f)))
print('{}Hz frequency resolution'.format(np.diff(pdg.f)[0]))
print('-'*20)

# Simple visualisation
plt.figure(figsize=(18,9))
plt.subplot(121)
plt.pcolormesh(pdg.f, pdg.t, pdg.spectrum, cmap='hot_r')
plt.title("STFT")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Time (seconds)')
plt.subplot(122)
plt.plot(pdg.f, pdg.spectrum.mean(axis=0))
plt.title("Welch's method")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')

#%%
# We can see several changes here. Firstly, the frequency resolution is much lower. The peaks in the final spectrum look much chunkier as we now only have 1 frequency bin for every 4 Hz. This happens as the FFT can only return as many frequency components as there are samples in a segment - so a shorter segment will have fewer frequency estimates spread over the same range.
#
# Secondly, we have many more time segments. There were 19 in the first example and 79 in the second. This means that we now have fewer time segments in the average, this doesn't make much difference here but can be important in noisy data - more on that later.
#
# Let's try a longer ``nperseg``...



# Compute the short-time Fourier transform (unaveraged periodogram)
pdg = sails.stft.periodogram(xx, nperseg=sample_rate*3, fs=sample_rate, average=None)

# Print out some helpful infoo
print('-'*20)
print('{} time segments'.format(len(pdg.t)))
print('{} frequency bins'.format(len(pdg.f)))
print('{}Hz frequency resolution'.format(np.diff(pdg.f)[0]))
print('-'*20)

# Simple visualisation
plt.figure(figsize=(18,9))
plt.subplot(121)
plt.pcolormesh(pdg.f, pdg.t, pdg.spectrum, cmap='hot_r')
plt.title("STFT")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Time (seconds)')
plt.subplot(122)
plt.plot(pdg.f, pdg.spectrum.mean(axis=0))
plt.title("Welch's method")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')

#%%
# As we'd expect by now, the longer window results in a higher frequency resolution and fewer windows.
#
# There is no right or wrong choice for window length as different analyses may want to emphasise one or the other depending on the research question. This trade off between time resolution and frequency resolution just needs to be set to provide a useful representation of the data for the case in hand. We recommend trying a range of values to explore it's effect when first exploring a new dataset.
#

# Challenge 1
# ***********
#
# Can you tweak the code below to change the ``sample_rate`` of the simulated data instead of the window length?
#
# Try a few different values (eg 64 and 256, or any positive value that your like..), what aspect of the spectrum is altered when we change the sample rate?
#

# Define some parameters
sample_rate = 128   # <- Change this parameter!!
seconds = 10
time = np.linspace(0, seconds, seconds*sample_rate)

# Stationary oscillation
f1 = np.sin(2*np.pi*10*time)
# Decreasing amplitude oscillatoin
f2_amp = np.linspace(1, 0.5, seconds*sample_rate)
f2 = f2_amp * np.sin(2*np.pi*22*time)

# Final signal
xx = f1 + f2

# Compute the short-time Fourier transform (unaveraged periodogram)
pdg = sails.stft.periodogram(xx, nperseg=sample_rate*3, fs=sample_rate, average=None)

# Print out some helpful infoo
print('-'*20)
print('{} time segments'.format(len(pdg.t)))
print('{} frequency bins'.format(len(pdg.f)))
print('{}Hz frequency resolution'.format(np.diff(pdg.f)[0]))
print('-'*20)

# Simple visualisation
plt.figure(figsize=(18,9))
plt.subplot(121)
plt.pcolormesh(pdg.f, pdg.t, pdg.spectrum, cmap='hot_r')
plt.title("STFT")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Time (seconds)')
plt.subplot(122)
plt.plot(pdg.f, pdg.spectrum.mean(axis=0))
plt.title("Welch's method")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')


# But what about those dynamics?
# ******************************
#
# Welch's method computes a single spectrum by taking the average across the time-windows. This feels appropriate for the 10Hz signal, but we can see that this may not represent the 22Hz signal well. The standard spectrum gives this peak the appearance of a single amplitude but this is actually the 'average' amplitude across a range of windows.
#
# The GLM-Spectrum method replaces this simple average with a multiple regression model which provides way to quantify changes in the spectrum across the sliding window time segments.
#
# The advantage of a GLM-Spectrum is that we can extend the model to describe more than just the mean term. This has a several advantages including some modelling of temporal dynamics and the abililty to accout for covariates and confounds when computing the spectrum.
#
# Here, we define a single covariate regressor describing a linear trend in time and fit GLM-Spectrum with a mean term and the linear trend covariate. We add the linear trend using the `reg_ztrans` keyword argument to `glm_spectrum` this specifies that we're adding a regressor and that we want the values in that regressor to be z-transformed prior to the regression.
#


# Define our covariate
cov = {'Linear': np.linspace(-1, 1, seconds*sample_rate)}

# Compute the GLM-Spectrum
glmsp = osl_ephys.glm.glm_spectrum(xx, nperseg=sample_rate, fs=sample_rate, reg_ztrans=cov)

# Simple visualisation
plt.figure(figsize=(18,9))
plt.subplot(121)
plt.plot(glmsp.f, glmsp.copes[0, :])
plt.title("Mean Cope-Spectrum")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.subplot(122)
plt.plot(glmsp.f, glmsp.copes[1, :])
plt.title("Linear Trend Cope-Spectrum")
plt.xlabel('Frequency (Hz)')

#%%
# The output GLM-Spectrum now has an extra dimension - the first contains the mean term which is identical to a standard spectrum in this case. The second dimension contains a spectrum of GLM parameter estimates describing the effect of a linear trend across time segments. This is zero for the 10Hz peak as it doesn't change in amplitude over the sliding window time segments - however the 22Hz peak has a large, negative parameter estimate suggesting that it decreases in amplitude over time.
# This GLM-Spectrum output class contains information about the GLM design matrix and contrasts in ``glmsp.design``, information about the STFT to be modelled in ``glmsp.data`` and the fitted GLM in ``glmsp.model``.
#
# Let's visualise the design matrix.



fig = glmsp.design.plot_summary()

#%%
# The top part of this figure contains the design matrix. This matrix is built from the input covariates. It contains one column predictor and one row per sliding window time segment in the STFT. The table at the bottom contains contrasts. These are linear weightings between regressors that can be useful for comparing parameter estimates. In this case (and for this whole tutorial) we'll keep things simple and use simple contrasts that isolate each regressor one at a time.
#
# ``glmspec.model.betas`` contains the fitted parameter estimates of the GLM-Spectrum. This uses the design matrix above to predict the power of the STFT across sliding windows. A separate GLM is fitted for each frequency bin in the STFT, so we end up with a spectrum of parameter estimates for each regressor.
#
# Let's take a look:



plt.figure(figsize=(9, 6))
plt.pcolormesh(glmsp.f, np.arange(1,3), glmsp.model.betas)
plt.yticks([1,2])
plt.colorbar()
plt.ylabel('Regressors')
plt.xlabel('Frequency (Hz)')
plt.yticks([1,2], ['Mean', 'Linear Trend'])
plt.title('Beta Spectrum Estimates')


# We can combine the design matrix and model parameters to compute model predictions. Here we compute the predicted spectral power at 22Hz for each time segment.



x = glmsp.design.design_matrix[:, 1]

freq_idx = 22

mean = glmsp.model.betas[0, freq_idx]
slope = glmsp.model.betas[1, freq_idx]

y = mean + x * slope

plt.figure(figsize=(9, 9))
plt.plot(glmsp.design.design_matrix[:, 1], glmsp.data.data[:, freq_idx], '.')
plt.plot(x, y)
plt.legend(['22Hz Power over time', 'Model Prediction'])
plt.xlabel('Time (demeaned)')
plt.ylabel('Power')
plt.title('GLM-Spectrum model fit')
plt.ylim(-0.05, 0.35)

#%%
# We can see that the model prediction is a reasonable fit to the data. Try rerunning this cell with a different frequency index - we didn't simulate a linear trend for any other frequencies, so what would you expect to see?
#
#
# Once more, with noise
# *********************
# This is all great but we're missing a critical ingredient from real data - Noise. Data recordings aren't sine waves so they can only be so instructive for real data analysis.
#
# Let's modify out simulation to include some noise and, to be even more realistic, let's make somme high frequency noise that changes over time. We'll also compute a longer time series of 100 seconds, rather than just 10.
#
# We compute our noise by computing the gradient (difference between adjacent time points) of some normally distribiuted white noise. The white noise will have a constant power spectrum across all frequencies. The gradient operation removes some of the slower drifts in the noise as it keeps only the difference between adjacent time-points and discards trends across many time points. This is a quick, convenient way of simulating noise with relatively high frequency activity.
#
# Let's generate our data.


# Define some
sample_rate = 128
seconds = 100
time = np.linspace(0, seconds, seconds*sample_rate)

# Stationary oscillation
f1 = np.sin(2*np.pi*10*time)
# Decreasing amplitude oscillatoin
f2_amp = np.linspace(1, 0.5, seconds*sample_rate)
f2 = f2_amp * np.sin(2*np.pi*22*time)

# Final signal
xx = f1 + f2

# Add some high-frequency noise which changes over time.
np.random.seed(42)
noise_ratio = 1
noise = np.gradient(np.random.randn(*xx.shape) * xx.std()*noise_ratio)
noise_freq = 0.1
artefact_amp = np.cos(2*np.pi*noise_freq*time + np.pi) + 1
yy = xx + artefact_amp*noise

# And a little bit of normal white noise
noise2 = np.random.randn(*xx.shape) * 0.25
yy = yy + noise2

# Quick plot
plt.figure(figsize=(16, 9))
plt.plot(time, f1+12, 'g')
plt.plot(time, f2+9, 'm')
plt.plot(time, artefact_amp*noise+5)
plt.plot(time, yy-4, 'k')
plt.xlabel('Time (seconds)')
# Some annotations
plt.text(10, 10.25, '+')
plt.text(10, 7.25, '+')
plt.text(10, 1.5, '=')
plt.xlim(0, 25)

#%%
# Our 10Hz and 22Hz oscillations behave as before, but our new noise component in blue dominates several parts of the signal. Note that we're only visualising the first 20 seconds of the 100 second simulation - change the ``xlim`` parameter to zoom in or out of the dataset.
#
# This is likely to impact our spectrum estimate - let's take a look by recomputing our GLM-Spectrum. Remember that we're fitting model with two regressors, one constant term and one z-transformed linear trend.



cov = {'Linear': np.linspace(-1, 1, seconds*sample_rate)}
glmsp = osl_ephys.glm.glm_spectrum(yy, nperseg=sample_rate, fs=sample_rate,
                             mode='magnitude', reg_ztrans=cov)

plt.figure(figsize=(18, 9))
plt.subplot(121)
plt.plot(glmsp.f, glmsp.copes[0, :])
plt.title("Mean Cope-Spectrum")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.subplot(122)
plt.plot(glmsp.f, glmsp.copes[1, :])
plt.title("Linear Trend Cope-Spectrum")
plt.xlabel('Frequency (Hz)')

#%%
# Our oscillatory peaks are still visible but there is now a noisy background to our spectrum. Luckily, as we know something about how the noise changes over time, we can add this to our model to try and attenuate its effect.
#
# We added the linear-trend covariate as a z-transformed regressor (``reg_ztrans``) - this regressor is zero-mean so whilst it models interesting dynamics it does not impact the parameter estimate of the mean term (though it can impact the standard error of that estimate).
#
# In contrast, we do want the artefact regressor to impact our estimate of the mean term. Specifically, we will add the artefact amplitude as a positive-valued regressor scaled between 0 and 1 (``reg_unitmax``). This additional non-zero mean regressor changes the interpretation of our constant regressor. It no longer models the mean, but the intercept of the overall model. The intercept is the modelled value where all predictors are zero, so this can be interpreted as the mean of the data after having removed the variability explained by the artefact regressor.
#
# Ok, let's fit the model and take a look at the design matrix.


# Define covariates
cov = {'Linear': np.linspace(-1, 1, seconds*sample_rate)}
con = {'Artefact': artefact_amp}
# Compute GLM-Spectrum
glmsp = osl_ephys.glm.glm_spectrum(yy, nperseg=sample_rate, fs=sample_rate,
                             mode='magnitude', reg_ztrans=cov, reg_unitmax=con)

fig = glmsp.design.plot_summary()

#%%
# We now have three regressors, our constant term and linear trend are included as before but we now have an additional 'Artefact' regressor whose values are scaled between zero and one. We can see that the value of the artefact fluctuates over time following the dynamics of the simulated noise source.
#
# In this case, we know the dynamics of the artefact as we've designed our own simulation. This is unlikely to be the case for real data but we can still create meaningful regressors from potential source of artefact. This might include EOG channels recording eye movements, bad segment annotations in the dataset or head movements estimated from maxfilter. Any of these potental artefact sources can be processed and added to the design matrix using ``reg_unitmax``.
#
# Let's visualise the fitted GLM-Spectra of this model


# Visualise all three COPEs
plt.figure(figsize=(18,9))
plt.subplots_adjust(wspace=0.4)
plt.subplot(131)
plt.plot(glmsp.f, glmsp.copes[0, :])
plt.title("Intercept\nCope-Spectrum")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')

plt.subplot(132)
plt.plot(glmsp.f, glmsp.copes[1, :])
plt.title("Linear Trend\nCope-Spectrum")
plt.xlabel('Frequency (Hz)')

plt.subplot(133)
plt.plot(glmsp.f, glmsp.copes[2, :])
plt.title("Artefact Term\nCope-Spectrum")
plt.xlabel('Frequency (Hz)')

#%%
# The Intercept term now contains a spectrum estimate without interference from the dynamic noise. In addition, the spectrum of the noise itself has been modelled by our non-zero mean regressor. The third plot shows that the noise component peaks around 30Hz.
#
# We see this effect even more clearly when plotting up the mean term of the original model against the intercept term of the model including the noise regressor.


cov = {'Linear': np.linspace(-1, 1, seconds*sample_rate)}
con = {'noise': artefact_amp}
glmsp1 = osl_ephys.glm.glm_spectrum(yy, nperseg=sample_rate, fs=sample_rate,
                              mode='magnitude', reg_ztrans=cov)

glmsp2 = osl_ephys.glm.glm_spectrum(yy, nperseg=sample_rate, fs=sample_rate,
                              mode='magnitude', reg_ztrans=cov, reg_unitmax=con)


plt.figure(figsize=(18, 9))
plt.subplot(121)
plt.plot(glmsp1.f, glmsp1.copes[0, :])
plt.plot(glmsp2.f, glmsp2.copes[0, :], '--')
plt.legend(['Original Model', 'Noise Model'])
plt.title("Mean/Intercept Cope-Spectrum")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.ylim(0, 0.0055)
plt.subplot(122)
norm_factor = glmsp2.design.design_matrix[:, 2].mean()
plt.plot(glmsp2.f, glmsp2.copes[2, :]*norm_factor)
plt.title("Artefact Term Cope-Spectrum")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.ylim(0, 0.0055)

#%%
# We can see that the noise is effectively supressed in the intercept term, and that the difference between the two models is captured by the cope-spectrum of the noise regressor itself.
#
# Great - this can be really useful for including confound regressors into our spectrum estimate in cases where we might know something about the dynamics of a noise source. In real EEG/MEG data this might come from movement, or blinking.
#
# Finally, we often want to go beyond a point estimate for an effect to get a statistical estimate that incorporates the variability around an estimate. For example, we may have a very large point estimate for our linear trend effect, but if the data are very noisy then we may not want to trust that estimate.
#
# We can use the GLM to compute a t-statistic to do this for us.



cov = {'Linear': np.linspace(-1, 1, seconds*sample_rate)}
glmspec = osl_ephys.glm.glm_spectrum(yy, nperseg=sample_rate, fs=sample_rate, reg_ztrans=cov)

plt.figure(figsize=(18, 9))
plt.subplot(131)
plt.plot(glmspec.f, glmspec.model.betas[0, :])
plt.title("Mean Cope-Spectrum")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.subplot(132)
plt.plot(glmspec.f, glmspec.model.copes[1, :])
plt.title("Linear Trend Cope-Spectrum")
plt.xlabel('Frequency (Hz)')
plt.subplot(133)
plt.plot(glmspec.f, np.sqrt(glmspec.model.varcopes[1, :]))
plt.title("Linear Trend Varcope-Spectrum")
plt.xlabel('Frequency (Hz)')

# The t-statistic is then the cope divided by the square root of the varcope.
plt.figure(figsize=(9,6))
plt.plot(glmspec.f, glmspec.model.tstats[1, :])
plt.title("Linear Trend t-spectrum")
plt.xlabel('Frequency (Hz)')
plt.ylabel('t-statistic')
plt.ylim(-20, 20)

#%%
# Spectra as Regressions
# ***********
#
# With this framework, we can apply any and all of the methods of multiple regression to our spectra.
#
# For example, we can easily plot metrics such as the R-square spectrum to explore how much variance is explained at each frequency in the spectrum

plt.figure(figsize=(7, 7))
plt.plot(glmspec.f, glmspec.model.r_square[0, :])
plt.title("R-square Spectrum")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Variance Explained')


#%%
# Challenge 2
# ***********
# As we now have a regression model, we can compute the spectrum that the GLM-Spectrum would predict for different combinations of the predictor variables.
#
# This is a linear prediction with a classic standard form. For the specific model we have been using, the equation looks like this.
#
# spectrum = InterceptParameter * InterceptValue + LinearParameter * LinearValue
#
# We can see the predictor values by looking at the design matrix.

print(glmsp.design.design_matrix)

# The intercept value is always the same with a value of 1. This is as we expect as the intercept does not vary with the values of our other predictor variables. The Linear predictor values vary between around -1.64 and +1.64, these slightly odd values are what we get when we z-transform a straight line.
#
# We can see the regression parameter estimates in the fitted model, let's look specifically at the betas for 22Hz. Remember we have vectors of coefficents for every frequency.

print(glmsp.model.betas)

# Now the challenge, can you use the information above to write some code to plot the GLM model predicted spectrum for the start, middle and end of the data?
#
# .. code-block:: python
# 
#     start_spec =  # <- your code goes here!
#     middle_spec = # <- your code goes here!
#     end_spec =    # <- your code goes here!

plt.figure()
plt.plot(glmsp.f, start_spec)
plt.plot(glmsp.f, middle_spec)
plt.plot(glmsp.f, end_spec)
plt.legend(['Start', 'Middle', 'End'])
plt.title('GLM predicted spectra')

#%%
# OPTIONAL - Confound regression in detail
# ****************************************
# This is an optional section going into detail on the intuition behind confound regression. This can be a tricky concept, even for those who are already familiar with regression. Here we try to provide some insight by visualising the difference between a model with a single constant regressor, and a model with one constant regressor and one non-zero mean regressor.
#
# So, let's dig into why this noise supression works. We'll fit three models in this section. The first is by a very simple model with a single constant regressor.



con = {'artefact': artefact_amp}

# A simple model with a single, constant regressor
glmsp_meanonly = osl_ephys.glm.glm_spectrum(yy, nperseg=sample_rate, fs=sample_rate,
                              mode='magnitude')

plt.figure()
fig = glmsp_meanonly.design.plot_summary()

#%%
# This first model is equivalent to computing the mean of the data.
#
# Our second model includes a covariate regressor based on the dynamic amplitude of the artefact component in the signal. This is designed to quantify the part of our data that covary with the artefact amplitude.
#
# Critcially, this regressor is z-transformed prior to fitting the model. This means that we have a zero-mean regressor. Let's take a look.


# A two regressor model with a constant and a covariate containing the z-transformed artefact amplitude
glmsp_artefact_ztrans = osl_ephys.glm.glm_spectrum(yy, nperseg=sample_rate, fs=sample_rate,
                                             mode='magnitude', reg_ztrans=con)

plt.figure()
fig = glmsp_artefact_ztrans.design.plot_summary()

#%%
# This model is the standard straight line equation $y = mx + c$.
#
# - $y$ contains our data to be described - in this case a single frequency of an STFT.
# - $x$ is a predictor value that we want to use to describe variability in $y$.
# - $m$ is a gradient describing the 'slope' relationship between $x$ and $y$.
# - $c$ is the 'intercept' term that describes where the line crosses the y-axis.
#
# This is the standard form of this equation. $x$ and $y$ are known in advance whilst the intercept and slope terms, $c$ and $m$ are estimated by the regression. In our model above, $m$ is the parametere from the 'Constant' regressor and $x$ is the parameter from the 'artefact' regressor.
#
# In the GLM literature, this equation is often written in terms of $\beta$ values and design matrices. In this form, our second model might look like this:
#
# $$ y = \beta_0 \begin{bmatrix} 1 \\ 1 \\ \vdots \\ 1 \end{bmatrix} + \beta_1 \begin{bmatrix} x_{1} \\ x_{2} \\ \vdots \\ x_{m} \end{bmatrix} $$
#
# The critical difference is that all values to be estimated are consistently notated with $\beta$ and the predictor values for all regressors are made explicit. Here, the intercept `c` is $\beta_0$ and `x` is $\beta_1$ and the two vectors are the two columns of our design matrix.
#
# In the second model, the vector of predictors had been z-transformed to be zero-mean. For our final model we'll instead use a unit-max scaling to create a non-zero mean second regressor.
#


# A two regressor model with a constant and a covariate containing the unit-max scaled artefact amplitude
glmsp_artefact_unitmax = osl_ephys.glm.glm_spectrum(yy, nperseg=sample_rate, fs=sample_rate,
                              mode='magnitude', reg_unitmax=con)

plt.figure()
fig = glmsp_artefact_unitmax.design.plot_summary()

#%%
# This design is very similar to our second, only the second column has changed. All the maths and interpretations above still hold.
#
# However, ths change in scaling makes an important difference to the interpretation of the fitted model parameters. Most importantly, the interpretation of $\beta_0$ as the intercept term is critically dependant on the scaling used in the second column.
#
# :*note*: Though all models contain a constant regressor, these are not doing the same thing. We must remember that the interpretation of each regressor changes when we add new ones. The interpretation of the terms in our model can only be created by considering the model as a whole.
#
# Let's take a closer look at the Mean term fitted by the first model. This computes a simple average across the data observations that weights each observation equally.
#
# Let’s look at single frequency, and visualise how the power at that frequency varies over time as a histogram, with the mean power estimate annotated by a black vertical line.


freq_idx = 45

# Mean from simple model
mean = glmsp_meanonly.model.betas[0, freq_idx]
print('Estimated mean : {}'.format(mean))

plt.figure(figsize=(9, 9))
h = plt.hist(glmsp_meanonly.data.data[:, freq_idx], 32)
plt.vlines(mean, 0, 40, 'k')
plt.xlabel('Magnitude Estimate')
plt.ylabel('Num Time Segments')
plt.legend(['Mean', 'Data Histogram'], frameon=False)
plt.title('Single Mean-term', fontsize=22)

#%%
# We can see that the mean is ``0.000832``.
#
# In contrast, the model with the added artefact term computes an intercept instead of a mean. We can visualise this by plotting a scatter graph with the value of the noise regressor on the x-axis and the data observations on the y-axis. These y-axis data are the same as the values used in the histogram when visualising the simple model.
#
# The artefact-regressor models a slope effect describing the extent to which the data observations increase with our predictor. The intercept models the data points where this line crosses zero - in other words, it models the data where the noise predictor has a value of zero.
#
# Let's take a look.



freq_idx = 45

# Intercept - 'c' or 'beta_0'
beta0 = glmsp_artefact_ztrans.model.betas[0, freq_idx]
# Slope - 'm' or 'beta_1'
beta1 = glmsp_artefact_ztrans.model.betas[1, freq_idx]
print('Estimated intercept : {}'.format(beta0))

# Visualise effects
plt.figure(figsize=(9,9));

# Scatter plot
plt.plot(glmsp_artefact_ztrans.design.design_matrix[:, 1], glmsp_artefact_ztrans.data.data[:, freq_idx], 'o')

# Intercept
plt.plot([-0.1, 0.1], [beta0, beta0], 'k--', lw=4)

# Slope effect
x_pred = beta0 + np.linspace(-1.5, 1.5)*beta1
plt.plot(np.linspace(-1.5, 1.5), x_pred, lw=4)

plt.legend(['Data observations', 'Intercept Term', 'Noise Effect'], frameon=False)
plt.xlabel('Artefact regressor value')
plt.ylabel('Magnitude Estimate')
plt.title('Constant + ztrans(artefact)', fontsize=22)

#%%
# Now we can clearly see the straight line equation in action.
#
# Each dot of the scatter plot is from a particular time-segment with the magnitude of our frequency-of-interest in the y-axis and the 'artefact' regressor value in the x-axis.
#
# Our fitted intercept, where the artefact regressor is zero, is shown in a black line and the full fitted straight line in orange. In this case, our estimate of the intercept is identical to the simple mean term from our first model.
#
# This happens as our artefact regressor has a mean value of zero, which acts to centre our data exactly around its mean point. What happens when we don't have a zero-mean regressor? Let's take a look at the final model.



freq_idx = 45

# Intercept - 'c' or 'beta_0'
beta0 = glmsp_artefact_unitmax.model.betas[0, freq_idx]
# Slope - 'm' or 'beta_1'
beta1 = glmsp_artefact_unitmax.model.betas[1, freq_idx]
print('Estimated intercept : {}'.format(beta0))

# Visualise effects
plt.figure(figsize=(9,9));

# Scatter plot
plt.plot(glmsp_artefact_unitmax.design.design_matrix[:, 1], glmsp_artefact_unitmax.data.data[:, freq_idx], 'o')

# Intercept
plt.plot([-0.1, 0.1], [beta0, beta0], 'k--', lw=4)

# Slope effect
x_pred = beta0 + np.linspace(0, 1)*beta1
plt.plot(np.linspace(0, 1), x_pred, lw=4)

plt.legend(['Data observations', 'Intercept Term', 'Noise Effect'], frameon=False)
plt.xlabel('Artefact regressor value')
plt.ylabel('Magnitude Estimate')
plt.title('Constant + unitmax(artefact)', fontsize=22)

#%%
# Only the scaling of our predictor values has changed, but you can quickly see that this has a large effect on the value of the intercept!
#
# The x-axis now crosses zero only where the value of our artefact regressor is equal to zero rather than in the middle of the data-distribution. As a result the intercept is much smaller that what we estimated with our first two models.
#
#
# We can double check this by combining our visualisations.
#
#



# Visualise effects
plt.figure(figsize=(18,6));
plt.subplots_adjust(wspace=0.4)

plt.subplot(131)
h = plt.hist(glmsp_meanonly.data.data[:, freq_idx], 32, orientation='horizontal')
plt.hlines(mean, 0, 40, 'k')
plt.ylabel('Magnitude Estimate')
plt.xlabel('Num Time Segments')
plt.legend(['Mean', 'Data Histogram'], frameon=False)
plt.title('Single Mean-term', fontsize=16)

# ----------------------------------------------

plt.subplot(132)
# Intercept - 'c' or 'beta_0'
beta0 = glmsp_artefact_ztrans.model.betas[0, freq_idx]
# Slope - 'm' or 'beta_1'
beta1 = glmsp_artefact_ztrans.model.betas[1, freq_idx]

# Scatter plot
plt.plot(glmsp_artefact_ztrans.design.design_matrix[:, 1], glmsp_artefact_ztrans.data.data[:, freq_idx], 'o')

# Intercept
plt.plot([-0.1, 0.1], [beta0, beta0], 'k--', lw=4)

# Slope effect
x_pred = beta0 + np.linspace(-1.5, 1.5)*beta1
plt.plot(np.linspace(-1.5, 1.5), x_pred, lw=4)

plt.legend(['Data observations', 'Intercept Term', 'Noise Effect'], frameon=False)
plt.xlabel('Artefact regressor value')
plt.ylabel('Magnitude Estimate')
plt.title('Constant + ztrans(artefact)', fontsize=16)

# ----------------------------------------------

plt.subplot(133)
# Intercept - 'c' or 'beta_0'
beta0 = glmsp_artefact_unitmax.model.betas[0, freq_idx]
# Slope - 'm' or 'beta_1'
beta1 = glmsp_artefact_unitmax.model.betas[1, freq_idx]

# Scatter plot
plt.plot(glmsp_artefact_unitmax.design.design_matrix[:, 1], glmsp_artefact_unitmax.data.data[:, freq_idx], 'o')

# Intercept
plt.plot([-0.1, 0.1], [beta0, beta0], 'k--', lw=4)

# Slope effect
x_pred = beta0 + np.linspace(0, 1)*beta1
plt.plot(np.linspace(0, 1), x_pred, lw=4)

plt.legend(['Data observations', 'Intercept Term', 'Noise Effect'], frameon=False)
plt.xlabel('Artefact regressor value')
plt.ylabel('Magnitude Estimate')
plt.title('Constant + unitmax(artefact)', fontsize=16)

#%%
# Our three panels share the same y-axis scale. Whilst the intercept of the first two models describes the centre of the whole distribution. In contrast, the inclusion of the non-zero mean covariate in the third model changes this drastically. It's intercept models the centre of the data distribution where the artefact covariate is zero. As a result the intercept estimate is much smaller than the other two.


#%%
# Futher reading
# **************
#
#   Andrew J. Quinn, Lauren Z. Atkinson, Chetan Gohil, Oliver Kohl, Jemma Pitt, Catharina Zich, Anna C. Nobre, Mark W. Woolrich; The GLM-spectrum: A multilevel framework for spectrum analysis with covariate and confound modelling. Imaging Neuroscience 2024; 2 1–26. doi: https://doi.org/10.1162/imag_a_00082
