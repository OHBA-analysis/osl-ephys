"""
Challenge 2 - SOLUTION!
=======================

As we now have a regression model, we can compute the spectrum that the GLM-Spectrum would predict for different combinations of the predictor variables.

This is a linear prediction with a classic standard form. For the specific model we have been using, the equation looks like this.

spectrum = InterceptParameter * InterceptValue + LinearParameter * LinearValue

We can see the predictor values by looking at the design matrix.
"""

print(glmsp.design.design_matrix)

# The intercept value is always the same with a value of 1. This is as we expect as the intercept does not vary with the values of our other predictor variables. The Linear predictor values vary between around -1.64 and +1.64, these slightly odd values are what we get when we z-transform a straight line.
#
# We can see the regression parameter estimates in the fitted model, let's look specifically at the betas for 22Hz. Remember we have vectors of coefficients for every frequency.

print(glmsp.model.betas)

# Now the challenge, can you use the information above to write some code to plot the GLM model predicted spectrum for the start, middle and end of the data?

# Solution - we start with the intercept (glmsp.model.betas[0, :]) and add the slope effect for time (glmsp.model.betas[1, :]) multiplied by some predictor value.
# One tricky part here is that we have standardised our time predictor - as far as the GLM knows time starts at -1.64 and ends at +1.64 - we can't use the 'real' time values


start_spec = glmsp.model.betas[0, :] + glmsp.model.betas[1, :] * -1.64
middle_spec = glmsp.model.betas[0, :] + glmsp.model.betas[1, :] * 0
end_spec = glmsp.model.betas[0, :] + glmsp.model.betas[1, :] * +1.64

plt.figure()
plt.plot(glmsp.f,start_spec)
plt.plot(glmsp.f,middle_spec)
plt.plot(glmsp.f,end_spec)
plt.legend(['Start', 'Middle', 'End'])
plt.title('GLM predicted spectra')
