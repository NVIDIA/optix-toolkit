# stochasticTextureFiltering sample

The stochasticTextureFiltering program implements stochastic texture filtering in the context
of demand load textures.  [Stochastic filtering](/ShaderUtil/docs/stochasticFiltering/README.md) allows an application to probabilistically simulate higher order filters (bicubic, Gaussian, EWA, Mitchell, Lanczos...) by jittering samples taken from a low order interpolation method (point or linear). The program allow the user to combine different pixel filters, texture interpolation modes, and stochastic filtering jitter kernels.
