The stochasticTextureFiltering sample implements stochastic texture filtering in the context
of demand load textures.  This inlcudes:

# Stochastic filtering using finest mip level 
# Stochastic filtering using calculated mip level
# Stochastic EWA filter
# Higher anisotropy than hardware
# Gaussian vs. trilinear
# Stochastic filtering on top of bilinear/trilinear sampling.

Marcos Fajardo, Bartlomiej Wronski, Marco Salvi, Matt Pharr.
Stochastic Texture Filtering. 2023.
https://arxiv.org/abs/2305.05810

Christian Sigg and Markus Hadwiger.
Fast Third-Order Texture Filtering.  Graphics Gems 2, Chapter 20. 2005.

Christopher D. Kulla, Alejandro Conty, +1 author Larry Gritz
Sony Pictures Imageworks Arnold
Published in ACM Transactions on Graphics 1 August 2018