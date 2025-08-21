# Stochastic Texture Filtering in the OTK

Stochastic texture filtering [1] jitters texture coordinates to achieve high quality filtering when combining multiple samples. The OTK implementation of stochastic texture filtering supports improved upsampling, extended anisotropy, stochastic EWA filtering, and contrast enhancement. The ShaderUtil file [stochastic_filtering.h](/ShaderUtil/include/OptiXToolkit/ShaderUtil/stochastic_filtering.h) implements the methods described here, and they are used in the [stochastic texture filtering](/examples/DemandLoading/StochasticTextureFiltering/) sample.

## Stochastic Texture Filtering

Hardware graphics API’s typically support three per-sample texture filter modes: point (a box filter), linear (a tent filter), and anisotropic (an optimized EWA filter [2]).  Stochastic filtering can expand the filtering repertoire of a texturing system. To perform stochastic texture filtering, an application starts with a per-sample **base filter** mode and then jitters (perturbs) texture coordinates according to a second distribution, the **jitter kernel**.  By averaging multiple jittered samples, the result is the same as convolving the base filter with the jitter kernel, up to some noise. Jitter kernels implemented in the OTK include *box*, *tent*, and *Gaussian*, as well as approximate kernels for *Mitchell* and *Lanczos* filters.

## Improved Texture Magnification

Bilinear upsampling artifacts are a common complaint in texture mapping.  Some renderers use a bicubic filter to magnify textures, achieving a smoother result.  Straightforward sampling of the bicubic kernel requires 16 texture reads, or “taps”. [3] cleverly reduced this to 4 bilinear taps.  The OTK implementation of cubic filtering uses this idea, and is described [here](/ShaderUtil/docs/cubicFiltering).  Stochastic texture filtering performs bicubic upsampling with a single tap per sample, although multiple samples must be averaged to smooth the result. This can be important when texture lookups are expensive, such as when using neural textures.

The original stochastic texture filtering paper implements the bicubic filter by picking a specific texel to sample based on the bicubic kernel, but we have found that a number of base filter/jitter kernel combinations visually match the bicubic very closely.  The trick is to choose a bell curve-like distribution and match the variance of the cubic (1/3).  In a stochastic texture filter, the variance of the base filter and jitter kernel add, so to match the cubic, an application decides on a base filter and jitter kernel, then scales the jitter amount to match the cubic variance.  The variances of different base filters and jitter kernels are as follows: point filter and box kernel (1/12), Linear filter and tent kernel (1/6), Gaussian kernel (1).  Based on this, some close approximations to the bicubic are listed below (and in fact, the linear/tent combination is exact).

<div align="center">
  
| Texture Filter | Jitter Kernel | Jitter Scale to match Bicubic |
| ----------- | ----------- | ---------- |
| Linear      | Tent        | 1.0        |
| Linear      | Gaussian    | 0.408      |
| Point       | Gaussian    | 0.5        |

</div>

Figure 1 shows how stochastic texture filtering can match bicubic interpolation and improve quality over bilinear interpolation. Bilinear upsampling has blocky diamond artifacts.  Switching to a bicubic filter smooths out the artifacts, at the cost of 4 taps per sample.  Stochastic filtering (linear base filter, tent jitter kernel) perturbs texture coordinates so that when samples are averaged the result is nearly identical to bicubic.

<img src="fig1-bilinear.png" width="24%"/> <img src="fig1-bicubic.png" width="24%"/> <img src="fig1-stochasticBicubic1.png" width="24%"/> <img src="fig1-stochasticBicubicConverged.png" width="24%"/>

Figure 1:  Stochastic texture filtering can match higher quality filters such as bicubic. (Left) Bilinear, (left middle) Bicubic,   (right middle) Stochastic 1 spp, (right) Stochastic converged.

The `tex2D` functions provided in the OTK include variants with an extra parameter for jittering the texture coordinate proportional to the texel size at mip level 0.  Assuming `xi` is a `float2` drawn from a uniform distribution in $[0,1)^2$, OptiX code for the stochastic filter shown in figure 1 could be written:

```
float2 texelJitter = 1.0f * tentFilter( xi );
float4 t = tex2DGrad<float4>( context, texId, x, y, ddx, ddy, resident, texelJitter );
```

Equivalently, the texelJitter can be rolled into the texture coordinates, useful for applications not using OptiX demand load textures:

```
float2 texelJitter = 1.0f * tentFilter( xi );
float xjitter = texelJitter.x / textureWidth;
float yjitter = texelJitter.y / textureHeight;
float4 t = tex2DGrad<float4>( texture, x + xjitter, y + yjitter, ddx, ddy, resident);
```

This filter performs stochastic bicubic upsampling, which seamlessly transitions to a hardware EWA filter at higher mip levels.  

## Blurring kernels

Since the jitter kernel can be scaled, stochastic filtering can fine-tune the amount of blur without increasing the number of taps.  Figure 2 shows the effect of changing the jitter kernel width, here using a Gaussian.

<img src="fig2-gaussian0.5.png" width="24%"/> <img src="fig2-gaussian1.0.png" width="24%"/> <img src="fig2-gaussian4.0.png" width="24%"/> <img src="fig2-gaussian8.0.png" width="24%"/>

Figure 2: Stochastic texture filtering with different jitter kernel widths.

## EWA Filtering

The EWA (elliptically weighted average) filter sets the standard for anisotropic texture filtering. On NVidia GPUs, the tex2DGrad intrinsic computes an approximate EWA filter sample based on two texture gradients (2D vectors describing an elliptical footprint in texture space).  This is fast and high quality, but anisotropy is limited to 16 on current hardware. If the longer of the texture gradients is more than 16 times longer than the short one, the short one is lengthened.  This is the right choice to prevent aliasing of individual samples, but blurs the texture. 

**Extending Anisotropy.**  Shortening the longer gradient before it is sent to tex2DGrad would prevent over blurring in multisample rendering.  This may be sufficient in many multi-sample renders, but individual samples would tend to alias.  A more principled approach shortens the longer gradient, and jitters the texture coordinate to account for the missing anisotropy.  The file `stochastic_filtering.h` provides the function extendAnisotropy for this purpose, which can be used as follows:

```
float2 xy = float2{x,y} + extendAnisotropy( ddx, ddy, xi );  
float2 texelJitter = GAUSSIAN_STANDARD_WIDTH * boxMuller( xi );
float4 t = tex2DGrad<float4>( context, texId, xy.x, xy.y, ddx, ddy, resident, texelJitter);
```

The code above combines two stochastic texture filters.  First, extendAnisotropy jitters the texture coordinate directly to extend the anisotropy, shortening the longer of ddx and ddy if needed. Second, texelJitter is passed to the `tex2DGrad` function which jitters according to texels at mip level 0 to perform approximate bicubic upsampling.  The constant `GAUSSIAN_STANDARD_WIDTH` is 0.408, which comes from the table above. In this example, the same random point `xi` is used by both jitter kernels.  Different random points could be used, but we have not noticed any issues from this reuse, likely because the two effects become important in different parts of the texture.

**EWA Filter at mip level 0.** For applications such as bump mapping, combining samples by a straight average does not work well because the texture values do not map linearly to colors.  In these cases it is preferable to average samples taken from mip level 0, or a higher level than would be chosen by the hardware.  An application can implement a stochastic EWA jitter kernel by scaling a Gaussian jitter by the texture gradients, as in the following:

```
float2 gauss = GAUSSIAN_STANDARD_WIDTH * boxMuller( xi );
float2 xy = float2{x, y} + ddx * gauss.x + ddy * gauss.y;
float4 t = tex2DGradUdim<float4>( context, texId, xy.x, xy.y, float2{0.0f}, float2{0.0f}, resident, gauss );
```
<br>

Figure 3 compares standard hardware rendering using `tex2DGrad` with stochastic filters to extend anisotropy for grazing angles.  As can be seen, the extended anisotropy filter and stochastic EWA filtering at mip level 0 fix the hardware blurring and achieve almost identical results when used on a diffuse map.

<img src="fig3-limitedAnisotropy.png" width="31%"/> <img src="fig3-ewa0Anisotropy.png" width="31%"/> <img src="fig3-extendedAnisotropy.png" width="31%"/>

Figure 3. (Left) tex2DGrad has limited anisotropy which leaves reconstruction errors at grazing angles.  In this example, the checkerboard blurs out at the top right of the image. (Middle and right) Extending anisotropy either using our extended anisotropy filter, or a stochastic EWA filter at mip level 0, removes the errors.

**Negative LOD biasing.** We note that bump mapping will probably not be fixed by the extended anisotropy filter, as pointed out by [1].  At the same time, sampling at mip level 0 may be undesirable due to memory concerns.  Negative LOD biasing is an alternative to always using mip level 0.  It represents a compromise to avoid loading the most detailed texture levels when not needed.  In this method, an application scales the texture gradients to access higher mip levels, but not necessarily level 0.  Stochastic filtering can then make up for the reduced footprint by jittering the texture coordinate.

## Contrast Enhancing Filters

Some filters increase contrast by including negative lobes in the filter kernel. [1] supports several of these by sampling the positive and negative filter lobes separately at exact pixel coordinates, subtracting them to get a final result (called positivization).  This is more challenging in the context of OptiX demand load textures, since it is not simple to determine the resolution of a texture when it is sampled, particularly for UDIM textures that can have different resolutions in different subtexture regions. Here we describe several different ways to implement stochastic contrast enhancement filters.

**Unsharp Mask Filter.**  Since the Gaussian jitter kernel is supported in the OTK, it is easy to create an unsharp mask filter, which is just a difference of Gaussians. For texture coordinate $(x, y)$, the unsharp mask image sample can be expressed:

$$t(x, y) = (1+w) G_p(x, y) – w G_n(x, y)$$

where $G_p$ and $G_n$ are the Gaussian kernels, and $w$ is the filter strength.  In code, the unsharp mask filter combines two taps, and can be written:

```
float2 gauss = filterwidthscale * boxMuller( xi );
float4 gp = tex2DGrad<float4>( context, texId, x, y, ddx, ddy, resident, gpwidth * gauss );
float4 gn = tex2DGrad<float4>( context, texId, x, y, ddx, ddy, resident, gnwidth * gauss );
float4 t = ( 1.0f + filterweight ) * gp – filterweight * gn;
```

The OTK stochasticTextureFiltering sample sets `gpwidth` to 0.5 and `gnwidth` to 0.7. These can be varied depending on the application, and it may be possible to determine values that approximate specific cylindrical filters.  The parameters `filterwidthscale` and `filterweight` tune the width and strength of the filter.  Figure 5 highlights the contrast enhancement that can be achieved by the unsharp mask filter. Note that sharpening filters can return negative values, so applications must be able to handle this. 

<img src="fig4-bilinear.png" width="31%"/> <img src="fig4-bicubic.png" width="31%"/> <img src="fig4-unsharpMask.png" width="31%"/>

Figure 4. (Left) Hardware bilinear sampling. (Middle) Stochastic filter with Gaussian jitter kernel removes jaggies but leaves blurred edges. (Right) The stochastic unsharp mask filter reduces the bilinear artifacts and sharpens the edges.

**Separable Lanczos and Mitchell Filters.**  Besides the unsharp mask filter, the OTK includes jitter kernels to approximate Lanczos and Mitchell filters for point and linear base filters.  These are implemented by numerically integrating and inverting the positive and negative lobes of the jitter kernels, approximating the resulting functions with cubic curves.  The inverse kernel functions have vertical derivatives, so the optimization transforms the input coordinate using a square root wherever there is a vertical derivative to better match a cubic to the function shape.  Under this scheme the parameters for a filter are stored as 10 floating point numbers: 8 for the cubic fits, 1 for the relative contributions of the two positive lobes, and one to mix between the positive and negative taps.  The appendix describes the fitting process in more detail.  Figure 6 compares the stochastic Lanczos filter to an exact Lanczos implementation that takes 16 taps per sample.  The stochastic Mitchell filter is even more accurate.

<img src="fig5-lanczos.png" width="31%"/> <img src="fig5-lanczosLinearBase.png" width="31%"/> <img src="fig5-lanczosPointBase.png" width="31%"/> 

Figure 5. (Left) Image processed with Lanczos filter.  (Middle) Stochastic Lanczos approximation with point base filter and difference with exact Lanczos.  (Right)  Stochastic Lanczos with linear base filter and difference with exact Lanczos.

To sample a texture with one of the separable sharpening filters, an application calls the functions sampleSharpenPos and sampleSharpenNeg to choose positive and negative tap locations, and then does a weighted subtraction of texture taps, as follows:

```
float2 pjitter = sampleSharpenPos( LANCZOS_TENT, xi );
float2 njitter = sampleSharpenNeg( LANCZOS_TENT, xi );
float4 gp = tex2DGradUdim<float4>( context, texId, x, y, ddx, ddy, resident, pjItter );
float4 gn = tex2DGradUdim<float4>( context, texId, x, y, ddx, ddy, resident, njitter );
float4 t = ( 1.0f + LANCZOS_TENT_NWEIGHT ) * gp – LANCZOS_TENT_NWEIGHT * gn;
```

**Cylindrical Lanczos Filter.**  The OTK also implements the cylindrical Lanczos filter, which inverts the *jinc* function instead of the *sinc*, and is radially symmetric.  Sampling this filter is similar to the separable sharpening filters, and the stochasticTextureFiltering sample shows its use in an OptiX program. 

## Discussion and Limitations

A key goal of this project has been to limit changes to the texturing API and avoid a proliferation of `tex2D` function variants while allowing all of the demand loaded `tex2D` variants to support stochastic filtering. Our design solution was to have the application insert some preamble code to calculate the jitter before the texture call.  Applications may wish to encapsulate this. 

The jittering done at mip level 0 related to upsampling could be phased out at higher mip levels, but the effect to diminish naturally as samples climb the mip pyramid anyway.

The Gaussian has infinite support, but an application can achieve finite support by scaling the x coordinate sent to the Box Muller transform. For example, scale by 0.997 to limit support to 3 standard deviations.

All of the stochastic filters require multiple samples or a denoiser to converge to a smooth result.  Stochastic filters with a linear base filter (or hardware anisotropic) will tend to converge a little faster than the point versions.

The OTK implementation has avoided sampling exact pixel coordinates because it is expensive to determine these at sample time, and it prevents building on the hardware filtering.  Instead, approximations were used that can easily be tuned in filter width and strength. 

## References

[1] . Marcos Fajardo, Bartlomiej Wronski, Marco Salvi, Matt Pharr. Stochastic Texture Filtering. 2023.

[2] McCormack, Joel & Perry, Ronald & Farkas, Keith & Jouppi, Norman. 1999. Feline: Fast Elliptical Lines for Anisotropic Texture Mapping. Proceedings of Siggraph 1999. pages 243-250. 

[3] Christian Sigg and Markus Hadwiger. 2005. Fast Third-Order Texture Filtering. GPU Gems 2, Chapter 20.

## Appendix: Numerically Inverting Lanczos and Mitchell Filters

**Finding the Jitter Kernel.** When designing a stochastic filter, one must keep in mind that the jitter kernel convolves with the base filter.  The jitter kernel needed for a Lanczos filter, for example, depends on the base filter.  If $F$ is the filter function being modeled, the desired jitter kernel $F’$ must be $F$ **de-convolved** with the base filter.  

To find $F’$, we guessed that an appropriate function could be found of the form  $(1+b|a^3x^3|)F(ax)$. We then used a hill climbing procedure to solve for a and b.  These approximations are surprisingly good for Lanczos and Mitchell, as seen in the plot below, which shows the least accurate of them.  The blue line in the plot is Lanczos, yellow is the jitter kernel, and the green line almost on top of Lanczos is the approximation, $F’ * Tent$.  

<p align="center">
<img src="ap-lanczosLinearFit.png" width="70%"/>
</p>

**Inverting the Jitter Kernel.**  Once a jitter kernel $F’$ has been found, the next step is to integrate and invert its positive and negative lobes numerically.  Only the positive kernel domains need to be modeled since the filters are symmetric.  The negative domain parts are handled by reflection.  This reduces the nine regions of the full filter to four, two positive and two negative (the top right corner in the figure below):

<p align="center">
<img src="ap-lobes.png" width="25%"/>
</p>

**Fitting the Filter Lobes with Cubics.**  The inverse kernel lobes for Lanczos and Mitchell filters have vertical derivatives that make fitting them with cubics a challenge.  To better match the function shapes to a cubic, we transform the input coordinates by a square root wherever there is a vertical derivative.  The left side of the figure below plots the inverted functions of $F’$ for a Lanczos filter, assuming a linear base filter.  The right side of the figure plots the stretched versions of these functions and cubic fits. Once stretched, the inverted functions are close to linear, and the cubic approximations are quite good.  

<img src="ap-invertedLanczos.png" width="48%"/> <img src="ap-invertedLanczosStretched.png" width="48%"/>

**Combining Positive and Negative Taps.**  To evaluate the sharpening filters, an application takes taps from the positive and negative kernel lobes and combines them in a weighted subtraction.  This involves two constants: $k_p$, the probability of sampling the smaller positive lobe vs. the larger one, and $k_n$, the weight of the negative tap.  The final sample combines the positive and negative taps, $t_p$ and $t_n$, in the equation $t = (1+k_n) t_p - k_n t_n$. The value for $k_p$ is calculated as $a_n^2/(1+a_n)^2$, where an is the ratio of the negative lobe area to the positive lobe area of $F$ (not $F’$).  $k_n$ is currently determined by hill climbing, starting at $2a_n / (1+a_n^2)$.  The hill climbing yields a good filter fit, but in the future we would like to find a closed form expression for $k_n$.
