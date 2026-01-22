# Cubic Texture Filtering in the OTK

Graphics hardware natively supports point and bilinear texture filtering. These are fast but inadequate for some applications. Production renderers often want cubic filtering when magnifying textures to smooth out blocky artifacts. This note describes the cubic texture filtering implementation included in the file [CubicFiltering.h](/ShaderUtil/include/OptiXToolkit/ShaderUtil/CubicFiltering.h) in ShaderUtil. The OTK code follows the implementation in [Open ImageIO](https://github.com/AcademySoftwareFoundation/OpenImageIO), and is intended to provide GPU functionality similar to the OIIO `texture` function.

## Linear Interpolation

GPU textures are *sampled functions*. Sample values are stored at specific locations (pixel centers), and function values at arbitrary positions are reconstructed from nearby samples by interpolation. Texture hardware supports *point* and *linear* interpolation in a single function call or "tap". In *point* interpolation, the nearest sample is used.  In *linear* interpolation, the two nearest samples are blended to form a linear segment, as shown below. 

<p align="center">
<img src="linearInterpolation.png" width="60%"/><br>
</p>

$$
f(x) = (1-m)F_{i-1} + m F(i)
$$

## Cubic Interpolation

*Cubic* interpolation is smoother than linear. It can be computed as a weighted sum of the four nearest function samples, basing the weights on the B-spline basis function $B$:

$$
B(x) = 
\begin{matrix}
-\frac{1}{2}|x|^3 - |x|^2 + \frac{2}{3}     & |x| \in [0,1) ~~ \\
\frac{1}{6} (2-|x|)^3                       & |x| \in [1,2) ~~ \\
0                                           & |x| \in [2,\infty)
\end{matrix}
$$

Let $i = \lfloor x \rfloor$, and $m=x-i$ (the fractional part of $x$). Then $f(x)$ can be calculated using cubic interpolation as shown below:

<p align="center">
<img src="cubicInterpolation.png" width="60%"/><br>
</p>

$$
f(x) = w_0 F_{i-1} + w_1 F_{i} + w_2 F_{i+1} + w_3 F_{i+2}
$$

The weights $w_0 ... w_3$ are the value of $B()$ evaluated at the sample positions:

$$
\begin{matrix}
w_0 = B(-1-m) = (-m^3 + 3m^2 - 3m + 1)/6 \\
w_1 = B(-m) = (3m^3 - 6m^2 + 4)/6 ~~~~~~~~~~~~~~~~~ \\
w_2 = B(1-m) = (-3m^3 + 3m^2 + 3m + 1)/6 ~ \\
w_3 = B(2-m) =(m^3)/6 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
\end{matrix}
$$

## Evaluating the Cubic in Fewer Taps

Straightforward cubic evaluation of $f(x)$ requires 4 table lookups (point taps). Sigg and Hadwiger [2] showed that it can be reduced to 2 interpolated lookups (linear taps). This technique works by positioning a linear tap to capture the relative contributions of $F_{i-1}$ and $F_{i}$, and another to capture the contributions of $F_{i+1}$ and $F_{i+2}$. The two taps then combine in a weighted sum, replacing the equation above with

$$
\begin{equation}
f(x) = (w_0 + w_1) T(i-1+d_{01}) + (w_2 + w_3) T(i+1+d_{23})
\end{equation}
$$

where $T()$ is a linear tap which interpolates between the function samples, and $d_{01} = w_1 / (w_0 + w_1)$ and $d_{23} = w_3 / (w_2 + w_3)$ are the fractional offsets needed for proper blending. 

The linear tap technique is separable, so 4 bilinear taps (`tex2D` calls) suffice to calculate 2D bicubic interpolation, rather than 16 table lookups. Mathematically, this can be expressed as follows:

$$
\begin{equation}
f(x,y) =
\begin{matrix}
(w_{x0}+w_{x1}) ~ (w_{y0}+w_{y1}) ~ T(i-1+d_{x01}, j-1+d_{y01}) ~ + \\
(w_{x2}+w_{x3}) ~ (w_{y0}+w_{y1}) ~ T(i+1+d_{x23}, j-1+d_{y01}) ~ + \\
(w_{x0}+w_{x1}) ~ (w_{y2}+w_{y3}) ~ T(i-1+d_{x01}, j+1+d_{y23}) ~ + \\
(w_{x2}+w_{x3}) ~ (w_{y2}+w_{y3}) ~ T(i+1+d_{x23}, j+1+d_{y23}) ~~~~ \\
\end{matrix}
\end{equation}
$$

where $w_{x..}$, $w_{y..}$, and $d_{x..}$, $d_{y..}$ are the weights and fractional offsets for $x$ and $y$, and $j = \lfloor y \rfloor$.

**Scaling Texture Coordinates.**
The derivation above assumes that function samples are located at integer positions. Texture coordinates must be scaled to work with these equations.  If $s$ and $t$ are the texture coordinates of a point, and $w$ and $h$ are the width and height of the texture, the $x$ and $y$ values used in equation 2 will be $x = w s - 0.5$, and $y = h t - 0.5$.

**Cubic Filtering at Different Mip Levels.**
The above equation can only evaluate cubic filtering at integer mip levels using `tex2DLod` calls to compute the $T$ values.  However, the linear taps needed for adjacent mip levels do not line up, so the code must filter each mip level separately and blend between them to achieve cubic filtering for fractional mip levels.

## The Cubic Derivative

The cubic derivative of $f$ in 1D can be calculated using the same linear tap method described earlier, replacing $B$ with its derivative $B'$ when calculating the weights. For $x \ge 0$, the weights can be expressed:

$$
B'(x) = 
\begin{matrix}
-\frac{3}{2}x^2 - 2x  &   x \in [0,1) ~~ \\
-\frac{1}{2} (2-x)^2  &   x \in [1,2) ~~ \\
0                     &   x \in [2,\infty)
\end{matrix}
$$

$B$ is symmetric about the origin, so $B'(x) = -B'(-x)$, and the derivative weights reduce to:

$$
\begin{matrix}
w'_0 =  -0.5m^2 + m - 0.5 \\
w'_1 =  1.5m^2 - 2m ~~~~~~~~~~~ \\
w'_2 =  -1.5m^2 + m + 0.5 \\
w'_3 =  0.5m^2 ~~~~~~~~~~~~~~~~~~~~
\end{matrix}
$$

This calculation works because $w'_0$ and $w'_1$ are always negative, and $w'_2$ and $w'_3$ are always positive. The signs for the derivative weights of each linear tap agree and the sum from equation 1 is still valid.

In 2D there are two partial derivatives. The $x$ partial is calculated by substituting $w'_x$ for the $x$ weights in equation 2, keeping $w_y$ for the $y$ weights, and analagously for the $y$ partial. 

**Rotating the Texture Derivatives.**
The partial derivatives constitute a texture gradient that can be rotated to find a directional derivative. If the derivatives returned from the texture call are $drds$ and $drdt$, then the directional derivative at angle $a$ is $cos(a)~drds+sin(a)~drdt$. 

## OTK Implementation

The OTK cubic filtering implementation is included in the file `CubicFiltering.h`. It supports the same interpolation modes as OIIO: *closest*, *bilinear*, *bicubic*, and *smart bicubic*.  The default mode, smart bicubic, uses bicubic interpolation when magnifying, bilinear otherwise. This avoids the cost of bicubic interpolation when when it is not needed.

## Smart Bicubic Implementation

Smart bicubic is the most popular sampling mode in many high quality renderers.  It switches from linear to cubic interpolation when the texture is magnified. This creates a discontinuity as the sampler goes from minifying the to magnifying. The OTK implementation fixes this by blending between bilinear and bicubic sampling over one mip level.

For texture colors, 1 anisotropic linear tap (`tex2DGrad` call) is sufficient when the texture is minified. When the texture is magnified, 4 isotropic taps (`tex2D` calls) are used, and the cubic filter is isotropic. In the transition zone, a blend between bilinear and bicubic is achieved using 5 taps. 

(It is possible to blend between bilinear and bicubic with just 4 taps by modifying the weights in equation 2.  Let $W_c=(w_0,w_1,w_2,w_3)$ be the cubic weights for a sample.  The linear weights are $W_l=(0,1-m,m,0)$. If $k$ is the blend between linear and cubic, the final weights are $W=kW_c+(1-k)W_l$. However, since the linear tap is anisotropic the weight modification scheme fails.) 

For texture derivatives in smart bicubic mode, 4 taps are needed to compute both derivatives when the texture is minified, 12 taps in the transition zone, and 8 taps when the texture is magnified. 8 or 12 taps may seem a heavy burden, but the performance is actually fairly good. The taps are cache coherent, and when demand load textures are used the implementation only requests texture tiles once for all the taps.

We also note that the partials for a scalar texture could be precomputed and stored in a 2 channel texture. The partials could then be evaluated simultaneously via cubic interpolation with 4 taps.

## Bilinear Derivative Precision

The texturing functions return bilinear texture derivatives in some filtering modes. These are computed using finite differences. Hardware texturing units convert texture coordinates to a fixed point format with 8 bits of fractional precision. To avoid precision issues, the code computes a difference across half a pixel at the requested mip level, and these derivatives closely match those given by OIIO.

## Cubic mode

The *cubic* filtering mode in the OTK code is not anisotropic. Instead, it blends between 4 tap cubic samples on two mip levels.

## Tests

**Sampling speed.**
The table gives timings we measured for the *smart bicubic* filter mode, using *demand load textures* calling the `tex2DCubic` function in the file `Texture2DCubic.h`. Results are encouraging. In our experiments, bicubic filtering with 4 taps rendered at about 93 percent of bilinear, and even the bilinear/cubic derivative transition, with 12 taps, rendered at 84 percent of bilinear, and has lower overhead than using the demand load texture system.

<div align="center">
  
| Filter      | Taps        | FPS        |
| ----------- | ----------- | ---------- |
| Bilinear    | 1           | 604        |
| Bicubic     | 4           | 564        |
| Bilinear Derivatives | 4  | 575        |
| Bicubic Derivatives | 8   | 530        |
| Transition Derivatives | 12 | 510      |

</div>

**Filtering results.**
The images below show bilinear and bicubic filtering using the `tex2DCubic` function. Bilinear interpolation results in blocky artifacts.  These are reduced using bicubic interpolation.

<p align="center">
<img src="CSbilinear.png" width="30%"/> <img src="CSbicubic.png" width="30%"/><br>
(Left) Bilinear filtering, 1 tap. (Right) Bicubic filtering, 4 taps.
</p>

**Texture Derivatives.**
When magnified, bilinear texture derivatives show blocky artifacts which are even more distracting when the texture is a bump map. Bicubic filtering smooths out the artifacts.

<p align="center">
<img src="CS1bilinearDX.png" width="30%"/> <img src="CS1bicubicDX.png" width="30%"/><br>
(Left) Bilinear X derivative, 2 taps. (Right) Bicubic X derivative, 4 taps.
</p>

## Comparing with Open ImageIO

The OTK cubic filtering routine attempts to match Open ImageIO filtering with just a few taps, and for the most part they come quite close. The images below compare the point spread function of the OIIO texture function to the OTK routine by including a single white pixel in a black image. The top of each image shows the value, and the bottom shows the derivatives. the output of Open ImageIO is shown on the left, the OTK cubic filtering routine is shown in the middle, and their difference is shown on the right.

<img src="smartBicubic0.png" width="49%"/> $~$ <img src="smartBicubic3.png" width="49%"/><br>
(Left) Smart bicubic at mip level 0. (Right) Smart bicubic at mip level 3.

The main cases in which the OTK output deviates from OIIO is in the transition between bicubic and bilinear filtering. As shown below, OIIO performs anisotropic cubic filtering in the transition zone, but the OTK routine uses bilinear. Keep in mind, however, that the texture is magnified here to show the point spread function. In a typical rendering, the streaks would be compressed into a single pixel.

<img src="bicubicAnisotropy.png" width="49%"/> $~$ <img src="bicubicTransition.png" width="49%"/><br>
Smart bicubic in the transition between bicubic and bilinear.

The OTK routine is also limited to 16x anisotropy, whereas OIIO can do much higher, as shown below. This could be extended by combining multiple taps.

<img src="largeAnisotropy.png" width="49%"/> <br>
Large anisotropy.

In the future if there is sufficient interest, we may add a higher quality filtering mode that extends anisotropy and has smoother transitions between bilinear and bicubic, and provides anisotropic cubic sampling. 

## References

[1] OpenImageIO. https://github.com/AcademySoftwareFoundation/OpenImageIO

[2] Christian Sigg and Markus Hadwiger. GPU Gems 2 Chapter 20. Fast Third-Order Texture Filtering. 2005.
