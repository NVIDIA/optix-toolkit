# Efficient Environment Map sampling

Ray tracers rely on environment map lighting to capture distant illumination. This note describes the efficient environment map sampling techniques implemented in the OTK ShaderUtil library, including cdf inversion tables, and the alias method.  These methods generate samples substantially faster than the traditional cdf binary search.  The files [PdfTable.h](/ShaderUtil/include/OptiXToolkit/ShaderUtil/PdfTable.h), [AliasTable.h](/ShaderUtil/include/OptiXToolkit/ShaderUtil/AliasTable.h) and [CdfInversionTable.h](/ShaderUtil/include/OptiXToolkit/ShaderUtil/CdfInversionTable.h) implement the algorithms described in this note, and the [cdfInversion](/examples/DemandLoading/CdfInversion) sample demonstrates their use.

## Environment Map Lighting

An environment map is a texture that represents a distant scene.  In a path tracer, the environment map is treated as a light source, and sample points are often chosen on it according to brightness.  Standard algorithms to do this build a **pdf table** from the environment map texture which can then be inverted to make either a **cdf table** or **alias table** that can transform uniform samples to the probability distribution defined in the pdf table.

## Building the PDF table  

Pdf table entries for an environment map must account for both brightness and solid angle (how much of the sphere is covered by a texel).   For texel $(i, j)$ in a map with resolution $(w, h)$, the unnormalized probability is given by the equation  $P_{ij}=A_{ij} B_{ij}$, where the $A$ and $B$ terms are solid angle and pixel brightness.

- The brightness term $B_{ij}$ is usually either luminance $(0.299r + 0.587g + 0.114b)$ or rgb sum $(r+g+b)$.  

- The angle term $A_{ij}$ is proportional to the solid angle that a texel covers.  Geometrically,  $A_{ij}$ is proportional to the texel’s area and the cosine of its deflection angle.  It is inversely proportional to the distance from the viewer. In a Lat/Long environment map, a rectangular texture wraps a sphere.  To calculate the angle term, suppose the viewer and sphere center are at the origin, and the sphere has radius 1.  All texels on the sphere directly face the origin so the deflection angle is 0, and texels are at a constant distance 1 from the origin.  Texels wrapped on the sphere have an area proportional to the sine of the angle away from the sphere's axis, $(0,0,1)$.  Based on this, the angle term for a Lat/Long map can be calculated as  $A_{ij}=sin(\pi (j+0.5) / h)$.  

- For a cube map or sky box, the environment map has six textures that cover the faces of a cube.  $A_{ij}$ for a texel on a cube face can be calculated by imagining that the face is a 2x2 square located 1 unit away from the origin on the z axis.  Texel $(i, j)$ is located at position $(x=(2i+1–w)/w, y=(2j+1–h)/h, z=1)$, and the squared distance from the origin to the texel center is $d^2=x^2+y^2+1$. The cosine of the deflection angle is $1/d$.  From this, the angle term for a cube map texel is $A_{ij}=1/d^3$.

The function `makePdfArray` in `CdfInversionTable.h` creates a pdf table from an environment map image. It has options to control how the A and B terms are calculated.

## CDF Tables and Binary Search

The standard method to sample a discrete pdf creates a cumulative distribution table (cdf) and then performs a binary search in it to find the final sample location.  For a 2D pdf, a marginal cdf is also made and two binary searches are needed.  Creating the cdf is a simple linear time preprocess on the cpu.  It involves calculating a prefix sum and dividing each element by the final value.  Sampling is fairly fast and tends to maintain the good sample spacing (stratification) of an input sequence, but the binary search has poor cache performance and can become a bottleneck on the GPU.  Morrical and Zellmann [3] found that the time needed to sample environment maps by a binary search may vary by as much as 10x depending on the environment map.

## Alias Tables

An **alias table** [2] is a data structure that can sample a discrete pdf in constant time by moving probability around to achieve a desired distribution.  Each table entry has an *alias* location and a *probability* of moving to the aliased location.

To sample with an alias table, an application looks up a table entry $i$ based on an input value $x$, and returns either $i$ or the alias location according to the probability in the table.  If the input value $x$ is a single float, the code must actually extract two values from it: the table index $i$, and a selection value $p$.  Since the float data type has only 23 mantissa bits, it will run out of precision for large tables.  `AliasTable.h` provides several methods to address this issue.  One method accepts an unsigned int as the input value rather than a float, providing 9 extra bits to work with.  Another, designed to sample 2D textures, takes a float2 as input, and returns a float2 texture coordinate.

Alias tables can be initialized from a pdf in linear time.  The OTK initialization code in `AliasTable.h` works as follows:  First the average value of the pdf table is found.  Then two indices walk through the pdf table looking for above and below average probability locations. At each iteration, the current below average position B is set to alias to the current above average position A.  The excess probability from B is subtracted from A.  If this causes A to go below the average probability, it is processed as the next below average value. Initialization finishes when both indices reach the end of the pdf table.

The main drawback of the alias method is that it disrupts the spacing of the input sequence, as shown in the figure below. Cdf inversion tables fix this issue while keeping constant time lookup.

## CDF Inversion Tables

A cdf inversion table [1] stores an inverted version of the cdf that is used as an oracle to speed up search.  It can be created in linear time with a single pass through the cdf.  For a 2D table, cdf rows are inverted separately along with the cdf marginal.

The OTK supports two ways of using these tables.  In the first method, the inversion table provides an initial guess for a linear search in the cdf (two searches for a 2D table).  The linear search comes up with the same points as binary search, but has better cache performance and runs in constant time on average.

The second method performs a direct lookup and linear interpolation in the inversion table to get a sample location. The OTK stores inversion tables as unsigned shorts, so memory usage for direct lookup is just 2 bytes per table entry, one third the cost of the linear search method.

The cdf search methods (either by binary search or using inversion tables) preserve sample spacing better than the alias method.  This translates to faster convergence in a Monte Carlo render.  Cdf inversion tables also have fewer numerical precision issues than the alias method.

<img src="inputSequence.png" width="24%"/><img src="binarySearch.png" width="24%"/><img src="directLookup.png" width="24%"/><img src="aliasMethod.png" width="24%"/><br/>
Input sequence (left) transformed by binary or linear cdf search (left center), direct lookup in the cdf inversion table (right center), and the alias method (right).  

## Table Initialization Time

Initialization for all of the methods discussed in this note is fast and scales linearly with table size. For an 8k x 4k map, initialization completes in about a third of a second on our test machine, which is 4 to 5 times faster than loading the same size exr image. The time could probably be reduced by multithreading or moving the inversion code to the GPU, but it is not currently a bottleneck.

## Memory Usage and Compression

Environment maps can be large, 4k and 8k are common. An environment map texture can be paged to reduce the memory burden, but alias tables and inversion tables cannot easily be paged at runtime.  Thus table memory use is a concern for all of the methods discussed here.  For an 8k x 4k Lat/Long environment map, table size ranges from 64 MB for direct lookup to 256 MB for the alias method.

Downsampling the tables can improve memory usage without sacrificing rendering quality.  This is because most environment map samples are not taken at the finest mip level anyway when texture LOD is controlled by ray cones or differentials.  A dull reflection yields a wide ray cone, calling for a coarse mip level.  A specular reflection may call for the finest mip level, but BRDF sampling will dominate in this case. Thus downsampling the inversion table has little effect on rendered images.  As an example of the memory savings available by downsampling, a 1k direct lookup inversion table weighs in at a tiny 1 MB of storage, with almost no change in image quality, as shown below. 

<img src="OneKInversionTable.png" width="49%"/> <img src="EightKInversionTable.png" width="49%"/>
Renderings with 1k (left) and 8k (right) inversion tables. Both use an 8k environment map.  These renderings are practically identical.  The only noticeable difference is some slight blurring of the shadow boundaries in the left image.

## Sampling Probability

To use an environment map as a light source, a Monte Carlo ray tracer must do two things: (1) sample the environment map according to some probability distribution, and (2) determine the probability $P$ that an arbitrary direction would be chosen by the sampling routine.  There are a number of ways to store or calculate the probability. Some of these are: 

- Store a separate pdf table.  This is the most straightforward way to evaluate the pdf. The drawback is that it involves extra storage and an extra memory read.  Another issue is that $P$ calculated in this way may not cancel exactly with an environment map sample in the Monte Carlo integration, leading to render noise.

- Compute the probability from the cdf table. If the sampling algorithm stores a cdf table, no extra storage is needed for this method.  P can be calculated as part of the sampling process for environment map samples.  Calculating $P$ for an arbitrary direction requires 4 reads. 

- Calculate the probability from the environment map.  This method is used by the OTK CdfInversion sample. One benefit is that no extra table is needed.  Storage is minimal, a single value for the weighted average brightness of the texture.  In this scheme, the probability is calculated as $P = B / (4 \pi W)$, where $B$ is the brightness of the environment map color evaluated in the sample direction, and $W$ is the weighted average brightness of environment map texels: $W = \sum{A_{ij} B_{ij}} / \sum{A_{ij}}$.  The $B$ term can be calculated in several ways.  First, the environment map can be evaluated by point sampling at the same mip level as the inversion table.  This is exact, but leads to image noise when the renderer uses ray cones or differentials to control the environment map sampling.  If some bias can be tolerated, it is better to simply reuse the environment map sample taken by the renderer.  That way, $P$ will cancel exactly with the environment map color in the Monte Carlo sum, reducing image noise substantially, as seen below. <br><br> 

<img src="accurateProbabilities.png" width="49%"/> <img src="emapProbabilities.png" width="49%"/>
Renderings with 32 spp.  (Left) ray cones determine the texture footprint for sampling the environment map, but accurate sampling probabilities are used in the Monte Carlo sum, leading to render noise.  (Right) Directly using the environment map sample from the ray cones for both the color and probability reduces noise substantially. 

In practice, an application should be consistent about how it samples the environment map and determines probability.  An unbiased render should sample the environment map at the same mip level as the inversion table, and an biased render should use the same environment map sample to compute both color and probability.

## Sampling Speed

The figures bleow plot render speed for two 8k environment maps obtained from HDRI Haven [4] with different sized inversion tables.  The Meadow map has a direct view of the sun, and most of the lighting power is concentrated in a small spot.  The Snowy Forest map shows more diffused cloudy day lighting. As seen in the plots, render time is dependent on the environment map contents as well as the inversion table size.  Concentrated lighting improves the efficiency of all the sampling methods, likely because of cache coherence.  Binary search fares well with concentrated lighting maps or small inversion tables, but struggles with larger sizes and diffused light environment maps.  Linear search and direct lookup performed well on all the environment maps we tested, with direct lookup slightly outperforming linear search most of the time.  The alias method is competitive in terms of speed for diffused lighting, but is slower in concentrated lighting.  We believe this is a caching effect as well.  In concentrated lighting, the inversion methods sample the marginal and the same few rows, but the alias method ranges evenly over the entire table, making it inefficient.

<img src="meadow2.png" width="49%"/> <img src="snowyForest.png" width="49%"/>
Render speed for different 8k environment maps, with different sampling strategies and inversion table sizes.

## References

[1] David Cline, Peter Wonka, Anshuman Rasdan. A Comparison of Tabular Pdf Inversion Methods. Poster, Computer Graphics Forum. 2009.  

[2] Alastair Walker. An efficient method for generating discrete random variables with general distributions. ACM Transactions on Mathematical Software, September 1977. vol. 3(3), pp. 253–256.

[3] Nate Morrical, Stefan Zellmann.  Inverse Transform Sampling Using Ray Tracing Hardware. In Ray Tracing Gems II. Chapter 39. pp. 625–641.

[4] Zaal, G. HDRI Haven. https://hdrihaven.com/, 2016.

