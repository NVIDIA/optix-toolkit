# NeuralTextureViewer sample

The **neuralTextureViewer** illustrates how to use the demand loading library to stream neural textures. Neural textures are attractive since they can achieve very high compression ratios on the GPU.

## Quick Start Guide

### Creating a demand-loaded neural texture in the OTK

An application can include Neural textures (NTC files) in the OTK demand texturing system using the `NeuralTextureSource` class. To do this, the app must create a `NeuralTextureSource` passing in the `textureName`,  set the `CU_TRSF_READ_AS_INTEGER` flag in the texture descriptor, and call `createTexture` in the demandLoader:

```
NeuralTextureSource* neuralTex = new NeuralTextureSource( textureName );

TextureDescriptor texDesc = makeTextureDescriptor( CU_TR_ADDRESS_MODE_CLAMP, filterMode );
texDesc.flags |= CU_TRSF_READ_AS_INTEGER;

const DemandTexture& texture = demandLoader->createTexture( neuralTextureSource, texDesc );
unsigned int textureId = texture.getId();
```

The `filterMode` can be any of FILTER_POINT, FILTER_BILINEAR, or FILTER_CUBIC, or FILTER_SMARTBICUBIC. It will be used to configure stochastic filtering in the `ntcTex2D*` sampling functions.

### Sampling a demand-loaded neural texture

In an OptiX shader, a neural texture can be sampled with one of the `ntcTex2D*` functions defined in `Texture2DNeural.h`.  These functions return an array that contains all channel values from the sampled texture set. Variants of sampling function also return a pointer to an `InferenceDataOptix` struct that lists the channels belonging to each texture in the texture set, in case they are needed by the shader. Note that it is more efficient to hard-code channel correspondences in the shader rather than query them, however. In shader code, sampling the texture might look like the following:

```
T_VEC_OUT_FLOAT out;
float2 xi = rnd2( seed );
float4 tex1;
float2 tex2;
InferenceDataOptix* infData;

bool resident = ntcTex2DGrad<T_VEC_OUT_FLOAT>( out, infData, dtContext, texId, u, v, ddx, ddy, xi );
tex1 = make_float4( out[0], out[1], out[2], out[3] );
tex2 = make_float2( out[4], out[5] );
```

Keep in mind that individual values are returned by `ntcTex2D*` are not filtered, but the texture coordinates are jittered to achieve stochastic filtering by averaging multiple samples.
