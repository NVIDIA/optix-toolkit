# Neural Textures

The NeuralTextures module supports on-demand streaming of neural textures in OptiX applications using the DemandLoading library, enabling high-quality texture compression with GPU-accelerated neural network inference.

## Overview

Neural Texture Compression (NTC) is a lossy compression format that uses neural networks to achieve very high compression ratios while maintaining visual quality. NTC files store textures as compressed latent features and a small multi-layer perceptron (MLP) network. During rendering, the neural network performs inference on-demand to reconstruct texture values, enabling efficient GPU memory usage.

### Benefits

- **High compression ratios**: Significantly better compression than traditional formats
- **GPU-accelerated inference**: Neural network evaluation runs efficiently on GPU using OptiX
- **On-demand streaming**: Works with the demand loading library to stream texture data as needed
- **Quality preservation**: Maintains high visual quality even at aggressive compression ratios

### Integration

The NeuralTextures library implements the `ImageSource` interface from the demand loading library, allowing neural textures to be used anywhere standard textures are supported. The library handles loading `.ntc` files, managing inference data, and providing device-side sampling functions.

To create `.ntc` files from source images, use the [Neural Texture SDK](https://github.com/NVIDIA-RTX/RTXNTC).

## Architecture

The library consists of host-side and device-side components:

### Host-Side

- **`NeuralTextureSource`** - Implements `ImageSource`, bridging `.ntc` files and the demand loading system
- **`NtcImageReader`** - Parses `.ntc` files, extracting latent features and MLP weights
- **`InferenceDataOptix.h`** - Stores per-device inference data (latent textures, MLP weights)

### Device-Side

- **`Texture2DNeural.h`** - Sampling functions: `ntcTex2DGrad()`, `ntcTex2DGradUdim()`
- **`InferenceOptix.h`** - OptiX inference kernels for MLP evaluation

## Quick Start Guide

The following guide demonstrates how to use neural textures in an OptiX application. 

See the [NeuralTextureViewer](../examples/DemandLoading/NeuralTextureViewer/) example for a complete working implementation that demonstrates:
- Loading single and multiple neural textures
- Creating UDIM textures
- Device-side sampling with different filtering modes
- Interactive texture viewing

### Host-Side Setup

```cpp
#include <OptiXToolkit/NeuralTextures/NeuralTextureSource.h>
#include <OptiXToolkit/DemandLoading/DemandLoader.h>
#include <OptiXToolkit/DemandLoading/TextureSampler.h>
using namespace neuralTextures;

// Create texture source and descriptor
NeuralTextureSource* neuralTextureSource = new NeuralTextureSource("texture.ntc");
demandLoading::TextureDescriptor texDesc = makeTextureDescriptor(
    CU_TR_ADDRESS_MODE_CLAMP, FILTER_BILINEAR);  // Also: FILTER_POINT, FILTER_SMARTBICUBIC; FILTER_BILINEAR is a good default
texDesc.flags |= CU_TRSF_READ_AS_INTEGER;

// Create demand-loaded texture
std::shared_ptr<imageSource::ImageSource> source(neuralTextureSource);
const demandLoading::DemandTexture& texture = demandLoader->createTexture(source, texDesc);
unsigned int textureId = texture.getId();
```

For UDIM textures (multiple textures in a grid), use `createUdimTexture()`:

```cpp
std::vector<std::shared_ptr<imageSource::ImageSource>> subImageSources;
std::vector<demandLoading::TextureDescriptor> subTexDescs;

for (const auto& filename : textureFiles) {
    subImageSources.push_back(std::make_shared<NeuralTextureSource>(filename));
    subTexDescs.push_back(texDesc);
}

const demandLoading::DemandTexture& udimTexture = 
    demandLoader->createUdimTexture(subImageSources, subTexDescs, 10, 10, -1);
```

### Device-Side Sampling

```cpp
#include <OptiXToolkit/NeuralTextures/InferenceOptix.h>
#include <OptiXToolkit/NeuralTextures/Texture2DNeural.h>
using namespace neuralTextures;

// Sample the neural texture
T_VEC_OUT_FLOAT out;  // Defined in Texture2DNeural.h
bool resident = ntcTex2DGrad<T_VEC_OUT_FLOAT>(
    out, context, textureId, uv.x, uv.y, ddx, ddy, xi);

// 'resident' indicates whether texel data was present; callers can supply a fallback color when false
if (resident) {
    float4 color = float4{out[0], out[1], out[2], out[3]};
}

// For UDIM textures, use ntcTex2DGradUdim() instead
```

## See Also

- [Demand Loading Library](../DemandLoading/)
- [Neural Texture SDK](https://github.com/NVIDIA-RTX/RTXNTC)
- [NeuralTextureViewer Example](../examples/DemandLoading/NeuralTextureViewer/)
