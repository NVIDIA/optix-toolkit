## CompressedTextureCache utilty

The OTK `compressedTextureCache` utility is designed to compress texture files (and particularly EXRs) to **Block Compressed (BC)** or **Neural Texture (NTC)** formats suitable for the OTK demand texture library. 

### Block Compressed Textures

Block compressed textures (BC1 ... BC7) are a family of texture compression formats designed to reduce GPU texture memory usage. They are well established, and have been used extensively in the game industry for over a decade. Each of the formats is designed to handle a specific texturing need, summarized as follows:

- BC1 stores RGB textures at 4 bits per texel. It has the highest compression ratio of the color formats, but only supports one bit alpha. Quality is lower compared to BC6 and BC7.
- BC2 and BC3 handle RGBA textures. These have been superceded by BC7 which takes the same space with higher quality.
- BC4 and BC5 are designed for 1 and 2 component textures, respectively.
- BC6 supports high dynamic range RGB images, but does not include an alpha channel.
- BC7 provides high quality for RGB and RGBA textures.

The BC format encodings divide textures into 4x4 pixel blocks that are compressed as either 8 or 16 bytes (4 or 8 bits per texel). Memory savings can be substantial, typically 2-8x smaller than uncompressed textures. The table below gives on-GPU compression ratios for different BC formats compared to textures they would replace:

| format | components | bits/pixel | uchar | uchar2 | uchar4 | half | half2 | half4 |
| ------ | ---------- | :--------: | :---: | :----: | :----: | :--: | :---: | :---: |
| BC1    | RGB(A)     | 4          |       |        | 8x     |      |       | 16x   |
| BC2    | RGBA       | 8          |       |        |        |      |       |       |
| BC3    | RGBA       | 8          |       |        |        |      |       |       |
| BC4    | R          | 4          | 2x    |        |        | 4x   |       |       |
| BC5    | RG         | 8          |       | 2x     |        |      | 4x    |       |
| BC6    | RGB (HDR)  | 8          |       |        |        | 2x   | 4x    | 8x    |
| BC7    | RGBA       | 8          |       |        | 4x     |      |       | 8x    |

BC textures render as fast as uncompressed textures thanks to hardware support. Also, they can be tiled to work seamlessly with the OTK demand loading library. On disk, BC texture files are usually stored in `DDS` image format. The OTK compressedTextureCache utility creates a *caching variant* of DDS format that is tiled on disk. That way individual tiles can be read from disk at runtime.

In the past, BC textures were considered slow to compress (for example see the excellent article [Understanding BCn Texture Compression Formats](https://www.reedbeta.com/blog/understanding-bcn-texture-compression-formats/)), but current tools that compress on the GPU can convert large images to BC formats in only a few seconds per texture.

The compressedTextureCache utility relies on the [NVIDIA Texture Tools](https://developer.nvidia.com/gpu-accelerated-texture-compression) `nvcompress` program for BC compression.

### Compressing to BC format using the compressedTextureCache utility

**Basic usage:**

```
compressedTextureCache [options] <src folders or files>
```

**Required options for BC compression**

```
--nvcompress | -nc <nvcompress path + flags>   Path the the nvcompress program plus flags to the program
--cacheFolder | -cf <cache folder>             Cache folder location
```

**Example**

```
# Compress all images in srcImages folder. Save the result to compressedCache.
./compressedTextureCache --nvcompress /path/to/nvcompress --cacheFolder compressedCache srcImages
```

**Compression profiles**

When compressing images to BC formats, compressedTextureCache analyzes each image and uses a profile to decide which BC format to use. Two profiles, **standard** or **small** are available. The table below give the BC format for different image types:


| Image Type         | non-EXR | R   | RG  | RGB | RGBA | R(hdr) | RG(hdr) | RGB(hdr) |
| ------------------ | ------- | --- | --- | --- | ---- | ------ | ------- | -------- |
| BC type (standard) | BC7     | BC4 | BC5 | BC7 | BC7  | BC6    | BC6     | BC6      |
| BC type (small)    | BC1     | BC4 | BC1 | BC1 | BC7  | BC6    | BC6     | BC6      |

**Tiled DDS files**

By default, compressed images are saved to a **tiled DDS** file format that can be read by the OTK demand loading library. Tiling the files on disk allows individual tiles to be read from disk at runtime, similar to how tiled EXR images are read. This feature can be turned off using the `--noTile` command line option, so that standard DDS files are saved. These will still be tiled on the GPU by the demand loading system at runtime, but requested mip levels will be cached on the cpu. 

**Additional options**

```
--dropMipLevels | -dl <numLevels>    mip levels to drop when putting files in cache. (default 0)
--small | -s                         use the small profile for higher compression BC formats. (default standard)
--threads | -t <numThreads>          number of threads to use. (default 8)
--multiGPU | -mg                     use multiple GPUs if available. (default off)
--noTile | -nt                       turn off tiling dds outputs.
```

**BC compression speed** 

The nvcompress tool used by `compressedTextureCache` is fast enough to make BC compression an option for interactive workflows, just a few seconds per texture. Below are times to compress EXR textures to BC7 using a GeForce 5080 GPU (other BC formats are even faster). The batching and multithreading really help here, so that a batch of 8K textures can be compressed in about 1.5 seconds per texture. Compression time and cache size can be further reduced by dropping mip levels when the use case allows it.

| Texture size | Single texture | 40 textures |
| :----------: | :------------: | :---------: |
| 8k x 8k      |  8.8 sec.      | 61.7 sec.   |
| 4k x 4k      |  2.2           | 15.5        |
| 2k x 2k      |  0.9           |  7.3        |
| 1k x 1k      |  0.5           |  6.7        |

### Neural texture compression

[Neural texture compression](https://github.com/NVIDIA-RTX/RTXNTC) (NTC) uses deep learning models to achieve surprisingly high compression, around an order of magnitude higher than the BC formats for similar quality. This high compression comes at a cost, however.  To achieve it, textures are bundled into sets. They are slower to render and compress than BC textures, and cannot take advantage of hardware-assisted filtering. Still, the tiny GPU footprint makes them an attractive option. For neural textures, the compressedTextureCache utility uses the `ntc-cli` program from the [RTX Neural Texture Compression SDK](https://github.com/NVIDIA-RTX/RTXNTC) to do compression.

An NTC texture is stored as a mipmapped *latent image* and a small *neural network* to decode the latents. This representation achieves high compression by compounding several factors:

- The latent image has lower resolution than the input image, typically 4x smaller.
- The components of the latent image are stored as 4 bit values.
- Multiple images can be compressed together as a single *texture set*.
- The neural network size is negligible, under 32 KB.

As an example, consider a material that has 2k mipmapped textures with 10 total channels: diffuseColor(3), specularColor(3), normal(2), roughness(1), and metalness(1). If each channel takes 1 byte, the texture memory usage is `10 bytes * 2k * 2k * 4/3 = 53.3 MB`. Suppose that the neural texture for this texture set has latents with 8 components. Latent pixels then occupy 4 bytes and total memory usage for the latents is `4 bytes * 512 * 512 * 4/3 = 1.33 MB`, about 40x compression.

Another benefit of the NTC format is that all texture channels are inferred as float16.  The inference always returns a high dynamic range value, so there is no need to distinguish between high and low dynamic range textures.

**Decompression**

To decompress a texel in an NTC image, a program samples the latent images at the texel coordinates and feeds these values, along with some spatial information, into the network which decodes the texel. The decoding step produces an unfiltered texel value for all the textures in the texture set. This can be done in real time because the decoder takes advantage of tensor cores on the GPU to run the network. Since decoded texel values are unfiltered, applications rely on *stochastic texture filtering* to smooth out the result.

### Compressing to NTC format using the compressedTextureCache utility

The `compressedTextureCache` program relies calls out to the `ntc-cli` utility to compress texture sets.  Texture sets can be provided to the compressedTextureCache in one of two ways:

- List the output .ntc output file followed by texture set input images on the command line.
- List output .ntc files and input images for them in a .txt file, which is included on the command line.

Other command line options for NTC compression include

```
--ntc-cli <ntc-cli executable>         ntc-cli executable.
--inputFolder | -if <input folder>     input folder to read texture sets from.
--numFeatures | -nf <numFeatures>      number of latent features. (default 8)
--flags | -f                           extra flags to pass to ntc-cli                        
```

**Examples**

```
# Compress the the texture set {a.exr, b.exr} to c.ntc using 4 latent features
./compressedTextureCache --ntc-cli /path/to/ntc-cli --cacheFolder compressedCache --numFeatures 4 c.ntc a.exr b.exr

# Compress the texture sets in contained in texsets.txt.
./compressedTextureCache --ntc-cli /path/to/ntc-cli --cacheFolder compressedCache --inputFolder srcImages texsets.txt

# Contents of texsets.txt
a.ntc
a1.exr
a2.exr

b.ntc
b1.exr
b2.exr
b3.exr
...
```

### Ntc compression speed

Neural texture compression is fast enough to be practical as a batch process. For EXR textures, reading the texture data from disk and decompressing it can take up to 50% of the time. Multithreading does not increase the compression throughput on a single GPU machine (as it does with BC compression), since the compression kernels saturate the graphics card.  While NTC compression is slower than BC compression, the dramatic reduction in GPU memory footprint makes the additional compression time worthwhile for applications where memory usage is critical. The table below gives compression speeds for different sized EXR (4 image) texture sets on a machine with a GeForce 5080 GPU:

| Texture size | Single texture set (4 textures) | 10 texture sets (40 textures) |
| :----------: | :------------: | :---------: |
| 8k x 8k      |  39 sec.       |  403 sec.   |
| 4k x 4k      |  20            |  196        |
| 2k x 2k      |  14            |  137        |
| 1k x 1k      |  12            |  124        |
