The udimTextureViewer demonstrates how to create an use UDIM textures
(arrays of texture images treated as a single texture) in the OptiX 
demand loading library.

This sample can also stress the texturing system by allocating large arrays 
of textures.  The command 

udimTextureViewer --texdim=8192x8192 --udim=32x32

creates an array of 32x32 textures, each with resolution 8192x8192. Fully 
loaded, these textures would require is 1.33 TB of GPU storage. 
