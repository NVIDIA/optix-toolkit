# OTK Memory Utilities

The OTK Memory utilities provide memory pools to manage both host and device memory. They were built to handle the memory management needs of the OTK demand loading library, and are used extensively by it, and may be useful for other applications. [MemoryPool](/Memory/include/OptiXToolkit/Memory/MemoryPool.h) is the main component.

## Memory Pool

Similar in spirit to the RAPIDS memory manager, `MemoryPool` is a thread-safe memory management class that combines an **allocator** that requests memory chunks from the operating system with a **suballocator** to track and give out memory blocks from the allocated chunks. MemoryPool uses a template design that allows a flexible combination of allocation type and memory management with support for both synchronous and asynchronous operations.

### Allocators

The file [Allocators.h](/Memory/include/OptiXToolkit/Memory/Allocators.h) defines allocator classes that wrap low level memory allocation functions:

- `HostAllocator` uses standard `malloc` / `free` for host memory.
- `PinnedAllocator` uses `cuMemAllocHost` / `cuMemFreeHost` for pinned (page-locked) host memory.
- `DeviceAllocator` uses `cuMemAlloc` / `cuMemFree` for device (GPU) memory.
- `DeviceAsyncAllocator` uses `cuMemAllocAsync` / `cuMemFreeAsync` for asynchronous device memory allocation.
- `TextureTileAllocator` uses `cuMemCreate` / `cuMemRelease` to allocate memory for texture tiles.

### Suballocators

Suballocators track the chunks of memory provided by the allocators and can give out blocks of that memory. The suballocators defined in the OTK are:

- [FixedSuballocator](/Memory/include/OptiXToolkit/Memory/FixedSuballocator.h) manages fixed-size memory blocks. FixedSuballocator is very fast, but can only handle fixed blocks.
- [HeapSuballocator](/Memory/include/OptiXToolkit/Memory/HeapSuballocator.h) manages variable-size memory blocks. HeapSuballocator uses a map to track free blocks, and uses a first fit fulfillment strategy.
- [BinnedSuballocator](/Memory/include/OptiXToolkit/Memory/BinnedSuballocator.h) combines multiple FixedSuballocators for small allocations with a HeapSuballocator for larger ones.
- [RingSuballocator](/Memory/include/OptiXToolkit/Memory/RingSuballocator.h) is like a ring buffer, allowing fast allocation and freeing of variable-sized temporary buffers, assuming all the buffers will be freed quickly.

### Memory Pool Examples

- Create a MemoryPool with PinnedAllocator and HeapSuballocator, using default values:

    ```
    MemoryPool<PinnedAllocator, HeapSuballocator> pool();
    ```

- Create a MemoryPool with a DeviceAllocator for the current CUDA device, and a FixedSuballocator with 64 byte allocations. Allocate the memory in 1 KB chunks, and limit the memory pool to 1 MB total:

    ```
    MemoryPool<DeviceAllocator, FixedSuballocator> 
    pool( new DeviceAllocator(), new FixedSuballocator(64), 1024, 1024*1024 );
    ```

- Create a MemoryPool with a TextureTileAllocator for the current CUDA device, and a HeapSuballocator. Allocate memory in 2 MB chunks, and limit the memory pool to 1 GB:

    ```
    MemoryPool<TextureTileAllocator, HeapSuballocator> pool( 2*1024*1024, 1024*1024*1024 );
    ```

### Using MemoryPool without allocator or suballocator

It is also possible to use a MemoryPool without an allocator, or without a suballocator, by passing in `nullptr` as the constructor argument. 

- When used without an allocator, the pool tracks a set of numbered resources, but does not manage the backing storage for the resources. The application calls the `track` function to indicate what resources need to be tracked.  
    
- When used without a suballocator, the MemoryPool acts as a simple wrapper around the allocator, providing synchronization and CUDA checks.

### Memory allocation and freeing

MemoryPool has a number of allocation functions, some of which are only valid for specific allocators or suballocators:

- `alloc()` allocates a block of memory with a given size and alignment (on a given CUDA stream, if needed)
- `allocItem()` allocates a single item. Only valid for `FixedAllocator`.
- `allocObject()` returns a pointer to memory for an object of a given type.
- `allocObjects()` returns a pointer to memory for an array of objects of a given type.
- `allocTextureTiles()` returns a contiguous set of texture tiles. Only valid for `TextureTileAllocator`.

Each `alloc` function has corresponding `free` and `freeAsync` functions. The async free functions wait on a CUDA stream before releasing the memory. 

### Notes

- Applications should always use a MemoryPool rather than trying to use an allocator or suballocator by itself, since it provides synchronization and stream checking.

- The general `alloc` function returns a `MemoryBlockDesc` struct. Some of the convenience alloc methods return pointers instead. Applications should use the `free` function associated with the `alloc` function used.

- `allocTextureTiles` returns a `TileBlockHandle` instead of a `MemoryBlockDesc`. This deals with the handle-based memory allocation of `cuMemCreate`.

- MemoryPool does not keep a record of allocated blocks, only free blocks.

