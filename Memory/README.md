# Memory allocators - OptiX Toolkit

This repository is part of the [OptiX Toolkit](https://github.com/NVIDIA/optix-toolkit).

For recent changes, see the [CHANGELOG](CHANGELOG.md).

It contains three libraries:

- Error is a header-only library that contains a mechanism for mapping error codes to C++ exceptions.
The headers are included from `<OptiXToolkit/Error>`, see [ErrorCheck.h](include/OptiXToolkit/Error/ErrorCheck.h) for details.

- [Memory](/Memory/include/OptiXToolkit/Memory/) contains a number of memory allocators that are used in the [Demand Loading library](https://github.com/NVIDIA/otk-demand-loading), which might also be useful in client applications.

- OptiXMemory is a header-only library that contains memory related helper classes
(`ProgramGroupDescBuilder`, `BuildInputBuilder`, `Record<T>`, and `SyncRecord<T>`) for OptiX
applications.
