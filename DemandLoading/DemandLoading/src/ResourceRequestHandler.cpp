// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/DemandLoading/DemandLoadLogger.h>
#include "ResourceRequestHandler.h"
#include "DemandLoaderImpl.h"

namespace demandLoading {

void ResourceRequestHandler::fillRequest( CUstream stream, unsigned int pageIndex ) 
{
    // We use MutexArray to ensure mutual exclusion on a per-page basis.  This is necessary because
    // multiple streams might race to fill the same tile (or the mip tail).
    unsigned int index = pageIndex - m_startPage;
    MutexArrayLock lock( m_mutex.get(), index);

    DL_LOG(4, "[Page " + std::to_string(pageIndex) + "] Resource request.");

    // Do nothing if the request has already been filled.
    PagingSystem* pagingSystem = m_loader->getPagingSystem();
    if( pagingSystem->isResident( pageIndex ) )
        return;

    // Invoke the callback that was provided when the resource was created, which returns a new page table entry.
    void* pageTableEntry;
    if( m_callback( stream, pageIndex, m_callbackContext, &pageTableEntry ) )
    {
        // Add a page table mapping from the requested page index to the new page table entry.
        // Page table updates are accumulated in the PagingSystem until launchPrepare is called, which
        // sends them to the device (via PagingSystem::pushMappings).
        m_loader->setPageTableEntry( pageIndex, false, reinterpret_cast<unsigned long long>( pageTableEntry ) );
    }
}

}
