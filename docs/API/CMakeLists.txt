# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(GNUInstallDirs)

find_package( Doxygen )

if( NOT DOXYGEN_FOUND )
  message( WARNING "Doxygen not found.  Skipping build of API documentation." )
  return()
endif()

set( OTK_DOCS_EXTRACT_ALL YES CACHE STRING "Extract documentation for all symbols (YES/NO)" )
set( OTK_DOCS_WARN_IF_UNDOCUMENTED NO CACHE STRING "Warn about undocumented symbols (YES/NO)" )

set( OTK_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../.. )
configure_file( Doxyfile Doxyfile @ONLY )

add_custom_target( docs ALL
  ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating API documentation with Doxygen: ${CMAKE_CURRENT_BINARY_DIR}" VERBATIM )

install( DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/API
  DESTINATION ${CMAKE_INSTALL_DOCDIR} )
