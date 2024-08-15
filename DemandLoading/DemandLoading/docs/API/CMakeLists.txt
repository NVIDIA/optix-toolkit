# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#

include(GNUInstallDirs)

find_package( Doxygen )

if( NOT DOXYGEN_FOUND )
  message( WARNING "Doxygen not found.  Skipping DemandLoading library documentation." )
  return()
endif()

set( OTK_DOCS_EXTRACT_ALL YES CACHE STRING "Extract documentation for all symbols (YES/NO)" )
set( OTK_DOCS_WARN_IF_UNDOCUMENTED NO CACHE STRING "Warn about undocumented symbols (YES/NO)" )

configure_file( Doxyfile Doxyfile @ONLY )
configure_file( header.tex header.tex COPYONLY )
configure_file( footer.tex footer.tex COPYONLY )
configure_file( cuda_header.png latex/cuda_header.png COPYONLY )

add_custom_target( docs ALL
  ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating API documentation with Doxygen: ${CMAKE_CURRENT_BINARY_DIR}" VERBATIM )

install( DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html
  DESTINATION ${CMAKE_INSTALL_DOCDIR}/DemandLoading )
