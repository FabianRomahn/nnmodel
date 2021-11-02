cmake_minimum_required(VERSION 2.8)

project ( nnmodel CXX Fortran )

# If no build type is specified, set it to RELEASE
if (NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE RELEASE )
endif (NOT CMAKE_BUILD_TYPE)

# Set the CXX Flags
set (CMAKE_CXX_FLAGS_RELEASE "-funroll-all-loops -O3 -fopenmp")
set (CMAKE_CXX_FLAGS_DEBUG   "-O0 -ggdb")

# Set the Fortran Flags
set (CMAKE_Fortran_FLAGS_RELEASE "-funroll-all-loops -fno-f2c -O3 -fopenmp -cpp")
set (CMAKE_Fortran_FLAGS_DEBUG   "-fno-f2c -O0 -ggdb -fbounds-check -fopenmp -cpp")

# Print out information about the compiling flags, etc.

# Fortran Flags
message(STATUS "CMAKE_Fortran_FLAGS_DEBUG: " ${CMAKE_Fortran_FLAGS_DEBUG})
message(STATUS "CMAKE_Fortran_FLAGS_RELEASE: " ${CMAKE_Fortran_FLAGS_RELEASE})

# CXX Flags
message(STATUS "CMAKE_CXX_FLAGS_DEBUG: " ${CMAKE_CXX_FLAGS_DEBUG})
message(STATUS "CMAKE_CXX_FLAGS_RELEASE: " ${CMAKE_CXX_FLAGS_RELEASE})

# Implicit link libraries
message(STATUS "CMAKE_FORTRAN_IMPLICIT_LINK_LIBRARIES: ${CMAKE_FORTRAN_IMPLICIT_LINK_LIBRARIES}")
message(STATUS "CMAKE_CXX_IMPLICIT_LINK_LIBRARIES: ${CMAKE_CXX_IMPLICIT_LINK_LIBRARIES}")

# add the directory flags to the compile flags
SET ( COMPILE_FLAGS "${COMPILE_FLAGS} ${VLIDORT_DIR_FLAGS}" )

# Set the resource installation directory and the corresponding include and library directories
# Some libraries (like netcdf4-c++) are needed from there if they are not part of the system

set ( INST_DIR "${CMAKE_SOURCE_DIR}/../build/inst" )
set ( INST_INC_DIR "${INST_DIR}/include" )
set ( INST_LIB_DIR "${INST_DIR}/lib64" )

if ( EXISTS "${INST_DIR}" )
	# normalize path
	get_filename_component( INST_DIR ${INST_DIR} ABSOLUTE )
	message ( STATUS "Found installation directory: ${INST_DIR} " )
	if ( EXISTS ${INST_INC_DIR} )
		# normalize the path
		get_filename_component( INST_INC_DIR ${INST_INC_DIR} ABSOLUTE ) 		
        message ( STATUS "Found installation include directory: ${INST_INC_DIR}" )
        include_directories( ${INST_INC_DIR} )
	else ()
        message ( FATAL_ERROR "Installation include directory not found: ${INST_INC_DIR}" )	
	endif ()
	
	if ( EXISTS ${INST_LIB_DIR} )
		# normalize the path
		get_filename_component( INST_LIB_DIR ${INST_LIB_DIR} ABSOLUTE ) 		
        message ( STATUS "Found installation library directory: ${INST_LIB_DIR}" )
        link_directories( ${INST_LIB_DIR} )	
	else ()
        message ( FATAL_ERROR "Installation library directory not found: ${INST_LIB_DIR}" )		
	endif ()
else ()
	message ( FATAL_ERROR "Installation directory ${INST_DIR} not found - please install the necessary resources there by running build_res.sh" )
endif ()

message ( STATUS "building nnmodel" )
add_subdirectory ( nnmodel )
# change the path of the library to the main binary directory            
set_target_properties( NNModel PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR} )
# change the path of the executable to the main binary directory            
set_target_properties( NNModelTest_eigen PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR} )
set_target_properties( NNModelTest_dlib PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR} )
