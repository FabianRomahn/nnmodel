cmake_minimum_required (VERSION 2.8)

project ( NNModel CXX )

# find the necessary libraries
find_library ( lib_dlib NAMES dlib PATHS ${INST_LIB_DIR} NO_DEFAULT_PATH )
find_library ( lib_openblas NAMES openblas PATHS ${INST_LIB_DIR} NO_DEFAULT_PATH )

add_library ( HDF5Interface_dlib
                aux/HDF5Interface.cpp )

add_library ( NNModel_dlib
				NNModel.cpp
                NNModelLayer.cpp
                NNModelScaler.cpp
                NNModelActivationFunctions.cpp )

target_link_libraries ( HDF5Interface_dlib
                        hdf5 )

target_link_libraries ( NNModel_dlib
                        HDF5Interface_dlib
                        ${lib_dlib}
                        ${lib_openblas} )
                        
add_executable ( NNModelTest_dlib
                    main.cpp )
                    
target_link_libraries ( NNModelTest_dlib
                        NNModel_dlib
                        boost_program_options )