cmake_minimum_required (VERSION 2.8)

project ( NNModel CXX )

# find the necessary libraries
find_library ( lib_boost_program_options boost_program_options PATHS ${INST_LIB_DIR} NO_DEFAULT_PATH )

add_library ( HDF5Interface
                aux/HDF5Interface.cpp )

add_library ( NNModel
                NNModel.cpp
                NNModelLayer.cpp
                NNModelScaler.cpp
                NNModelActivationFunctions.cpp )

target_link_libraries ( HDF5Interface
                        hdf5 )

target_link_libraries ( NNModel
                        HDF5Interface )
                        
add_executable ( NNModelTest_eigen
                    main.cpp )

target_link_libraries ( NNModelTest_eigen
                        NNModel
                        ${lib_boost_program_options} )