#!/bin/sh
# build the necessary resources for all parts of the optimized version

echo "This script builds the libraries in the resources folder"

RESOURCEDIR="$(pwd)/../res"
COMPILEDIR="$(pwd)/tmp"
INSTALLDIR="$(pwd)/inst"

# save the location of the current directory
BASEDIR=$(pwd)

# first check if the installation directory is already available
if [ -d $INSTALLDIR ]; then
    echo "Installation directory is already available - will not build the resources again ..."
    exit 0
fi

if [ ! -d $RESOURCEDIR ]; then
    echo "[ERROR] Could not find resource directory: $RESOURCEDIR"
    exit -1
fi

if [ ! -d $INSTALLDIR ]; then
    echo "Creating installation directory: $INSTALLDIR"
    mkdir -p $INSTALLDIR
fi

if [ ! -d $COMPILEDIR ]; then
    echo "Creating temporary compilation directory: $COMPILEDIR"
    mkdir -p $COMPILEDIR
fi

HDF5PACKAGE=`ls -1rth $RESOURCEDIR/hdf5*|tail -1`
EIGENPACKAGE=`ls -1rth $RESOURCEDIR/eigen-*|tail -1`
DLIBPACKAGE=`ls -1rth $RESOURCEDIR/dlib-*|tail -1`
OPENBLASPACKAGE=`ls -1rth $RESOURCEDIR/OpenBLAS-*|tail -1`
BOOSTPACKAGE=`ls -1rth $RESOURCEDIR/boost*|tail -1`

# determine the number of jobs for the make parameters
MAKEPAR="-j`nproc`"

echo "
found libraries:
hdf5:         $HDF5PACKAGE
eigen:        $EIGENPACKAGE
dlib:         $DLIBPACKAGE
open-blas:    $OPENBLASPACKAGE
boost:        $BOOSTPACKAGE

using make parameters: $MAKEPAR
"

# sleep some seconds so that the packages can be reviewed
sleep 3s

# make a lib directory and link the lib64 directory to it
cd $INSTALLDIR
mkdir lib
ln -s lib lib64
cd -

# HDF5
if [ $HDF5PACKAGE != "" ]; then
    echo "Installing hdf5 ..."
    cd $COMPILEDIR
    tar xf $HDF5PACKAGE
    cd hdf5*
    ./configure --prefix=$INSTALLDIR
    make $MAKEPAR
    make install
    cd $BASEDIR
    echo "Finished installing hdf5 ..."    
else
    echo "[ERROR] Could not find hdf5 library"
fi

# EIGEN
if [ $EIGENPACKAGE != "" ]; then
    echo "Installing eigen ..."
    cd $COMPILEDIR
    tar xf $EIGENPACKAGE
    cd eigen-*
    mkdir build
    cd build
    cmake ..
    cmake . -DCMAKE_INSTALL_PREFIX=$INSTALLDIR
    make $MAKEPAR
    make install
    cd $BASEDIR
    echo "Finished eigen installation"
else
    echo "[ERROR] Could not find eigen library"
fi

# OPENBLAS
if [ $OPENBLASPACKAGE != "" ]; then
    echo "Installing OpenBLAS"
    cd $COMPILEDIR
    tar xf $OPENBLASPACKAGE
    cd OpenBLAS-*    
    make $MAKEPAR USE_OPENMP=1
    make PREFIX=$INSTALLDIR install
    cd $BASEDIR
    echo "Finsihed OpenBLAS installation"
else
    echo "[ERROR] Could not find OpenBLAS library"
fi

# DLIB
if [ $DLIBPACKAGE != "" ]; then
    echo "Installing dlib"
    cd $COMPILEDIR
    tar xf $DLIBPACKAGE
    cd dlib-*
    mkdir build
    cd build
    cmake ..
    cmake . -DCMAKE_INSTALL_PREFIX=$INSTALLDIR
    make $MAKEPAR
    make install
    cd $BASEDIR
    echo "Finished dlib installation"
else
    echo "[ERROR] Could not find dlib library"
fi

# BOOST
if [ $BOOSTPACKAGE != "" ]; then
    echo "Installing boost"
    cd $COMPILEDIR
    tar xf $BOOSTPACKAGE
    cd boost*
    ./bootstrap.sh --prefix=$INSTALLDIR
    ./b2 install --with-program_options
    cd $BASEDIR
    echo "Finished boost installation"
else
    echo "[ERROR] Could not find boost library"
fi

# finally remove the temporary compilation directory
rm -r $COMPILEDIR 

echo "Resources have been installed to $INSTALLDIR"
echo "Finished all"
