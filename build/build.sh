#!/bin/sh
# create the nnmodel build after checkout

# definition of the source directories
declare -a SOURCEDIRS=("nnmodel")

# this function gets the build directory as a parameter and then builds the binaries
build()
{

BUILDDIR=$1 && shift
EXTRAFLAGS=($@)

echo "TARGET DIRECTORY: $BUILDDIR"
echo "EXTRAFLAGS SET TO: $EXTRAFLAGS"

# remove the build directory if it already existed
if [ -d $BUILDDIR ]; then
	rm -rf $BUILDDIR
fi

# create the build directory
mkdir -p $BUILDDIR
cd $BUILDDIR

# calculate the length of the build-subdiretory
IFS="/" read -ra BUILDDIR_SEP <<< "$BUILDDIR"
BUILDDIR_LEN=${#BUILDDIR_SEP[@]}

# create the path to the source code for cmake depending on the build path length
SRCPATH="../src"
for ((i=1;i<=$BUILDDIR_LEN;i++));
do    
    SRCPATH=../"$SRCPATH"
done

cmake $SRCPATH ${EXTRAFLAGS[@]}
make $PARALLELFLAGS
cd -

}

SRCDIR=""
BUILDMODE=""
BUILDDIRSUFFIX=""
BUILDDIR=""
EXTRAFLAGS=()
PARALLELFLAGS="-j8"

BUILDRESSCRIPT="./build_res.sh"

# first of all call the script to build the resources
echo "calling $BUILDRESSCRIPT to build the necessary resources"
$BUILDRESSCRIPT

# then, check the arguments

# check the first argument
if [ ! -z "$1" ]; then
	# check if the argument is a build mode
	case $1 in
		-d|-r) BUILDMODE=$1 ;;
	esac
		
	# if the build mode was not set, check if the argument is a valid source directory
	if [ "$BUILDMODE" == "" ]; then
		# check if the argument is a valid source directory
		found=false
		for item in "${SOURCEDIRS[@]}"
		do
			if [ $item == $1 ]; then
				found=true
				break
			fi
		done
				
		if [ "$found" = true ]; then
			SRCDIR=$1
			EXTRAFLAGS+=("-DBUILD_${1^^}=ON")		
		else
			echo "$1 is an invalid source directory!"; exit -1
		fi
	fi
fi

# if the build mode is not set - check the second argument
if [ "$BUILDMODE" == "" ]; then
	if [ ! -z "$2" ]; then
		case $2 in
			-d|-r) BUILDMODE=$2 ;;
			*) echo "$2 is an invalid build mode!"; exit -1 ;;
		esac
	fi
fi

if [ "$SRCDIR" == "" ]; then
	echo "no source directory specified - building all sources"
	EXTRAFLAGS+=("-DBUILD_ALL=ON")
else
	echo "source directory is: $SRCDIR"
fi

if [ "$BUILDMODE" == "" ]; then
	echo "no build mode specified - setting it to: release"
	BUILDMODE="-r"
else
	if [ "$BUILDMODE" == "-d" ]; then
		echo "build mode is set to: debug"
	elif [ "$BUILDMODE" == "-r" ]; then
		echo "build mode is set to: release"
	fi
fi

# set the directory suffix and the flags according to the build mode 
if [ "$BUILDMODE" == "-d" ]; then
	BUILDDIRSUFFIX="debug"
	EXTRAFLAGS+=('-DCMAKE_BUILD_TYPE=DEBUG')
elif [ "$BUILDMODE" == "-r" ]; then	
	BUILDDIRSUFFIX="release"
	EXTRAFLAGS+=('-DCMAKE_BUILD_TYPE=RELEASE')
fi

if [ ! -z "$SRCDIR" ]; then
	BUILDDIR="$SRCDIR"/"$BUILDDIRSUFFIX"
	build $BUILDDIR ${EXTRAFLAGS[@]}
else
	BUILDDIR="$BUILDDIRSUFFIX"
	echo "BUILDDIR: $BUILDDIR"
	echo "SRCDIR: $SRCDIR"
	echo "EXTRAFLAGS: $EXTRAFLAGS"
	build $BUILDDIR ${EXTRAFLAGS[@]}
fi

exit 0
