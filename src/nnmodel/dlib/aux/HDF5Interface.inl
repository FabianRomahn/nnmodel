/*
 * HDF5Interface.inl
 *
 *  Created on: May 3, 2019
 *      Author: roma_fa
 */

#include "hdf5.h"
#include "hdf5_hl.h"

template< typename T>
void HDF5Interface::readDataset(std::string const& path, dlib::matrix<T, 0, 1>& data) const
{
	hid_t dataset_id = H5Dopen1(m_nFileId, path.c_str());
	hid_t dataset_type_id = H5Dget_type(dataset_id);
	hid_t dataspace_id = H5Dget_space(dataset_id);

	int nDims = H5Sget_simple_extent_ndims(dataspace_id);

	hsize_t dimExtent[nDims];
	hsize_t dimExtentMax[nDims];

	H5Sget_simple_extent_dims(dataspace_id, dimExtent, dimExtentMax);

	data.set_size(dimExtent[0]);

	H5Dread(dataset_id, dataset_type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(data(0, 0)));
}

template <typename T>
void HDF5Interface::readDataset(std::string const& path, dlib::matrix<T, 0, 0, dlib::default_memory_manager, dlib::row_major_layout>& data) const
{
	hid_t dataset_id = H5Dopen1(m_nFileId, path.c_str());
	hid_t dataset_type_id = H5Dget_type(dataset_id);
	hid_t dataspace_id = H5Dget_space(dataset_id);

	int nDims = H5Sget_simple_extent_ndims(dataspace_id);

	hsize_t dimExtent[nDims];
	hsize_t dimExtentMax[nDims];

	H5Sget_simple_extent_dims(dataspace_id, dimExtent, dimExtentMax);

	data.set_size(dimExtent[0], dimExtent[1]);

	H5Dread(dataset_id, dataset_type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(data(0, 0)));
}

