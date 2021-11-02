/*
 * HDF5Interface.cpp
 *
 *  Created on: Apr 24, 2019
 *      Author: roma_fa
 */

#include "HDF5Interface.hpp"

#include <iostream>

HDF5Interface::HDF5Interface() :
	m_nFileId(-1)
{

}

HDF5Interface::HDF5Interface(std::string const& fileName) :
	m_nFileId(-1)
{
	openFile(fileName);
}

HDF5Interface::~HDF5Interface()
{
	closeFile();
}

void HDF5Interface::openFile(std::string const& fileName)
{
	// Just to be sure, close the current file - if there was one
	closeFile();
	// Open the file
	m_nFileId = H5Fopen(fileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
}

void HDF5Interface::closeFile()
{
	if (m_nFileId > 0)
	{
		H5Fclose(m_nFileId);
		m_nFileId = -1;
	}
}

bool HDF5Interface::checkExistence(std::string const& path) const
{
	return H5Lexists(m_nFileId, path.c_str(), H5P_DEFAULT);
}

bool HDF5Interface::checkAttributeExistence(std::string const& attributePath) const
{
	// Split the path in the group and attribute itself
	size_t idx = attributePath.rfind("/");
	if (idx == std::string::npos)
	{
		// The attribute is in the root path
		return H5Aexists(m_nFileId, attributePath.c_str());
	}
	else
	{
		// The attribute belongs to a group, therefore the existence of the group is checked first
		size_t n = attributePath.size() - idx - 1;
		std::string attributeName = attributePath.substr(idx+1, n);
		std::string groupPath = attributePath.substr(0, idx);
		bool bGroupExists = checkExistence(groupPath);
		if (!bGroupExists)
		{
			return false;
		}
		else
		{
			// Open the group
			hid_t nLocId = H5Gopen1(m_nFileId, groupPath.c_str());
			// Check if the attribute exists
			return H5Aexists(nLocId, attributeName.c_str());
		}
	}
}

std::string HDF5Interface::getAttribute(std::string const& attributePath) const
{
	// Split the path in the group and attribute itself
	size_t idx = attributePath.rfind("/");
	std::string attributeName = "";
	hid_t nLocId;
	if (idx == std::string::npos)
	{
		// The attribute is in the root path
		attributeName = attributePath;
		nLocId = m_nFileId;
	}
	else
	{
		// The attribute belongs to a group, therefore the group has to be opened first
		size_t n = attributePath.size() - idx - 1;
		attributeName = attributePath.substr(idx+1, n);
		std::string groupName = attributePath.substr(0, idx);
		nLocId = H5Gopen1(m_nFileId, groupName.c_str());
	}
	// Get the attribute id
	hid_t nAttrId = H5Aopen(nLocId, attributeName.c_str(), H5P_DEFAULT);
	// Get the type id of the attribute
	hid_t nAttrTypeId = H5Aget_type(nAttrId);
	// Get the native type id of the attribute
	hid_t nAttrNativeTypeId = H5Tget_native_type(nAttrTypeId, H5T_DIR_ASCEND);
	// Declare a buffer for reading the attribute
	char * attribute_buffer;
	// Read the attribute
	H5Aread(nAttrId, nAttrNativeTypeId, &attribute_buffer);
	// Convert the buffer to a string
	std::string attributeValue (attribute_buffer);
	// Close the attribute
	H5Aclose(nAttrId);
	// Return the value
	return attributeValue;
}

unsigned int HDF5Interface::getGroupNumObjects(std::string const& path) const
{
	// Open the group
	hid_t nGroupId = H5Gopen1(m_nFileId, path.c_str());
	// get the number of objects
	hsize_t numObjects = 0;
	H5Gget_num_objs(nGroupId, &numObjects);
	// Close the group
	H5Gclose(nGroupId);
	// return the result
	return numObjects;
}

std::string HDF5Interface::getObjectName(std::string const& path, unsigned int id) const
{
	// Open the group
	hid_t nGroupId = H5Gopen1(m_nFileId, path.c_str());
	// Get the name
	char object_name_buffer [255];
	hsize_t iObjId = id;
	H5Gget_objname_by_idx(nGroupId, iObjId, object_name_buffer, 255);
	// Convert the name to a string
	std::string objectName (object_name_buffer);
	// Close the group
	H5Gclose(nGroupId);
	// return the result
	return objectName;
}
