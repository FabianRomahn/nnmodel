/*
 * HDF5Interface.hpp
 *
 *  Created on: Apr 24, 2019
 *      Author: roma_fa
 */

#ifndef HDF5INTERFACE_HPP_
#define HDF5INTERFACE_HPP_

#include <string>
#include <memory>

#include <dlib/matrix/matrix.h>

typedef int64_t hid_t;

class HDF5Interface;
typedef std::shared_ptr< HDF5Interface > HDF5InterfaceSPtr;
typedef std::shared_ptr< const HDF5Interface > HDF5InterfaceCSPtr;

class HDF5Interface
{

public:

	HDF5Interface();

	HDF5Interface(std::string const& fileName);

	~HDF5Interface();

	void openFile(std::string const& fileName);
	void closeFile();

	bool checkExistence(std::string const& path) const;
	bool checkAttributeExistence(std::string const& attributePath) const;

	std::string getAttribute(std::string const& attributeName) const;
	unsigned int getGroupNumObjects(std::string const& path) const;

	std::string getObjectName(std::string const& path, unsigned int id) const;

	template< typename T>
	void readDataset(std::string const& path, dlib::matrix<T, 0, 1>& data) const;
	template <typename T>
	void readDataset(std::string const& path, dlib::matrix<T, 0, 0, dlib::default_memory_manager, dlib::row_major_layout>& data) const;

private:

	hid_t m_nFileId;

};

#include "HDF5Interface.inl"

#endif /* HDF5INTERFACE_HPP_ */
