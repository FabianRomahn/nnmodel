/*
 * NNModel.hpp
 *
 *  Created on: Apr 24, 2019
 *      Author: roma_fa
 */

#ifndef SRC_NNMODEL_HPP_
#define SRC_NNMODEL_HPP_

#include <vector>
#include <string>
#include <memory>

#include <dlib/matrix/matrix_la.h>

class HDF5Interface;
typedef std::shared_ptr< HDF5Interface > HDF5InterfaceSPtr;
typedef std::shared_ptr< const HDF5Interface > HDF5InterfaceCSPtr;

namespace NeuralNetwork
{

class ModelLayer;
typedef std::shared_ptr< ModelLayer > ModelLayerSPtr;
typedef std::shared_ptr< const ModelLayer > ModelLayerCSPtr;

class Scaler;
typedef std::shared_ptr< Scaler > ScalerSPtr;
typedef std::shared_ptr< const Scaler > ScalerCSPtr;

class Model;
typedef std::shared_ptr< Model> ModelSPtr;
typedef std::shared_ptr< const Model> ModelCSPtr;

class Model
{

public:

	Model();
	Model(std::string const& fileName);

	void loadH5File(std::string const& fileName);

	void predict(dlib::matrix<double, 0, 1> const& input, dlib::matrix<double, 0, 1>& output) const;
	void predict(dlib::matrix<double, 0, 1> const& input, dlib::matrix<double, 0, 1>& output, dlib::matrix<double>& der) const;

	unsigned int getInputDimension() const;
	unsigned int getOutputDimension() const;

	std::vector<std::string> getInputParameterDescription() const;
	std::vector<double> getOutputGrid() const;

private:

	std::vector<ModelLayerCSPtr> m_vLayers;

	ScalerCSPtr m_pInputScaler;
	ScalerCSPtr m_pOutputScaler;

	std::vector<std::string> m_vsInputParameterDescription;
	std::vector<double> m_vdOutputGrid;

	std::vector<std::string> getActivationFunctions(std::string const& modelConfiguration);
	ScalerCSPtr loadScaler(HDF5InterfaceCSPtr const& hdf5Reader, std::string const& scalerPath);

	struct Workspace
	{

		Workspace(std::vector<unsigned int> const& vuDimensions);
		Workspace(Workspace&& workspace);

		std::vector<dlib::matrix<double, 0, 1> > m_vIntermediateResults;
		std::vector<dlib::matrix<double> > m_vIntermediateDerivatives;

	};

	mutable std::vector<Workspace> m_workspaces;
};

}

#endif /* SRC_NNMODEL_HPP_ */
