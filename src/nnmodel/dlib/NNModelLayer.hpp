/*
 * NNModelLayer.hpp
 *
 *  Created on: Apr 24, 2019
 *      Author: roma_fa
 */

#ifndef SRC_NNMODELLAYER_HPP_
#define SRC_NNMODELLAYER_HPP_

#include "NNModelActivationFunctions.hpp"

#include <vector>
#include <memory>

#include <dlib/matrix/matrix_la.h>

namespace NeuralNetwork
{

class ModelLayer;
typedef std::shared_ptr< ModelLayer > ModelLayerSPtr;
typedef std::shared_ptr< const ModelLayer > ModelLayerCSPtr;

class ModelLayer
{

public:

	ModelLayer();
	virtual  ~ModelLayer();

	virtual void process(dlib::matrix<double, 0, 1> const& input, dlib::matrix<double, 0, 1>& output) const = 0;
	virtual void process(dlib::matrix<double, 0, 1> const& input, dlib::matrix<double, 0, 1>& output, dlib::matrix<double>& der) const = 0;
	virtual void process(dlib::matrix<double, 0, 1> const& input, dlib::matrix<double, 0, 1>& output, dlib::matrix<double> const& der_input, dlib::matrix<double>& der_output) const = 0;

	virtual unsigned int getInputDimension() const = 0;
	virtual unsigned int getOutputDimension() const = 0;

};

class DenseLayer;
typedef std::shared_ptr< DenseLayer > DenseLayerSPtr;
typedef std::shared_ptr< const DenseLayer > DenseLayerCSPtr;

class DenseLayer : public ModelLayer
{

public:

	DenseLayer(dlib::matrix<float, 0, 1> const& bias,
			dlib::matrix<float, 0, 0, dlib::default_memory_manager, dlib::row_major_layout> const& kernel,
			std::string const& sActivationFunction);

	unsigned int getInputDimension() const;
	unsigned int getOutputDimension() const;

	void process(dlib::matrix<double, 0, 1> const& input, dlib::matrix<double, 0, 1>& output) const;
	void process(dlib::matrix<double, 0, 1> const& input, dlib::matrix<double, 0, 1>& output, dlib::matrix<double>& der) const;
	void process(dlib::matrix<double, 0, 1> const& input, dlib::matrix<double, 0, 1>& output, dlib::matrix<double> const& der_input, dlib::matrix<double>& der_output) const;

private:

	dlib::matrix<double, 0, 1> m_bias;
	dlib::matrix<double> m_kernel;

	ActivationFunction m_ActivationFunction;
	ActivationFunctionDerivative m_ActivationFunctionDerivative;

};

} // namespace NeuralNetwork

#endif /* SRC_NNMODELLAYER_HPP_ */
