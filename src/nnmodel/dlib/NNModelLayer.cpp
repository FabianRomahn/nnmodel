/*
 * NNModelLayer.cpp
 *
 *  Created on: Apr 24, 2019
 *      Author: roma_fa
 */

#include "NNModelLayer.hpp"

#include <iostream>

using namespace NeuralNetwork;

ModelLayer::ModelLayer()
{

}

ModelLayer::~ModelLayer()
{

}


DenseLayer::DenseLayer(dlib::matrix<float, 0, 1> const& bias,
		dlib::matrix<float, 0, 0, dlib::default_memory_manager, dlib::row_major_layout> const& kernel,
		std::string const& sActivationFunction) :
	m_bias(dlib::matrix_cast<double>(bias)), m_kernel(dlib::matrix_cast<double>(kernel))
{
	// determine the activation function
	if (sActivationFunction == "relu")
	{
		m_ActivationFunction = reLU;
		m_ActivationFunctionDerivative = reLUDer;
	}
	else if (sActivationFunction == "linear")
	{
		m_ActivationFunction = linear;
		m_ActivationFunctionDerivative = linearDer;
	}
	else if (sActivationFunction == "sigmoid")
	{
		m_ActivationFunction = sigmoid;
		m_ActivationFunctionDerivative = sigmoidDer;
	}
	else if (sActivationFunction == "tanH")
	{
		m_ActivationFunction = tanH;
		m_ActivationFunctionDerivative = tanHDer;
	}
	else
	{
		throw std::invalid_argument("Unknown activation function: " + sActivationFunction);
	}
}

unsigned int DenseLayer::getInputDimension() const
{
	return m_kernel.nr();
}

unsigned int DenseLayer::getOutputDimension() const
{
	return m_kernel.nc();
}

void DenseLayer::process(dlib::matrix<double, 0, 1> const& input, dlib::matrix<double, 0, 1>& output) const
{
	// Matrix multiplication
	output = (dlib::trans(m_kernel) * input);
	// Add the bias
	output += m_bias;
	// Activation function - TODO
	for (unsigned int i = 0; i < output.nr(); i++)
	{
		output(i) = m_ActivationFunction(output(i));
	}
}

void DenseLayer::process(dlib::matrix<double, 0, 1> const& input, dlib::matrix<double, 0, 1>& output,
		dlib::matrix<double>& der) const
{
	// Matrix multiplication
	output = (dlib::trans(m_kernel) * input);
	// Add the bias
	output += m_bias;
	// Activation function - TODO
	for (unsigned int i = 0; i < output.nr(); i++)
	{
		output(i) = m_ActivationFunction(output(i));
	}

	// compute the output
	process(input, output);

	// compute the derivatives
	for (unsigned int i = 0; i < der.nr(); i++)
	{
		dlib::set_rowm(der, i) = dlib::trans(output);
		for (unsigned int j = 0; j < der.nc(); j++)
		{
			der(i, j) = m_ActivationFunctionDerivative(der(i, j));
		}
		dlib::set_rowm(der, i) = dlib::pointwise_multiply(dlib::rowm(der, i), dlib::rowm(m_kernel, i));
	}
}

void DenseLayer::process(dlib::matrix<double, 0, 1> const& input, dlib::matrix<double, 0, 1>& output,
		dlib::matrix<double> const& der_input, dlib::matrix<double>& der_output) const
{
	// compute the output
	process(input, output);

	// compute the derivatives
	for (unsigned int i = 0; i < der_output.nr(); i++)
	{
		dlib::set_rowm(der_output, i) = dlib::trans(output);
		for (unsigned int j = 0; j < der_output.nc(); j++)
		{
			der_output(i, j) = m_ActivationFunctionDerivative(der_output(i, j));
		}
	}
	der_output = dlib::pointwise_multiply(der_output, der_input * m_kernel);
}
