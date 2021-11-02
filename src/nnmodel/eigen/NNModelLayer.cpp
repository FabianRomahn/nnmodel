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

DenseLayer::DenseLayer(Eigen::Matrix<double, Eigen::Dynamic, 1> const& bias,
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const& kernel,
		std::string const& sActivationFunction) :
	m_bias(bias.cast<double>()), m_kernel(kernel.cast<double>())
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
	else if ((sActivationFunction == "tanH") || (sActivationFunction == "tanh"))
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
	return m_kernel.rows();
}

unsigned int DenseLayer::getOutputDimension() const
{
	return m_kernel.cols();
}

void DenseLayer::process(Eigen::VectorXd const& input, Eigen::VectorXd& output) const
{
	// Matrix multiplication
	output = (m_kernel.transpose() * input);
	// Add the bias
	output += m_bias;
	// Activation function
	output = output.unaryExpr(m_ActivationFunction);
}

void DenseLayer::process(Eigen::VectorXd const& input, Eigen::VectorXd& output,
		Eigen::MatrixXd& der) const
{
	// Matrix multiplication
	output = (m_kernel.transpose() * input);
	// Add the bias
	output += m_bias;
	// Activation function
	output = output.unaryExpr(m_ActivationFunction);

	// compute the output
	process(input, output);

	// compute the derivatives
	for (unsigned int i = 0; i < der.rows(); i++)
	{
		der.row(i) = output;
		der.row(i) = der.row(i).unaryExpr(m_ActivationFunctionDerivative);
		der.row(i).array() *= m_kernel.row(i).array();
	}
}

void DenseLayer::process(Eigen::VectorXd const& input, Eigen::VectorXd& output,
		Eigen::MatrixXd const& der_input, Eigen::MatrixXd& der_output) const
{
	// compute the output
	process(input, output);

	// compute the derivatives
	for (unsigned int i = 0; i < der_output.rows(); i++)
	{
		der_output.row(i) = output;
		der_output.row(i) = der_output.row(i).unaryExpr(m_ActivationFunctionDerivative);
	}
	der_output.array() *= (der_input * m_kernel).array();
}
