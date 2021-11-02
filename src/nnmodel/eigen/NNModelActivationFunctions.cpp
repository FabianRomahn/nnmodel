/*
 * NNModelActivationFunctions.cpp
 *
 *  Created on: Apr 24, 2019
 *      Author: roma_fa
 */

#include "NNModelActivationFunctions.hpp"
#include <cmath>

//using namespace NeuralNetwork;

NeuralNetwork::ActivationFunction NeuralNetwork::linear = [](double x)
{
	return x;
};

NeuralNetwork::ActivationFunctionDerivative NeuralNetwork::linearDer = [](double )
{
	return 1;
};

NeuralNetwork::ActivationFunction NeuralNetwork::reLU = [](double x)
{
	return std::max(0.0, x);
};

NeuralNetwork::ActivationFunctionDerivative NeuralNetwork::reLUDer = [](double x)
{
	return (x > 0 ? 1.0 : 0.0);
};

NeuralNetwork::ActivationFunction NeuralNetwork::sigmoid = [](double x)
{
	return 1.0 / (1.0 + std::exp(-x));
};

NeuralNetwork::ActivationFunctionDerivative NeuralNetwork::sigmoidDer = [](double x)
{
	// WARNING: It is assumed that x is already the value of activation function!
	return x * (1.0 - x);
};

NeuralNetwork::ActivationFunction NeuralNetwork::tanH = [](double x)
{
	return (2 / (1.0 + std::exp(-2.0 * x))) - 1.0;
};

NeuralNetwork::ActivationFunctionDerivative NeuralNetwork::tanHDer = [](double x)
{
	// WARNING: It is assumed that x is already the value of activation function!
	return (1.0 - x * x);
};
