/*
 * NNModelActivationFunctions.hpp
 *
 *  Created on: Apr 24, 2019
 *      Author: roma_fa
 */

#ifndef SRC_NNMODELACTIVATIONFUNCTIONS_HPP_
#define SRC_NNMODELACTIVATIONFUNCTIONS_HPP_

#include <functional>

namespace NeuralNetwork
{

typedef std::function<double(double)> ActivationFunction;
typedef std::function<double(double)> ActivationFunctionDerivative;

// Linear activation function
extern ActivationFunction linear;
extern ActivationFunctionDerivative linearDer;

// ReLU activation function
extern ActivationFunction reLU;
extern ActivationFunctionDerivative reLUDer;

// Sigmoid (logistic) activation function
extern ActivationFunction sigmoid;
extern ActivationFunctionDerivative sigmoidDer;

// TanH activation function
extern ActivationFunction tanH;
extern ActivationFunctionDerivative tanHDer;

}

#endif /* SRC_NNMODELACTIVATIONFUNCTIONS_HPP_ */
