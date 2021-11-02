/*
 * NNModelScaler.cpp
 *
 *  Created on: May 2, 2019
 *      Author: roma_fa
 */

#include "NNModelScaler.hpp"

#include <iostream>

using namespace NeuralNetwork;

Scaler::Scaler()
{

}

Scaler::~Scaler()
{

}

MinMaxScaler::MinMaxScaler(dlib::matrix<double, 0, 1> const& scale, dlib::matrix<double, 0, 1> const& min) :
		Scaler(), m_scale(scale), m_min(min)
{

}

dlib::matrix<double, 0, 1> MinMaxScaler::scale(dlib::matrix<double, 0, 1> const& input) const
{
	return dlib::pointwise_multiply(m_scale, input) + m_min;
}


dlib::matrix<double, 0, 1> MinMaxScaler::scaleInverse(dlib::matrix<double, 0, 1> const& input) const
{
	return dlib::pointwise_multiply((input - m_min), 1.0 / m_scale);
}

dlib::matrix<double, 0, 1> MinMaxScaler::getScaleDerivative() const
{
	return m_scale;
}

dlib::matrix<double, 0, 1> MinMaxScaler::getScaleInverseDerivative() const
{
	return 1.0 / m_scale;
}
