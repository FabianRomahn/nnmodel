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

// MinMaxScaler

MinMaxScaler::MinMaxScaler(Eigen::VectorXd const& scale, Eigen::VectorXd const& min) :
		Scaler(), m_scale(scale), m_min(min)
{

}

Eigen::VectorXd MinMaxScaler::scale(Eigen::VectorXd const& input) const
{
	return (m_scale.array() * input.array()) + m_min.array();
}


Eigen::VectorXd MinMaxScaler::scaleInverse(Eigen::VectorXd const& input) const
{
	return (input.array() - m_min.array()) / m_scale.array();
}

Eigen::VectorXd MinMaxScaler::getScaleDerivative() const
{
	return m_scale;
}

Eigen::VectorXd MinMaxScaler::getScaleInverseDerivative() const
{
	return 1.0 / m_scale.array();
}

// StandardScaler

StandardScaler::StandardScaler(Eigen::VectorXd const& scale, Eigen::VectorXd const& mean) :
		Scaler(), m_scale(scale), m_mean(mean)
{

}

Eigen::VectorXd StandardScaler::scale(Eigen::VectorXd const& input) const
{
	return ((input.array() - m_mean.array()) / m_scale.array());
}

Eigen::VectorXd StandardScaler::scaleInverse(Eigen::VectorXd const& input) const
{
	return (input.array() * m_scale.array() + m_mean.array());
}

Eigen::VectorXd StandardScaler::getScaleDerivative() const
{
	return 1.0 / m_scale.array();
}

Eigen::VectorXd StandardScaler::getScaleInverseDerivative() const
{
	return m_scale;
}
