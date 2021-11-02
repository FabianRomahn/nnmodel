/*
 * NNModelScaler.hpp
 *
 *  Created on: May 2, 2019
 *      Author: roma_fa
 */

#ifndef NNMODELSCALER_HPP_
#define NNMODELSCALER_HPP_

#include <memory>

#include <eigen3/Eigen/Core>

namespace NeuralNetwork
{

class Scaler;
typedef std::shared_ptr< Scaler > ScalerSPtr;
typedef std::shared_ptr< const Scaler > ScalerCSPtr;

class Scaler
{

public:

	Scaler();
	virtual ~Scaler();

	virtual Eigen::VectorXd scale(Eigen::VectorXd const& input) const = 0;
	virtual Eigen::VectorXd scaleInverse(Eigen::VectorXd const& input) const = 0;

	virtual Eigen::VectorXd getScaleDerivative() const = 0;
	virtual Eigen::VectorXd getScaleInverseDerivative() const = 0;

};

class MinMaxScaler;
typedef std::shared_ptr< MinMaxScaler > MinMaxScalerSPtr;
typedef std::shared_ptr< const MinMaxScaler > MinMaxScalerCSPtr;

class MinMaxScaler : public Scaler
{

public:

	MinMaxScaler(Eigen::VectorXd const& scale, Eigen::VectorXd const& min);

	Eigen::VectorXd scale(Eigen::VectorXd const& input) const;
	Eigen::VectorXd scaleInverse(Eigen::VectorXd const& input) const;

	Eigen::VectorXd getScaleDerivative() const;
	Eigen::VectorXd getScaleInverseDerivative() const;

private:

	Eigen::VectorXd m_scale;
	Eigen::VectorXd m_min;

};

class StandardScaler : public Scaler
{

public:

	StandardScaler(Eigen::VectorXd const& scale, Eigen::VectorXd const& mean);

	Eigen::VectorXd scale(Eigen::VectorXd const& input) const;
	Eigen::VectorXd scaleInverse(Eigen::VectorXd const& input) const;

	Eigen::VectorXd getScaleDerivative() const;
	Eigen::VectorXd getScaleInverseDerivative() const;

private:

	Eigen::VectorXd m_scale;
	Eigen::VectorXd m_mean;

};

}

#endif /* NNMODELSCALER_HPP_ */
