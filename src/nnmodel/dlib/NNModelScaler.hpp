/*
 * NNModelScaler.hpp
 *
 *  Created on: May 2, 2019
 *      Author: roma_fa
 */

#ifndef NNMODELSCALER_HPP_
#define NNMODELSCALER_HPP_

#include <memory>

#include <dlib/matrix/matrix_la.h>

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

	virtual dlib::matrix<double, 0, 1> scale(dlib::matrix<double, 0, 1> const& input) const = 0;
	virtual dlib::matrix<double, 0, 1> scaleInverse(dlib::matrix<double, 0, 1> const& input) const = 0;

	virtual dlib::matrix<double, 0, 1> getScaleDerivative() const = 0;
	virtual dlib::matrix<double, 0, 1> getScaleInverseDerivative() const = 0;

};

class MinMaxScaler;
typedef std::shared_ptr< MinMaxScaler > MinMaxScalerSPtr;
typedef std::shared_ptr< const MinMaxScaler > MinMaxScalerCSPtr;

class MinMaxScaler : public Scaler
{

public:

	MinMaxScaler(dlib::matrix<double, 0, 1> const& scale, dlib::matrix<double, 0, 1> const& min);

	dlib::matrix<double, 0, 1> scale(dlib::matrix<double, 0, 1> const& input) const;
	dlib::matrix<double, 0, 1> scaleInverse(dlib::matrix<double, 0, 1> const& input) const;

	dlib::matrix<double, 0, 1> getScaleDerivative() const;
	dlib::matrix<double, 0, 1> getScaleInverseDerivative() const;

private:

	dlib::matrix<double, 0, 1> m_scale;
	dlib::matrix<double, 0, 1> m_min;

};


}

#endif /* NNMODELSCALER_HPP_ */
