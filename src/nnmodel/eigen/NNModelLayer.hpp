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

#include <eigen3/Eigen/Core>

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

	virtual void process(Eigen::VectorXd const& input, Eigen::VectorXd& output) const = 0;
	virtual void process(Eigen::VectorXd const& input, Eigen::VectorXd& output, Eigen::MatrixXd& der) const = 0;
	virtual void process(Eigen::VectorXd const& input, Eigen::VectorXd& output, Eigen::MatrixXd const& der_input, Eigen::MatrixXd& der_output) const = 0;

	virtual unsigned int getInputDimension() const = 0;
	virtual unsigned int getOutputDimension() const = 0;

};

class DenseLayer;
typedef std::shared_ptr< DenseLayer > DenseLayerSPtr;
typedef std::shared_ptr< const DenseLayer > DenseLayerCSPtr;

class DenseLayer : public ModelLayer
{

public:

	DenseLayer(Eigen::Matrix<double, Eigen::Dynamic, 1> const& bias,
			Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> const& kernel,
			std::string const& sActivationFunction);

	unsigned int getInputDimension() const;
	unsigned int getOutputDimension() const;

	void process(Eigen::VectorXd const& input, Eigen::VectorXd& output) const;
	void process(Eigen::VectorXd const& input, Eigen::VectorXd& output, Eigen::MatrixXd& der) const;
	void process(Eigen::VectorXd const& input, Eigen::VectorXd& output, Eigen::MatrixXd const& der_input, Eigen::MatrixXd& der_output) const;

private:

	Eigen::VectorXd m_bias;
	Eigen::MatrixXd m_kernel;

	ActivationFunction m_ActivationFunction;
	ActivationFunctionDerivative m_ActivationFunctionDerivative;

};

} // namespace NeuralNetwork

#endif /* SRC_NNMODELLAYER_HPP_ */
