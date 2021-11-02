/*
 * NNModel.cpp
 *
 *  Created on: Apr 24, 2019
 *      Author: roma_fa
 */

#include "NNModel.hpp"
#include "NNModelLayer.hpp"
#include "NNModelScaler.hpp"
#include "aux/HDF5Interface.hpp"

#include <boost/algorithm/string.hpp>

#include <iostream>
#include <string>
#include <algorithm>

using namespace NeuralNetwork;

Model::Model()
{

}

Model::Model(std::string const& fileName)
{
	loadH5File(fileName);
}

void Model::loadH5File(std::string const& fileName)
{
	HDF5InterfaceSPtr hdf5Reader (new HDF5Interface(fileName));

	// Get the model configuration
	std::string modelConfiguration = hdf5Reader->getAttribute("model_config");
	// Get the activation functions
	std::vector<std::string> vsActivationFunctions = getActivationFunctions(modelConfiguration);

	// Get the number of layers
	unsigned int uNumLayers = hdf5Reader->getGroupNumObjects("/model_weights");

	// There should be an activation function for each layer
	assert (uNumLayers == vsActivationFunctions.size());

	// Load the scaling if it is specified
	if (hdf5Reader->checkExistence("/data_scaling"))
	{
		// Read the attributes for the scaling
		std::string sInputScaling = hdf5Reader->getAttribute("/data_scaling/input_scaling");
		std::string sOutputScaling = hdf5Reader->getAttribute("/data_scaling/output_scaling");
		// Transform the strings to lowercase
		std::transform(sInputScaling.begin(), sInputScaling.end(), sInputScaling.begin(), ::tolower);
		std::transform(sOutputScaling.begin(), sOutputScaling.end(), sOutputScaling.begin(), ::tolower);
		bool bInputScaling = false;
		if (sInputScaling == "true")
		{
			bInputScaling = true;
		}
		bool bOutputScaling = false;
		if (sOutputScaling == "true")
		{
			bOutputScaling = true;
		}
		if (bInputScaling)
		{
			// Create the input scaler
			m_pInputScaler = loadScaler(hdf5Reader, "/data_scaling/input_scaler");
		}
		if (bOutputScaling)
		{
			// Create the output scaler
			m_pOutputScaler = loadScaler(hdf5Reader, "/data_scaling/output_scaler");
		}
	}

	for (unsigned int i = 0; i < uNumLayers; i++)
	{
		std::string objectName = hdf5Reader->getObjectName("/model_weights", i);
		// Only dense layers are supported at the moment

		// Construct the path of the bias and kernels
		std::string bias_path = "/model_weights/" + objectName + "/" + objectName + "/bias:0";
		std::string kernel_path = "/model_weights/" + objectName + "/" + objectName + "/kernel:0";

		// Read the bias
		Eigen::Matrix<double, Eigen::Dynamic, 1> bias;
		hdf5Reader->readDataset(bias_path, bias);

		// Read the kernel
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> kernel;
		hdf5Reader->readDataset(kernel_path, kernel);

		// Create the layer
		m_vLayers.push_back(DenseLayerCSPtr(new DenseLayer(bias, kernel, vsActivationFunctions[i])));
	}

	// Get the parameter description if there is any
	if (hdf5Reader->checkAttributeExistence("/metadata/description_input_parameters"))
	{
		// Split the string into the different elements
		std::string sInputParameterDescription = hdf5Reader->getAttribute("/metadata/description_input_parameters");
		std::vector<std::string> vsInputParameterDescription;
		boost::split(vsInputParameterDescription, sInputParameterDescription, [](char c){return ((c == ' ') || (c == ','));});
		m_vsInputParameterDescription.clear();
		for (unsigned int i = 0; i < vsInputParameterDescription.size(); i++)
		{
			if (vsInputParameterDescription[i].size() > 0)
			{
				m_vsInputParameterDescription.push_back(vsInputParameterDescription[i]);
			}
		}
	}

	// Get the wavelength grid if it is defined
	if (hdf5Reader->checkExistence("/wavelengths"))
	{
		// Read the wavelength grid
		Eigen::Matrix<float, Eigen::Dynamic, 1> outputGrid;
		hdf5Reader->readDataset("/wavelengths", outputGrid);

		// Copy the values to the output grid variable
		m_vdOutputGrid.clear();
		for (unsigned int i = 0; i < outputGrid.size(); i++)
		{
			m_vdOutputGrid.push_back(outputGrid[i]);
		}
	}

	// close the file
	hdf5Reader->closeFile();

	std::vector<unsigned int> vuLayerInputDimensions;
	for (unsigned int i = 0; i < m_vLayers.size(); i++)
	{
		vuLayerInputDimensions.push_back(m_vLayers[i]->getInputDimension());
	}

	// clear the workspaces
	m_workspaces.clear();

	// create the workspaces
	unsigned int uNumThreads = 1;
	#ifdef _OPENMP
	uNumThreads = omp_get_max_threads();
	#endif

	m_workspaces.reserve(uNumThreads);
	for (unsigned int i = 0; i < uNumThreads; i++)
	{
		m_workspaces.emplace_back(vuLayerInputDimensions);
	}
}

std::vector<std::string> Model::getActivationFunctions(std::string const& modelConfiguration)
{
	std::vector<std::string> vsActivationFunctions;
	int iActivationIdx = modelConfiguration.find("activation");
	while (iActivationIdx >= 0)
	{
		int iStartIdx = modelConfiguration.find(":", iActivationIdx);
		iStartIdx = modelConfiguration.find("\"", iStartIdx + 1) + 1;
		int iEndIdx = modelConfiguration.find("\"", iStartIdx) - 1;
		std::string sActivationFunction = modelConfiguration.substr(iStartIdx, iEndIdx - iStartIdx + 1);
		vsActivationFunctions.push_back(sActivationFunction);
		iActivationIdx = modelConfiguration.find("activation", iEndIdx);
	}
	return vsActivationFunctions;
}

ScalerCSPtr Model::loadScaler(HDF5InterfaceCSPtr const& hdf5Reader, std::string const& scalerPath)
{
	std::string typePath = scalerPath + "/type";
	std::string type = hdf5Reader->getAttribute(typePath.c_str());

	if (type == "MinMaxScaler")
	{
		// Define the path of the scale and the min variables
		std::string scalePath = scalerPath + "/scale";
		std::string minPath = scalerPath + "/min";

		// Read the scale
		Eigen::VectorXd scale;
		hdf5Reader->readDataset(scalePath, scale);

		// Read the min
		Eigen::VectorXd min;
		hdf5Reader->readDataset(minPath, min);

		// Create the scaler
		return ScalerCSPtr(new MinMaxScaler(scale, min));
	}
	else if (type == "StandardScaler")
	{
		// Define the path of the scale and the min variables
		std::string scalePath = scalerPath + "/scale";
		std::string meanPath = scalerPath + "/mean";

		// Read the scale
		Eigen::VectorXd scale;
		hdf5Reader->readDataset(scalePath, scale);

		// Read the min
		Eigen::VectorXd mean;
		hdf5Reader->readDataset(meanPath, mean);

		// Create the scaler
		return ScalerCSPtr(new StandardScaler(scale, mean));
	}
	else
	{
		throw std::invalid_argument("Unknown scaler type: " + type);
	}
}

void Model::predict(Eigen::VectorXd const& input, Eigen::VectorXd& output) const
{
	unsigned int uThreadNum = 0;
	#ifdef _OPENMP
	uThreadNum = omp_get_thread_num();
	#endif
	Model::Workspace& workspace = m_workspaces[uThreadNum];

	for (unsigned int i = 0; i < m_vLayers.size(); i++)
	{
		// Input layer
		if (i == 0)
		{
			// If the input is also the output layer
			if (i == m_vLayers.size() - 1)
			{
				// Check if the input has to be scaled
				if (m_pInputScaler)
				{
					m_vLayers[0]->process(m_pInputScaler->scale(input), output);
				}
				else
				{
					m_vLayers[0]->process(input, output);
				}
			}
			else
			{
				Eigen::VectorXd& outputTmp = workspace.m_vIntermediateResults[0];
				// Check if the input has to be scaled
				if (m_pInputScaler)
				{
					m_vLayers[0]->process(m_pInputScaler->scale(input), outputTmp);
				}
				else
				{
					m_vLayers[0]->process(input, outputTmp);
				}
			}
		}
		// Output layer
		else if (i == m_vLayers.size() - 1)
		{
			Eigen::VectorXd& inputTmp = workspace.m_vIntermediateResults[i-1];
			m_vLayers[i]->process(inputTmp, output);
		}
		// Hidden layer
		else
		{
			Eigen::VectorXd& inputTmp = workspace.m_vIntermediateResults[i-1];
			Eigen::VectorXd& outputTmp = workspace.m_vIntermediateResults[i];
			m_vLayers[i]->process(inputTmp, outputTmp);
		}
	}

	// Check if the output has to be scaled (inversely)
	if (m_pOutputScaler)
	{
		output = m_pOutputScaler->scaleInverse(output);
	}
}

void Model::predict(Eigen::VectorXd const& input, Eigen::VectorXd& output, Eigen::MatrixXd& der) const
{
	unsigned int uThreadNum = 0;
	#ifdef _OPENMP
	uThreadNum = omp_get_thread_num();
	#endif
	Model::Workspace& workspace = m_workspaces[uThreadNum];

	for (unsigned int i = 0; i < m_vLayers.size(); i++)
	{
		// Input layer
		if (i == 0)
		{
			// If the input is also the output layer
			if (i == m_vLayers.size() - 1)
			{
				if (m_pInputScaler)
				{
					// Scale the inputs
					m_vLayers[0]->process(m_pInputScaler->scale(input), output, der);
				}
				else
				{
					m_vLayers[0]->process(input, output, der);
				}
			}
			else
			{
				Eigen::VectorXd& outputTmp = workspace.m_vIntermediateResults[0];
				Eigen::MatrixXd& derTmp = workspace.m_vIntermediateDerivatives[0];
				if (m_pInputScaler)
				{
					// Scale the inputs
					m_vLayers[0]->process(m_pInputScaler->scale(input), outputTmp, derTmp);
				}
				else
				{
					m_vLayers[0]->process(input, outputTmp, derTmp);
				}
			}
		}
		// Output layer
		else if (i == m_vLayers.size() - 1)
		{
			Eigen::VectorXd& inputTmp = workspace.m_vIntermediateResults[i-1];
			Eigen::MatrixXd& der_inputTmp = workspace.m_vIntermediateDerivatives[i-1];
			m_vLayers[i]->process(inputTmp, output, der_inputTmp, der);
		}
		// Hidden layer
		else
		{
			Eigen::VectorXd& inputTmp = workspace.m_vIntermediateResults[i-1];
			Eigen::VectorXd& outputTmp = workspace.m_vIntermediateResults[i];
			Eigen::MatrixXd& der_inputTmp = workspace.m_vIntermediateDerivatives[i-1];
			Eigen::MatrixXd& der_outputTmp = workspace.m_vIntermediateDerivatives[i];
			m_vLayers[i]->process(inputTmp, outputTmp, der_inputTmp, der_outputTmp);
		}
	}

	if (m_pInputScaler)
	{
		// Scale the derivatives
		for (unsigned int i = 0; i < der.rows(); i++)
		{
			der.row(i) *= m_pInputScaler->getScaleDerivative()(i);
		}
	}

	if (m_pOutputScaler)
	{
		// Scale the outputs (inversely)
		output = m_pOutputScaler->scaleInverse(output);
		// Scale the derivatives (inversely)
		for (unsigned int i = 0; i < der.rows(); i++)
		{
			der.row(i).array() *= m_pOutputScaler->getScaleInverseDerivative().array();
		}
	}
}

unsigned int Model::getInputDimension() const
{
	return m_vLayers[0]->getInputDimension();
}

unsigned int Model::getOutputDimension() const
{
	return m_vLayers[m_vLayers.size() - 1]->getOutputDimension();
}

std::vector<std::string> Model::getInputParameterDescription() const
{
	return m_vsInputParameterDescription;
}

std::vector<double> Model::getOutputGrid() const
{
	return m_vdOutputGrid;
}

Model::Workspace::Workspace(std::vector<unsigned int> const& vuDimensions)
{
	for (unsigned int i = 1; i < vuDimensions.size(); i++)
	{
		m_vIntermediateResults.emplace_back(Eigen::VectorXd(vuDimensions[i]));
	}
	for (unsigned int i = 1; i < vuDimensions.size(); i++)
	{
		m_vIntermediateDerivatives.emplace_back(Eigen::MatrixXd(vuDimensions[0], vuDimensions[i]));
	}
}

Model::Workspace::Workspace(Workspace&& workspace) :
		m_vIntermediateResults(std::move(workspace.m_vIntermediateResults))
{

}
