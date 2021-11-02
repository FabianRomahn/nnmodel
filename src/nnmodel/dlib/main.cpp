/*
 * main.cpp
 *
 *  Created on: Apr 25, 2019
 *      Author: roma_fa
 */

#include "NNModel.hpp"

#include <iostream>
#include <boost/program_options.hpp>

// This function is needed in order to parse also negative values from the command line
// See (https://stackoverflow.com/questions/4107087/accepting-negative-doubles-with-boostprogram-options)
std::vector<boost::program_options::option> ignore_numbers(std::vector<std::string>& args)
{
    std::vector<boost::program_options::option> result;
    int pos = 0;

    while(!args.empty()) {
        const auto& arg = args[0];
        double num;
        try
        {
        	boost::lexical_cast<double>(arg);
            result.push_back(boost::program_options::option());
            boost::program_options::option& opt = result.back();

            opt.position_key = pos++;
            opt.value.push_back(arg);
            opt.original_tokens.push_back(arg);

            args.erase(args.begin());
        }
        catch (boost::bad_lexical_cast& e)
        {
        	break;
        }
    }

    return result;
}

void printInput(dlib::matrix<double, 0, 1> const& input)
{
	std::cout << "input: " << std::endl;
	for (unsigned int i = 0; i < input.size(); i++)
	{
		if (i > 0)
		{
			std::cout << ", ";
		}
		std::cout << input(i);
	}
	std::cout << std::endl;
}

void printOutput(dlib::matrix<double, 0, 1> const& output)
{
	std::cout << "output: " << std::endl;
	for (unsigned int i = 0; i < output.size(); i++)
	{
		if (i > 0)
		{
			std::cout << ", ";
		}
		std::cout << output(i);
	}
	std::cout << std::endl;
}

void printGradient(dlib::matrix<double> const& gradient)
{
	std::cout << "gradient: " << std::endl;
	for (unsigned int i = 0; i < gradient.nr(); i++)
	{
		if (i > 0)
		{
			std::cout << std::endl;
		}
		std::cout << "derivative of outputs w.r.t. to input " << i << ": " << std::endl;
		for (unsigned int j = 0; j < gradient.nc(); j++)
		{
			if (j > 0)
			{
				std::cout << ", ";
			}
			std::cout << gradient(i, j);
		}
		std::cout << std::endl;
	}
}

int main(int argc, char * argv[])
{
	// Default model filename
	std::string modelFilename = "../../aux/clearsky_model.h5";

	// Gradient flag
	bool bGradient = false;

	// Input parameters
	std::vector<float> vfParameters;

	// Benchmark
	unsigned int uBenchmark = 0;

	// Output
	bool bOutput = false;

	// Read the command line with boost program options
	try
	{
		namespace po = boost::program_options;

		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h", "produce help message")
			("model,m", po::value<std::string>(&modelFilename)->default_value(modelFilename), "name of the h5 model file specifiying the neural network")
			("gradient,g", po::bool_switch(&bGradient), "flag that determines if the gradient is calculated")
			("parameters,p", po::value<std::vector<float> >(&vfParameters)->multitoken(), "specify the input parameters for the model")
			("benchmark,b", po::value<unsigned int>(&uBenchmark)->default_value(uBenchmark), "if set to a value higher than 0 - this number of calls to the model is made and the parameters are interpreted as range")
			("output,o", po::bool_switch(&bOutput), "output of the results in benchmark mode")
		;

		po::variables_map vm;
		po::store(po::command_line_parser(argc, argv).extra_style_parser(&ignore_numbers).options(desc).run(), vm);
		po::notify(vm);

		if (vm.count("help"))
		{
		    std::cout << desc << "\n";
		    return 1;
		}

		if (vm.count("model") == 0)
		{
			throw std::invalid_argument("No model specified");
		}

		if (vm.count("parameters") == 0 && (uBenchmark == 0))
		{
			throw std::invalid_argument("Neither input parameters nor benchmark specified");
		}
	}
	catch (std::exception& e)
	{
		std::cerr << "ERROR while reading the command line: " << e.what() << "\n";
		return -1;
	}

	// load the model
	NeuralNetwork::Model model (modelFilename);

	dlib::matrix<double, 0, 1> input (model.getInputDimension());
	dlib::matrix<double, 0, 1> output (model.getOutputDimension());
	dlib::matrix<double> gradient;
	// Resize the gradient matrix if needed
	if (bGradient)
	{
		gradient.set_size(model.getInputDimension(), model.getOutputDimension());
//		gradient.resize(model.getInputDimension(), model.getOutputDimension());
	}

	if (uBenchmark > 0)
	{
		if (vfParameters.size() != 2 * model.getInputDimension())
		{
			throw std::invalid_argument("Invalid number of input parameters - has to be: " + boost::lexical_cast<std::string>(2 * model.getInputDimension()));
		}

		for (unsigned int i = 0; i < uBenchmark; i++)
		{
			// construct the input parameters
			for (unsigned int j = 0; j < input.size(); j++)
			{
				input(j) = vfParameters[2*j] + ((vfParameters[(2*j)+1] - vfParameters[2*j]) / uBenchmark * i);
			}
			if (bOutput)
			{
				if (i > 0)
				{
					std::cout << std::endl;
				}
				std::cout << "Sample " << i << ": " << std::endl << std::endl;
				// print the input
				printInput(input);
			}
			// call the model
			if (bGradient)
			{
				model.predict(input, output, gradient);
			}
			else
			{
				model.predict(input, output);
			}
			if (bOutput)
			{
				std::cout << std::endl;
				// print the output
				printOutput(output);
				if (bGradient)
				{
					std::cout << std::endl;
					printGradient(gradient);
				}
			}
		}
		if (bGradient)
		{
			std::cout << "called the model " << uBenchmark << " times with gradient" << std::endl;
		}
		else
		{
			std::cout << "called the model " << uBenchmark << " times without gradient" << std::endl;
		}
	}
	else
	{
		if (vfParameters.size() != model.getInputDimension())
		{
			throw std::invalid_argument("Invalid number of input parameters - has to be: " + boost::lexical_cast<std::string>(model.getInputDimension()));
		}

		// copy the input parameters
		for (unsigned int i = 0; i < input.size(); i++)
		{
			input(i) = vfParameters[i];
		}

		// print the input
		printInput(input);

		if (bGradient)
		{
			model.predict(input, output, gradient);
		}
		else
		{
			model.predict(input, output);
		}

		// print the output
		printOutput(output);

		// print the gradient
		if (bGradient)
		{
			printGradient(gradient);
		}
	}

	return 0;
}
