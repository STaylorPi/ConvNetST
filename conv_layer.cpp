#include "conv_layer.hpp"

#include "random/random.hpp"

ConvLayer::ConvLayer(std::size_t width, std::size_t height, std::size_t inputs, std::size_t outputs, std::size_t kernel_dim, std::size_t padding_, std::size_t stride_)
	:output(get_new_dim(width, kernel_dim, padding_, stride_),
			get_new_dim(height, kernel_dim, padding_, stride_), outputs),
	kernels(outputs, ConvKernel(kernel_dim)), input_dim{ inputs },
	output_dim{ outputs }, kernel_size{ kernel_dim }, stride{ stride_ }, padding{ padding_ }
{
}

void ConvLayer::compute(const FeatureMap& input_map)
{
	// each layer of the output is the result of one kernel convolving over the whole of the input
	for (std::size_t i = 0; i < output_dim; ++i)
	{
		convolve_to(input_map, output, kernels[i], padding, stride, i);
	}

	output.normalise_ReLU();
}

void ConvLayer::random_init_kernels(float max_bound)
{
	for (ConvKernel& kernel : kernels)
	{
		for (float& data : kernel.view())
		{
			data = Rand::random_bound(max_bound);
		}
	}
}
