#include "conv_layer.hpp"

#include "random/random.hpp"
#include "neuron_layer.hpp"

ConvLayer::ConvLayer(const FeatureMap& input_map, std::size_t kernels, std::size_t kernel_dim)
	:output(get_new_dim(input_map.get_width(), kernel_dim, 0, 1),
			get_new_dim(input_map.get_height(), kernel_dim, 0, 1), kernels),
	kernels(kernels, ConvKernel(kernel_dim)),
	input_deltas(input_map.get_width(), input_map.get_height(), input_map.get_layers()),
	input_dim(input_map.get_layers()), output_dim(kernels), kernel_size(kernel_dim),
	stride{ 1 }, padding{ 0 }
{

}

ConvLayer::ConvLayer(std::size_t width, std::size_t height, std::size_t inputs, std::size_t outputs, std::size_t kernel_dim, std::size_t padding_, std::size_t stride_)
	:output(get_new_dim(width, kernel_dim, padding_, stride_),
			get_new_dim(height, kernel_dim, padding_, stride_), outputs),
	kernels(outputs, ConvKernel(kernel_dim)), kernel_deltas(outputs, ConvKernel(kernel_dim)),
	input_deltas(width, height, inputs),
	input_dim{ inputs }, output_dim{ outputs },
	kernel_size{ kernel_dim }, stride{ stride_ }, padding{ padding_ }
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

float ConvLayer::get_weight_delta_at(const FeatureMap& input_map, std::size_t kernel, std::size_t x_off, std::size_t y_off, std::size_t x_out, std::size_t y_out)
{
	if (dReLU(output.get_at(x_out, y_out, kernel)) == 0.0f) return 0.0f;

	float delta = 0;
	for (std::size_t l=0; l < kernels.size(); ++l)
	{
		delta += input_map.get_at(x_out * stride + x_off - padding, y_out * stride + y_off - padding, l);
	}

	return delta;
}

void ConvLayer::update_backprop(const FeatureMap &input_map, const FeatureMap &output_deltas)
{
	for (std::size_t l=0; l < kernels.size(); ++l)
	{
		for (std::size_t out_y=0; out_y < output.get_height(); ++out_y)
		{
			for (std::size_t out_x=0; out_x < output.get_width(); ++out_x)
			{
				for (std::size_t kernel_pos; kernel_pos < pow(kernel_size, 2); ++kernel_pos)
				{
					kernel_deltas[l].view()[kernel_pos] +=
						get_weight_delta_at(input_map, l, kernel_pos % kernel_size, kernel_pos / kernel_size,
								out_x, out_y) * 2 * output_deltas.get_at(out_x, out_y, l);
				}
			}
		}
	}

	// TODO: calculate the input deltas
	input_deltas.fill_with(0.0f);

	// go through each output (width, height)
	// for each compute the dReLU
	// then go through each weight in each kernel, marching along each input layer and adding to deltas
	for (std::size_t out_y=0; out_y < output.get_height(); out_y++)
	{
		for (std::size_t out_x=0; out_x < output.get_width(); out_x++)
		{
			for (std::size_t k=0; k < output_dim; ++k)
			{
				// no point computing the deltas if the derivative is zero anyway
				if (dReLU(output.get_at(out_x, out_x, k)) == 0.0f) continue;

				int64_t x_tl = out_x * stride - padding;
				int64_t y_tl = out_y * stride - padding;
				float delta = output_deltas.get_at(out_x, out_y, k);

				for (std::size_t k_y=0; k_y < kernel_size; ++k_y)
				{
					for (std::size_t k_x=0; k_x < kernel_size; ++k_x)
					{
						int64_t x = x_tl + k_x;
						int64_t y = y_tl + k_y;

						if (x < 0 || y < 0) continue;
						input_deltas.set_at(x, y,
								delta * 2 * kernels[k].view()[k_y * kernel_size + k_x]);
					}
				}
			}
		}
	}
}

void ConvLayer::update_params(float grad_mul)
{
	for (std::size_t kernel=0; kernel < kernels.size(); ++kernel)
	{
		kernels[kernel] += kernel_deltas[kernel] * grad_mul;
	}
}
