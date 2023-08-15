#include "neuron_layer.hpp"

#include <algorithm>

#include "random/random.hpp"
#include "conv_layer.hpp"

NeuronLayer::NeuronLayer(std::size_t input_dim_, std::size_t output_dim_)
	:input_dim( input_dim_ ), output_dim( output_dim_ ),
	weights(input_dim * output_dim), biases(output_dim),
	weight_deltas(input_dim * output_dim), bias_deltas(output_dim), input_deltas(input_dim),
	output_layer(output_dim)
{
}

void NeuronLayer::random_init_params(float w_range, float b_range)
{
	for (auto& w : weights) w = Rand::random() * w_range;
	for (auto& b : biases) b = Rand::random() * b_range;
}

void NeuronLayer::compute(const FeatureMap& f_map)
{
	compute(f_map.get_data());
}

void NeuronLayer::compute(const std::vector<float> &inputs)
{
	std::fill(output_layer.begin(), output_layer.end(), 0.0f);
	if (inputs.size() != input_dim) return;

	for (std::size_t out_i = 0; out_i < output_layer.size(); ++out_i)
	{
		for (std::size_t in_i = 0; in_i < inputs.size(); ++in_i)
		{
			output_layer[out_i] += weights[out_i * input_dim + in_i] * inputs[in_i];
		}
		output_layer[out_i] += biases[out_i];
	}

	std::for_each(output_layer.begin(), output_layer.end(), ReLU);
}

void NeuronLayer::update_backprop(const std::vector<float> &inputs, const std::vector<float> &output_deltas)
{
	// update the weight_deltas
	for (std::size_t oi=0; oi < output_dim; ++oi)
	{
		for (std::size_t ii=0; ii < input_dim; ++ii)
		{
			// compute d(cost) / d(weight) and add its negative value to the delta
			weight_deltas[oi * input_dim + ii] +=
				inputs[ii] * dReLU(output_layer[oi]) * 2 * output_deltas[oi];
			// exploit the fact that dReLU(ReLU(x)) = dReLU(x)
		}
	}

	// update the bias deltas
	for (std::size_t oi=0; oi < output_dim; ++oi)
	{
		// compute d(cost) / d(bias) and add its negative value to the delta
		bias_deltas[oi] += dReLU(output_layer[oi]) * 2 * output_deltas[oi];
	}

	std::fill(input_deltas.begin(), input_deltas.end(), 0.0f);
	// update the previous layer activation deltas
	for (std::size_t ii=0; ii < input_dim; ++ii)
	{
		for (std::size_t oi=0; oi < output_dim; ++oi)
		{
			input_deltas[ii] +=
				weights[oi * input_dim + ii] * dReLU(output_layer[oi]) * 2 * output_deltas[oi];
		}
	}
}

void NeuronLayer::update_params(float grad_mul)
{
	for (std::size_t w=0; w < input_dim * output_dim; ++w) { weights[w] += grad_mul * weight_deltas[w]; }
	for (std::size_t b=0; b < output_dim; ++b) { biases[b] += grad_mul * bias_deltas[b]; }

	std::fill(weight_deltas.begin(), weight_deltas.end(), 0.0f);
	std::fill(bias_deltas.begin(), bias_deltas.end(), 0.0f);
}
