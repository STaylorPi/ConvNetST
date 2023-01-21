#include "neuron_layer.hpp"

#include <algorithm>

#include "random/random.hpp"
#include "conv_layer.hpp"

NeuronLayer::NeuronLayer(std::size_t input_dim_, std::size_t output_dim_)
	:input_dim( input_dim_ ), output_dim( output_dim_ ),
	weights(input_dim * output_dim), biases(output_dim),
	output_layer(output_dim)
{
}

void NeuronLayer::random_init_params(float w_range, float b_range)
{
	for (auto& w : weights) w = Rand::random_bound(w_range);
	for (auto& b : biases) b = Rand::random_bound(b_range);
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
