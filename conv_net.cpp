#include "conv_net.hpp"

ConvNet::ConvNet(const NetworkLayout& layout, float kernel_bound, float weight_bound, float bias_bound)
	:structure(layout), previous_input(layout.in_x, layout.in_y, 1)
{
	// build conv_layer/pool_layer structure
	conv_layers.emplace_back(structure.in_x, structure.in_y, 1,
				structure.kernels[0], structure.kernel_dims[0],
				structure.paddings[0], structure.strides[0]);

	pool_layers.emplace_back(conv_layers[0], structure.pool_params[0]);

	for (std::size_t i=1; i < structure.conv_layers; ++i)
	{
		conv_layers.emplace_back(pool_layers[i-1].width(),
					pool_layers[i-1].height(), pool_layers[i-1].layers(),
					structure.kernels[i], structure.kernel_dims[i],
					structure.paddings[i], structure.strides[i]);

		pool_layers.emplace_back(conv_layers[i], structure.pool_params[i]);
	}

	// TODO: remove the assumption that the last pool layer has 1 width and 1 height
	neuron_layers.emplace_back(conv_layers[conv_layers.size() - 1].layers(),
				structure.neurons[0]);

	// build neuron_layer structure
	for (std::size_t n=1; n < structure.neuron_layers; ++n)
	{
		neuron_layers.emplace_back(neuron_layers[n-1].get_outputs(),
					structure.neurons[n]);
	}

	// randomly initialise the kernels, weights and biases
	for (auto& cl : conv_layers) { cl.random_init_kernels(kernel_bound); }
	for (auto& nl : neuron_layers) { nl.random_init_params(weight_bound, bias_bound); }
}

void ConvNet::compute(const FeatureMap &input_map)
{
	previous_input = input_map;

	conv_layers[0].compute(input_map);
	pool_layers[0].compute(conv_layers[0].get_output());

	for (std::size_t i=1; i < conv_layers.size(); ++i)
	{
		conv_layers[i].compute(pool_layers[i-1].get_output());
		pool_layers[i].compute(conv_layers[i].get_output());
	}

	neuron_layers[0].compute(pool_layers[pool_layers.size() - 1].get_output());

	for (std::size_t n=1; n < neuron_layers.size(); ++n)
	{
		neuron_layers[n].compute(neuron_layers[n-1].get_output());
	}
}

void ConvNet::update_backprop(const std::vector<float> &deltas)
{
	// Update the neuron layers
	// update the last layer from the previous neuron layer and output deltas
	if (neuron_layers.size() >= 2) {
		neuron_layers[neuron_layers.size() - 1].update_backprop(
			neuron_layers[neuron_layers.size() - 2].get_output(), deltas);
	}

	// update the middle layers from the sandwiching layers
	if (neuron_layers.size() >= 3) {
		for (std::size_t n = neuron_layers.size() - 2; n >= 1; --n)
		{
			neuron_layers[n].update_backprop(
					neuron_layers[n-1].get_output(), neuron_layers[n+1].get_backprop_deltas());
		}
	}

	// update the first
	if (neuron_layers.size() >= 2) {
		neuron_layers[0].update_backprop(conv_layers[conv_layers.size()-1].get_output().get_data(),
				neuron_layers[1].get_backprop_deltas());
	}

	if (neuron_layers.size() == 1) {
		neuron_layers[0].update_backprop(conv_layers[conv_layers.size()-1].get_output().get_data(), deltas);
	}

	// Update the convolution layers (through the pool layers)

	// update the last conv_layer/pool_layer pair from the first neuron_layer
	// and the previous pool_layer
	if (conv_layers.size() >= 2) {
		pool_layers[pool_layers.size() - 1].update_backprop(
				conv_layers[conv_layers.size() - 1].get_output(), neuron_layers[0].get_backprop_deltas());
		conv_layers[conv_layers.size() - 1].update_backprop(
				conv_layers[conv_layers.size() - 2].get_output(),
				pool_layers[pool_layers.size() - 1].get_backprop_deltas());
	}

	// update the middle layers from the sandwiching layers
	if (conv_layers.size() >= 3) {
		for (std::size_t i = conv_layers.size() - 2; i >= 1; --i)
		{
			pool_layers[i].update_backprop(
					conv_layers[i].get_output(),
					conv_layers[i+1].get_backprop_deltas());

			conv_layers[i].update_backprop(pool_layers[i-1].get_output(),
					pool_layers[i].get_backprop_deltas());
		}
	}

	// update the first conv_layer/pool_layer pair
	if (conv_layers.size() >= 2) {
		pool_layers[0].update_backprop(conv_layers[0].get_output(),
				conv_layers[1].get_backprop_deltas());

		conv_layers[0].update_backprop(previous_input, pool_layers[0].get_backprop_deltas());
	}

	if (conv_layers.size() == 1)
	{
		pool_layers[0].update_backprop(conv_layers[0].get_output(),
				neuron_layers[0].get_backprop_deltas());
		conv_layers[0].update_backprop(previous_input, pool_layers[0].get_backprop_deltas());
	}
}

void ConvNet::write_changes(float grad_mul)
{
	for (auto& cl : conv_layers) { cl.update_params(grad_mul); }
	for (auto& nl : neuron_layers) { nl.update_params(grad_mul); }
}
