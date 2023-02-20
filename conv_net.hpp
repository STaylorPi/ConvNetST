#pragma once

#include "conv_layer.hpp"
#include "pool_layer.hpp"
#include "neuron_layer.hpp"

struct NetworkLayout
{
	// all conv_layers are followed by a pool_layer, even if that makes the pool size 1
	std::size_t conv_layers;
	std::size_t neuron_layers;

	// all other parameters can be deduced
	std::size_t in_x, in_y;

	// sizes are all == conv_layers in this block
	std::vector<std::size_t> kernels; // number of output kernels in the i'th layer
	std::vector<std::size_t> kernel_dims;
	std::vector<std::size_t> paddings;
	std::vector<std::size_t> strides;
	std::vector<std::size_t> pool_params;

	// size == neuron_layers
	std::vector<std::size_t> neurons; // neurons in each layer: neurons.size() == neuron_layers
};

class ConvNet
{
public:
	ConvNet(const NetworkLayout& layout, float kernel_bound, float weight_bound, float bias_bound);

	void compute(const FeatureMap& input_map);

	void update_backprop(const std::vector<float>& deltas);
	void write_changes(float grad_mul);

	const auto& get_output() const { return neuron_layers[neuron_layers.size()-1].get_output(); }

private:
	NetworkLayout structure;

	// equal sizes
	std::vector<ConvLayer> conv_layers;
	std::vector<PoolLayer> pool_layers;

	std::vector<NeuronLayer> neuron_layers;

	FeatureMap previous_input;
};
