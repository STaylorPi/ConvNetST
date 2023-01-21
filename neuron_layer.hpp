#pragma once

#include "feature_map.hpp"


class NeuronLayer
{
public:
	NeuronLayer(std::size_t input_dim, std::size_t output_dim);

	void random_init_params(float w_range, float b_range);

	void compute(const std::vector<float>& inputs);

private:
	// number of input node (how many weights per node)
	std::size_t input_dim;
	std::size_t output_dim;

	std::vector<float> weights; // dimension input_dim * output_dim
	std::vector<float> biases; // dimension output_dim

	// node outputs
	std::vector<float> output_layer;
};

inline void ReLU(float& value)
{
	value = std::max(0.0f, value);
}
