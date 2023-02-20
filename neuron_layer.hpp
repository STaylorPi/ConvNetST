#pragma once

#include "feature_map.hpp"


class NeuronLayer
{
public:
	NeuronLayer(std::size_t input_dim, std::size_t output_dim);

	void random_init_params(float w_range, float b_range);

	void compute(const FeatureMap& f_map); // NB: MUST be a 1 * 1 * input_dim feature map
	void compute(const std::vector<float>& inputs);

	void update_backprop(const std::vector<float>& inputs, const std::vector<float>& output_deltas);
	void update_params(float grad_mul); // writes the deltas calculated from the above

	const auto& get_backprop_deltas() const { return input_deltas; }

	const auto& get_output() const { return output_layer; }

	const std::size_t& get_outputs() const { return output_dim; }

private:
	// number of input node (how many weights per node)
	std::size_t input_dim;
	std::size_t output_dim;

	std::vector<float> weights; // dimension input_dim * output_dim
	std::vector<float> biases; // dimension output_dim

	// backprop values
	std::vector<float> weight_deltas; // accmulated
	std::vector<float> bias_deltas; // accumulated
	std::vector<float> input_deltas; // updated every training datum

	// node outputs
	std::vector<float> output_layer;
};

inline void ReLU(float& value)
{
	value = std::max(0.0f, value);
}

inline float dReLU(float value)
{
	if (value > 0.0f) return 1.0f;
	else return 0.0f;
}
