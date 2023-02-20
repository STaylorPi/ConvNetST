#pragma once

#include "conv_kernel.hpp"

// layer input: a feature map with n layers
// layer output: a feature map with m layers (m kernels convolve over the n layered feature map)


class ConvLayer
{
public:
	ConvLayer(const FeatureMap& input_map, std::size_t kernels, std::size_t kernel_dim);
	ConvLayer(std::size_t width, std::size_t height, std::size_t inputs, std::size_t outputs, std::size_t kernel_dim, std::size_t padding_=0, std::size_t stride_=1);

	// calculates and updates the output feature_map, including normalise_ReLU
	void compute(const FeatureMap& input_map);

	void random_init_kernels(float max_bound);

	const auto& get_output() const { return output; }

	// output dimensions
	auto width() const { return output.get_width(); }
	auto height() const { return output.get_height(); }
	auto layers() const { return output.get_layers(); }

	void update_backprop(const FeatureMap& input_map, const FeatureMap& output_deltas);
	void update_params(float grad_mul);

	const auto& get_backprop_deltas() const { return input_deltas; }

private:
	// computes d(output_map)/d(kernel_weight) at the given position
	float get_weight_delta_at(const FeatureMap& input_map, std::size_t kernel, std::size_t x_off, std::size_t y_off, std::size_t x_out, std::size_t y_out);

private:
	FeatureMap output;
	std::vector<ConvKernel> kernels;
	std::vector<ConvKernel> kernel_deltas; // accumulated for backprop

	FeatureMap input_deltas; // updated once per training example

	std::size_t input_dim; // number of input layers
	std::size_t output_dim; // number of output layers
	std::size_t kernel_size;

	std::size_t stride;
	std::size_t padding;
};
