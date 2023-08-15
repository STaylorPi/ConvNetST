#pragma once

#include "conv_layer.hpp"

class PoolLayer
{
public:
	PoolLayer(const ConvLayer& in_layer, std::size_t pool_dim); // init out_map from in_layer

	void compute(const FeatureMap& input_map);

	const auto& get_output() const { return output_map; }

	// output dimensions
	auto width() const { return output_map.get_width(); }
	auto height() const { return output_map.get_height(); }
	auto layers() const { return output_map.get_layers(); }
	
	void update_backprop(const FeatureMap& input_map, const FeatureMap& output_deltas);
	void update_backprop(const FeatureMap& input_map, const std::vector<float>& output_deltas);
	const auto& get_backprop_deltas() const { return input_deltas; }

private:
	// gets the pooled value at tl (x * pool_size, y * pool_size), also setting the bit mask
	float get_pool_value(const FeatureMap& input_map, std::size_t x, std::size_t y, std::size_t layer);

	struct Point { std::size_t x; std::size_t y; };

	// finds the input coordinate that was mapped to a given output (using pool_mask)
	Point find_input_location(std::size_t out_x, std::size_t out_y, std::size_t out_layer);

private:
	std::size_t pool_size; // size of square to sample from
	// stride == pool_size
		
	std::size_t input_width;
	std::size_t input_height;
	std::size_t input_layers;

	FeatureMap output_map;
	std::vector<bool> pool_mask; // which pixels are selected from the input map

	FeatureMap input_deltas;
};
