#pragma once

#include "conv_layer.hpp"

class PoolLayer
{
public:
	PoolLayer(const ConvLayer& in_layer, std::size_t pool_dim); // init out_map from in_layer
	
	void compute(const FeatureMap& input_map);

	const auto& get_output() const { return output_map; }

private:
	// gets the pooled value at tl (x * pool_size, y * pool_size), also setting the bit mask
	float get_pool_value(const FeatureMap& input_map, std::size_t x, std::size_t y, std::size_t layer);

private:
	std::size_t pool_size; // size of square to sample from
	// stride == pool_size
	
	FeatureMap output_map;
	std::vector<bool> pool_mask; // which pixels are selected from the input map
};
