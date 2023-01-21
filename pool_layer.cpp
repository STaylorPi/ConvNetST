#include "pool_layer.hpp"

PoolLayer::PoolLayer(const ConvLayer& in_layer, std::size_t pool_dim=2)
	:pool_size{ pool_dim }, output_map(
			in_layer.get_output().get_width() / pool_dim,
			in_layer.get_output().get_height() / pool_dim,
			in_layer.get_output().get_layers()),
	pool_mask(in_layer.get_output().get_width() * in_layer.get_output().get_height() * output_map.get_layers())
{
}

void PoolLayer::compute(const FeatureMap &input_map)
{
	std::fill(pool_mask.begin(), pool_mask.end(), false);
	for (std::size_t layer=0; layer < output_map.get_layers(); ++layer) {
		for (std::size_t y=0; y < output_map.get_height(); ++y) {
			for (std::size_t x=0; x < output_map.get_width(); ++x) {
				output_map.set_at(x, y, layer,
						get_pool_value(input_map, x, y, layer));
			}
		}
	}
}

float PoolLayer::get_pool_value(const FeatureMap& input_map, std::size_t x, std::size_t y, std::size_t layer)
{
	std::size_t chosen_i = 0, chosen_j = 0;
	float max_value = 0.0f;
	for (std::size_t j=0; j < pool_size; ++j) {
		for (std::size_t i=0; i < pool_size; ++i) {
			float current_value = input_map.get_at(x * pool_size + i, y * pool_size + j, layer);
			if ( current_value > max_value )
			{
				chosen_i = i;
				chosen_j = j;
				max_value = current_value;
			}
		}
	}

	pool_mask[layer * input_map.get_width() * input_map.get_height() + (y * pool_size + chosen_j) * input_map.get_width() + (x * pool_size + chosen_i)] = true;

	return max_value;
}
