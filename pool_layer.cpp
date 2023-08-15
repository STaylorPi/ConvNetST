#include "pool_layer.hpp"

PoolLayer::PoolLayer(const ConvLayer& in_layer, std::size_t pool_dim=2)
	:pool_size{ pool_dim },
	input_width{ in_layer.width() },
	input_height{ in_layer.height() },
	input_layers{ in_layer.layers() },
	output_map(
			in_layer.get_output().get_width() / pool_dim,
			in_layer.get_output().get_height() / pool_dim,
			in_layer.get_output().get_layers()),
	pool_mask(in_layer.get_output().get_width() * in_layer.get_output().get_height() * output_map.get_layers()),
	input_deltas(input_width, input_height, input_layers)
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

PoolLayer::Point PoolLayer::find_input_location(std::size_t out_x, std::size_t out_y, std::size_t out_layer)
{
	Point coord;

	for (std::size_t y_off=0; y_off < pool_size; ++y_off)
	{
		for (std::size_t x_off=0; x_off < pool_size; ++x_off)
		{
			if (pool_mask[out_layer * input_width * input_height + (out_y * pool_size + y_off) * input_height + out_x * pool_size + x_off]) {
				coord.x = out_x * pool_size + x_off;
				coord.y = out_y * pool_size + y_off;
				return coord;
			}
		}
	}

	return coord; // should never be run, ever
}

void PoolLayer::update_backprop(const FeatureMap& input_map, const FeatureMap& output_deltas)
{
	input_deltas.fill_with(0.0f);
	for (std::size_t layer=0; layer < output_map.get_layers(); ++layer)
	{
		for (std::size_t y=0; y < output_map.get_height(); ++y)
		{
			for (std::size_t x=0; x < output_map.get_width(); ++x)
			{
				auto input_point = find_input_location(x, y, layer);
				input_deltas.set_at(input_point.x, input_point.y, layer, output_deltas.get_at(x, y, layer));
			}
		}
	}
}

void PoolLayer::update_backprop(const FeatureMap &input_map, const std::vector<float> &output_deltas)
{
	update_backprop(input_map, FeatureMap(output_map.get_width(),
				output_map.get_height(), output_map.get_layers(), output_deltas));
}
