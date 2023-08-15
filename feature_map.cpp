#include "feature_map.hpp"

FeatureMap::FeatureMap(std::size_t width_, std::size_t height_, std::size_t layers_)
	:data(width_ * height_ * layers_), width{ width_ }, height{ height_ }, layers{ layers_ }
{
}

FeatureMap::FeatureMap(std::size_t width_, std::size_t height_, std::size_t layers_, const std::vector<float>& fill_from)
	:data(width_ * height_ * layers_), width{ width_ }, height{ height_ }, layers{ layers_ }
{
	if (fill_from.size() == data.size()) data = fill_from;
}

FeatureMap::FeatureMap(const ImageGrey& image)
	:data(image.get_data()), width{ image.get_width() },
	height{ image.get_height() }, layers{ 1 }
{
}

void FeatureMap::set_at(std::size_t w_pos, std::size_t h_pos, std::size_t layer, float value)
{
	if (w_pos >= width) return;
	if (h_pos >= height) return;
	if (layer >= layers) return;

	data[layer * width * height + h_pos * width + w_pos] = value;
}

void FeatureMap::set_at(std::size_t w_pos, std::size_t h_pos, float value)
{
	for (std::size_t l=0; l < layers; ++l)
	{
		set_at(w_pos, h_pos, l, value);
	}
}

float FeatureMap::get_at(std::size_t w_pos, std::size_t h_pos, std::size_t layer) const
{
	if (w_pos >= width) return 0.0f;
	if (h_pos >= height) return 0.0f;
	if (layer >= layers) return 0.0f;

	return data[layer * width * height + h_pos * width + w_pos];
}

void FeatureMap::normalise_ReLU()
{
	for (float& pix : data) pix = std::max(0.0f, pix);
}
