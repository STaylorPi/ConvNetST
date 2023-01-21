#pragma once

#include "image.hpp"

class FeatureMap
{
public:
	FeatureMap(std::size_t width_, std::size_t height_, std::size_t layers_);
	FeatureMap(const ImageGrey& image);

	void set_at(std::size_t w_pos, std::size_t h_pos, std::size_t layer, float value);
	float get_at(std::size_t w_pos, std::size_t h_pos, std::size_t layer) const;

	const auto& get_data() const { return data; }

	auto get_width() const { return width; }
	auto get_height() const { return height; }
	auto get_layers() const { return layers; }

	void normalise_ReLU();

private:
	std::vector<float> data; // contiguous 3d array (width * height * layers)
	std::size_t width, height, layers;
};
