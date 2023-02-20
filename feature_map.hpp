#pragma once

#include "image.hpp"

class FeatureMap
{
public:
	FeatureMap(std::size_t width_, std::size_t height_, std::size_t layers_);
	FeatureMap(const ImageGrey& image);

	// assumes width = height = 1
	FeatureMap(const std::vector<float>& fill_from);

	void set_at(std::size_t w_pos, std::size_t h_pos, std::size_t layer, float value);
	void set_at(std::size_t w_pos, std::size_t h_pos, float value); // sets for all layers
	float get_at(std::size_t w_pos, std::size_t h_pos, std::size_t layer) const;

	const auto& get_data() const { return data; }

	auto get_width() const { return width; }
	auto get_height() const { return height; }
	auto get_layers() const { return layers; }

	void normalise_ReLU();

	void fill_with(float value) { std::fill(data.begin(), data.end(), value); }

private:
	std::vector<float> data; // contiguous 3d array (width * height * layers)
	std::size_t width, height, layers;
};
