#pragma once

#include <array>
#include <algorithm>

#include "image.hpp"
#include "feature_map.hpp"

// an NxN convolution kernel
class ConvKernel {
public:
	ConvKernel(std::size_t dimension);
	ConvKernel(const std::vector<float>& fill_from, std::size_t dimension);

	void normalise(); // ensure that the weights sum to one

	const auto& view() const { return data; }
	auto& view() { return data; }
	auto dim() const { return n; }

	float dot_at(const ImageRGBA& image, int pixel_x, int pixel_y, Padding padding) const;
	float dot_at(const ImageGrey& image, int pixel_x, int pixel_y, Padding padding) const;
	float dot_at(const FeatureMap& f_map, int tl_x, int tl_y) const;

	ConvKernel operator+(const ConvKernel& rhs); // component adds same-size kernels
	ConvKernel& operator+=(const ConvKernel& rhs);

	ConvKernel operator*(float coeff);

private:
	std::vector<float> data; // inner rows, outer columns
	std::size_t n=0; // nxn grid
};

ImageGrey convolve(const ImageGrey& image, const ConvKernel& kernel, std::size_t pad_size, Padding pad_type);
FeatureMap convolve(const FeatureMap& f_map, const ConvKernel& kernel, std::size_t padding, std::size_t stride);

std::size_t get_new_dim(std::size_t old_dim, std::size_t kernel_dim, std::size_t padding, std::size_t stride);


void convolve_to(const FeatureMap& in_map, FeatureMap& out_map, const ConvKernel& kernel, std::size_t padding, std::size_t stride, std::size_t layer);

