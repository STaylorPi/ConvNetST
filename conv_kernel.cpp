#include "conv_layer.hpp"

ConvKernel::ConvKernel(std::size_t dimension)
	:data(dimension * dimension), n{ dimension }
{
}

ConvKernel::ConvKernel(const std::vector<float>& fill_from, std::size_t dimension)
		:n{ dimension }
{
	if (fill_from.size() != dimension * dimension)
	{
		data.resize(dimension * dimension);
		return;
	}
	data = fill_from;
}

void ConvKernel::normalise()
{
	float sum = 0.0f;
	for (std::size_t i=0; i < n*n; ++i) { sum += data[i]; }
	if (sum == 0.0) return;

	for (std::size_t i=0; i < n*n; ++i) { data[i] /= sum; }
}

float ConvKernel::dot_at(const ImageRGBA& image, int pixel_x, int pixel_y, Padding padding) const
{
	float product = 0.0f;
	for (std::size_t y=0; y < n; ++y)
	{
		for (std::size_t x=0; x < n; ++x)
			product += image.get_pixel_at(x, y,
					padding).dot_rgb(
						Pixel{data[y * n + x], data[y * n + x], data[y * n + x], 0.0f});
	}

	return product;
}

float ConvKernel::dot_at(const ImageGrey& image, int pixel_x, int pixel_y, Padding padding) const
{
	float product = 0.0f;
	for (std::size_t y=0; y < n; ++y)
	{
		for (std::size_t x=0; x < n; ++x)
			product += image.get_value_at(x, y, padding) * data[y * n + x];
	}

	return product;
}

float ConvKernel::dot_at(const FeatureMap& f_map, int tl_x, int tl_y) const
{
	float product = 0.0f;
	for (std::size_t layer=0; layer < f_map.get_layers(); layer++) {
		for (std::size_t y=0; y < n; ++y) {
			for (std::size_t x=0; x < n; ++x) {
				product += f_map.get_at(tl_x + x, tl_y + y, layer) * data[y * n + x];
			}
		}
	}

	return product;
}

ConvKernel ConvKernel::operator+(const ConvKernel& rhs)
{
	if (rhs.dim() != dim()) return ConvKernel(dim());
	ConvKernel sum(dim());
	std::transform(data.begin(), data.end(), rhs.data.begin(), sum.data.begin(), std::plus<float>());

	return sum;
}

ConvKernel& ConvKernel::operator+=(const ConvKernel& rhs)
{
	if (rhs.dim() != dim()) return *this;

	std::transform(data.begin(), data.end(), rhs.data.begin(), data.begin(), std::plus<float>());

	return *this;
}

ConvKernel ConvKernel::operator*(float coeff)
{
	ConvKernel out(data, n);
	std::transform(out.data.begin(), out.data.end(), out.data.begin(), [&coeff](auto& val){
			return val * coeff;
			});
	return out;
}

ImageGrey convolve(const ImageGrey& image, const ConvKernel& kernel, std::size_t pad_size, Padding pad_type)
{
	std::size_t new_width = image.get_width() - kernel.dim() + 2 * pad_size + 1;
	std::size_t new_height = image.get_height() - kernel.dim() + 2 * pad_size + 1;
	ImageGrey output{new_width, new_height};

	for (std::size_t y=0; y < new_height; ++y)
	{
		for (std::size_t x=0; x < new_width; ++x)
		{
			float new_pixel = kernel.dot_at(image, x - pad_size, y - pad_size, pad_type);
			output.set_pixel_at(x, y, new_pixel);
		}
	}

	return output;
}

FeatureMap convolve(const FeatureMap& f_map, const ConvKernel& kernel, std::size_t padding, std::size_t stride)
{
	std::size_t new_width = ((f_map.get_width() - kernel.dim() + 2 * padding + 1) / stride);
	std::size_t new_height = ((f_map.get_height() - kernel.dim() + 2 * padding + 1) / stride);
	FeatureMap result(new_width, new_height, 1);

	for (std::size_t y = 0; y < new_height; ++y) {
		for (std::size_t x = 0; x < new_width; ++x) {
			result.set_at(x, y, 0,
					kernel.dot_at(f_map, x * stride, y * stride));
		}
	}

	return result;
}

std::size_t get_new_dim(std::size_t old_dim, std::size_t kernel_dim, std::size_t padding, std::size_t stride)
{
	return ((old_dim - kernel_dim + 2 * padding + 1) / stride);
}

// leaves data unchanged if the dimensionality is not correct
void convolve_to(const FeatureMap& in_map, FeatureMap& out_map, const ConvKernel& kernel, std::size_t padding, std::size_t stride, std::size_t layer)
{
	std::size_t new_width = get_new_dim(in_map.get_width(), kernel.dim(), padding, stride);
	std::size_t new_height = get_new_dim(in_map.get_height(), kernel.dim(), padding, stride);
	if ((out_map.get_width() != new_width) || (out_map.get_height() != new_height)) return;

	for (std::size_t y = 0; y < new_height; ++y) {
		for (std::size_t x = 0; x < new_width; ++x) {
			out_map.set_at(x, y, layer,
					kernel.dot_at(in_map, x * stride, y * stride));
		}
	}
}
