#include "image.hpp"
#include "vendor/stb_image/stb_image.h"

#include <numeric>

ImageRGBA::ImageRGBA(std::size_t width_, std::size_t height_)
	:width{ width_ }, height{ height_ },
	r_channel(width * height),
	g_channel(width * height),
	b_channel(width * height),
	a_channel(width * height)
{
	mean_pixel = get_average_pixel();
}

ImageRGBA::ImageRGBA(const char* filename)
{
	int load_width, load_height, channels;
	unsigned char* image_buffer = stbi_load(filename, &load_width, &load_height, &channels, STBI_rgb_alpha);

	width = (std::size_t)load_width;
	height = (std::size_t)load_height;

	r_channel.resize(width * height);
	g_channel.resize(width * height);
	b_channel.resize(width * height);
	a_channel.resize(width * height);

	for (std::size_t y=0; y < height; ++y) {
		for (std::size_t x=0; x < width; ++x) {
			std::size_t pixel_pos = y * width + x;
			std::size_t pixel_byte = 4 * pixel_pos;

			// compiler SIMD optimise please :)
			r_channel[pixel_pos] = image_buffer[pixel_byte] / 255.0f;
			g_channel[pixel_pos] = image_buffer[pixel_byte + 1] / 255.0f;
			b_channel[pixel_pos] = image_buffer[pixel_byte + 2] / 255.0f;
			a_channel[pixel_pos] = image_buffer[pixel_byte + 3] / 255.0f;
		}
	}

	stbi_image_free((void*)image_buffer);

	mean_pixel = get_average_pixel();
}

float ImageRGBA::get_red_at(int x_pos, int y_pos, Padding padding) const
{
	if (padding == Padding::zero) {
		if (x_pos < 0) return 0.0f;
		if (y_pos < 0) return 0.0f;
		if (x_pos >= int(width)) return 0.0f;
		if (y_pos >= int(height)) return 0.0f;
	}
	else if (padding == Padding::mean)
	{
		if (x_pos < 0) return mean_pixel.r;
		if (y_pos < 0) return mean_pixel.r;
		if (x_pos >= int(width)) return mean_pixel.r;
		if (y_pos >= int(height)) return mean_pixel.r;
	}

	return r_channel[y_pos * width + x_pos];
}

float ImageRGBA::get_green_at(int x_pos, int y_pos, Padding padding) const
{
	if (padding == Padding::zero) {
		if (x_pos < 0) return 0.0f;
		if (y_pos < 0) return 0.0f;
		if (x_pos >= int(width)) return 0.0f;
		if (y_pos >= int(height)) return 0.0f;
	}
	else if (padding == Padding::mean)
	{
		if (x_pos < 0) return mean_pixel.g;
		if (y_pos < 0) return mean_pixel.g;
		if (x_pos >= int(width)) return mean_pixel.g;
		if (y_pos >= int(height)) return mean_pixel.g;
	}

	return g_channel[y_pos * width + x_pos];
}

float ImageRGBA::get_blue_at(int x_pos, int y_pos, Padding padding) const
{
	if (padding == Padding::zero) {
		if (x_pos < 0) return 0.0f;
		if (y_pos < 0) return 0.0f;
		if (x_pos >= int(width)) return 0.0f;
		if (y_pos >= int(height)) return 0.0f;
	}
	else if (padding == Padding::mean)
	{
		if (x_pos < 0) return mean_pixel.b;
		if (y_pos < 0) return mean_pixel.b;
		if (x_pos >= int(width)) return mean_pixel.b;
		if (y_pos >= int(height)) return mean_pixel.b;
	}

	return b_channel[y_pos * width + x_pos];
}

float ImageRGBA::get_alpha_at(int x_pos, int y_pos, Padding padding) const
{
	if (padding == Padding::zero) {
		if (x_pos < 0) return 0.0f;
		if (y_pos < 0) return 0.0f;
		if (x_pos >= int(width)) return 0.0f;
		if (y_pos >= int(height)) return 0.0f;
	}
	else if (padding == Padding::mean)
	{
		if (x_pos < 0) return mean_pixel.a;
		if (y_pos < 0) return mean_pixel.a;
		if (x_pos >= int(width)) return mean_pixel.a;
		if (y_pos >= int(height)) return mean_pixel.a;
	}

	return a_channel[y_pos * width + x_pos];
}

Pixel ImageRGBA::get_average_pixel() const
{
	Pixel mean;
	for (std::size_t i=0; i< width * height; ++i)
	{
		mean += Pixel{ r_channel[i], g_channel[i], b_channel[i], a_channel[i] } * ( 1.0f / (width * height) );
	}

	return mean;
}

Pixel ImageRGBA::get_pixel_at(int x_pos, int y_pos, Padding padding) const
{
	return Pixel{
		get_red_at(x_pos, y_pos, padding),
		get_green_at(x_pos, y_pos, padding),
		get_blue_at(x_pos, y_pos, padding),
		get_alpha_at(x_pos, y_pos, padding)
	};
}

void ImageRGBA::set_pixel_at(std::size_t x, std::size_t y, const Pixel &p)
{
	r_channel[y * width + x] = p.r;
	g_channel[y * width + x] = p.g;
	b_channel[y * width + x] = p.b;
	a_channel[y * width + x] = p.a;

	mean_pixel = get_average_pixel();
}

// ImageGrey -----------------------------------------------------------------------------------------------------

ImageGrey::ImageGrey(std::size_t width_, std::size_t height_)
	:width{ width_ }, height{ height_ }, data(width * height),
	mean_pixel{ 0.0f }
{
}

ImageGrey::ImageGrey(const ImageRGBA& colour_image)
	:width{ colour_image.get_width() },
	height{ colour_image.get_height() },
	data(width* height)
{
	for (int y=0; y < int(height); ++y)
	{
		for (int x=0; x < int(width); ++x)
		{
			const auto& pixel = colour_image.get_pixel_at(x, y, Padding::zero);
			data[y * width + x] = pixel.r * 0.29970f + pixel.g * 0.587130f + pixel.b * 0.114180f;
		}
	}

	mean_pixel = get_average_pixel();
}

float ImageGrey::get_average_pixel() const
{
	return float(std::accumulate(data.begin(), data.end(), 0.0) / (width * height));
}

float ImageGrey::get_value_at(int x_pos, int y_pos, Padding padding) const
{
	if (padding == Padding::zero) {
		if (x_pos < 0) return 0.0f;
		if (y_pos < 0) return 0.0f;
		if (x_pos >= int(width)) return 0.0f;
		if (y_pos >= int(height)) return 0.0f;
	}
	else if (padding == Padding::mean)
	{
		if (x_pos < 0) return mean_pixel;
		if (y_pos < 0) return mean_pixel;
		if (x_pos >= int(width)) return mean_pixel;
		if (y_pos >= int(height)) return mean_pixel;
	}

	return data[y_pos * width + x_pos];
}
