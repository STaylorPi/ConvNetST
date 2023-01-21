#pragma once

#include <vector>

struct Pixel
{
	float r=0.0, g=0.0, b=0.0, a=0.0;

	Pixel& operator+=(const Pixel& rhs)
	{
		r += rhs.r; g += rhs.g; b += rhs.b; a += rhs.a;
		return *this;
	}

	Pixel& operator*=(float c)
	{
		r *= c; g *= c; b *= c; a *= c;
		return *this;
	}

	friend Pixel operator*(const Pixel& lhs, float c)
	{
		return {lhs.r * c, lhs.g * c, lhs.b * c, lhs.a * c};
	}

	float dot_rgb(const Pixel& rhs) const
	{
		return rhs.r * r + rhs.g * g + rhs.b * b;
	}
};

enum class Padding
{zero, mean};

class ImageRGBA
{
public:
	ImageRGBA(std::size_t width_, std::size_t height);
	ImageRGBA(const char* filename);

	float get_red_at(int x_pos, int y_pos, Padding padding) const;
	float get_green_at(int x_pos, int y_pos, Padding padding) const;
	float get_blue_at(int x_pos, int y_pos, Padding padding) const;
	float get_alpha_at(int x_pos, int y_pos, Padding padding) const;
	Pixel get_pixel_at(int x_pos, int y_pos, Padding padding) const;

	Pixel get_average_pixel() const;

	void set_pixel_at(std::size_t x, std::size_t y, const Pixel& p);

	auto get_width() const { return width; }
	auto get_height() const { return height; }

private:
	std::size_t width;
	std::size_t height;

	std::vector<float> r_channel;
	std::vector<float> g_channel;
	std::vector<float> b_channel;
	std::vector<float> a_channel;

	Pixel mean_pixel;
};

class ImageGrey
{
public:
	ImageGrey(std::size_t width_, std::size_t height);
	ImageGrey(const ImageRGBA& colour_image);

	float get_value_at(int x_pos, int y_pos, Padding padding) const;
	void set_pixel_at(std::size_t x, std::size_t y, float value)
	{ data[y * width + x] = value; }

	float get_average_pixel() const;

	auto get_width() const { return width; }
	auto get_height() const { return height; }

	const auto& get_data() const { return data; }

private:
	std::size_t width;
	std::size_t height;

	std::vector<float> data;
	float mean_pixel;
};
