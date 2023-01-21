#include <iostream>

#include "pool_layer.hpp"

int main()
{
	ImageRGBA image{"../../test2.png"};
	ImageGrey map{ image };
	FeatureMap f_map{ map };

	ConvLayer conv(f_map.get_width(), f_map.get_height(), 1, 1, 3, 0);
	conv.random_init_kernels(1.0f);
	conv.compute(f_map);

	PoolLayer pool(conv, 2);
	pool.compute(conv.get_output());


	return 0;
}
