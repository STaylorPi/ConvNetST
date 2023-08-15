#include <iostream>
#include <sstream>

#include "conv_net.hpp"
#include "random/random.hpp"

int main()
{
	NetworkLayout nl;
	nl.in_x = 16;
	nl.in_y = 16;
	nl.conv_layers = 2;
	nl.kernel_dims = {3, 3};
	nl.kernels = {8, 4};
	nl.neurons = {16, 4, 1};
	nl.neuron_layers = 3;
	nl.paddings = {1, 1};
	nl.strides = {1, 1};
	nl.pool_params = {2, 2};

	ConvNet cnn(nl, 1.0f, 1.0f, 1.0f);

	FeatureMap input(3, 3, 1);
	std::size_t batch = 16;

	while (true) {
		for (int i=0; i < batch; ++i) {
			std::ostringstream name;
			name << "../../td/c" << i << ".png";
			auto input = FeatureMap(ImageGrey(ImageRGBA(name.str().c_str())));

			cnn.compute(input);
			cnn.update_backprop({1.0f - cnn.get_output()[0]});
			std::cout << fabs(1.0f - cnn.get_output()[0]) << '\n';

			std::ostringstream name2;
			name2 << "../../td/nc" << i << ".png";
			auto input2 = FeatureMap(ImageGrey(ImageRGBA(name.str().c_str())));

			cnn.compute(input2);
			cnn.update_backprop({0.0f - cnn.get_output()[0]});
			std::cout << fabs(0.0f - cnn.get_output()[0]) << '\n';
		}
		cnn.write_changes(0.0001);
	}

	std::cout << "----------------\n";

	for (int i=0; i < batch; ++i) {
		std::ostringstream name;
		name << "../../td/c" << i << ".png";
		auto input = FeatureMap(ImageGrey(ImageRGBA(name.str().c_str())));

		cnn.compute(input);
		cnn.update_backprop({1.0f - cnn.get_output()[0]});
		std::cout << fabs(1.0f - cnn.get_output()[0]) << '\n';

		std::ostringstream name2;
		name2 << "../../td/nc" << i << ".png";
		auto input2 = FeatureMap(ImageGrey(ImageRGBA(name.str().c_str())));

		cnn.compute(input2);
		cnn.update_backprop({0.0f - cnn.get_output()[0]});
		std::cout << fabs(0.0f - cnn.get_output()[0]) << '\n';
	}

	return 0;
}
