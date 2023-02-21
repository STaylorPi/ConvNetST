#include <iostream>

#include "conv_net.hpp"
#include "random/random.hpp"

int main()
{
	NetworkLayout nl;
	nl.in_x = 3;
	nl.in_y = 3;
	nl.conv_layers = 1;
	nl.kernel_dims = {3};
	nl.kernels = {1};
	nl.neurons = {1};
	nl.neuron_layers = 1;
	nl.paddings = {0};
	nl.strides = {1};
	nl.pool_params = {1};

	ConvNet cnn(nl, 1.0f, 1.0f, 1.0f);

	FeatureMap input(3, 3, 1);
	std::size_t batch = 16;

	while (true)
	{
		float res = 0.0f;
		for (int i=0; i < batch; ++i) {
			auto value = Rand::random();
			input.fill_with(value);

			cnn.compute(input);
			cnn.update_backprop({value - cnn.get_output()[0]});
			res += fabs(value - cnn.get_output()[0]);
		}

		std::cout << res / float(batch) << '\n';

		cnn.write_changes(0.001 * fabs(res / float(batch)));
	}

	return 0;
}
