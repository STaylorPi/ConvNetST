#include <iostream>

#include "conv_net_st.hpp"

int main()
{
	std::vector<NeuronLayer> layers(4, {1, 1});
	for (auto &l : layers) { l.random_init_params(1.0f, 1.0f); }
	
	std::size_t count = 0;
	while (true)
	{
		float input = Rand::random(); auto output = 1 - input;
		layers[0].compute({input});
		for (int i=1; i < layers.size(); ++i) { layers[i].compute(layers[i-1].get_output()); }

		std::cout << output - layers[layers.size() - 1].get_output()[0] << '\n';

		layers[layers.size()-1].update_backprop(layers[layers.size()-2].get_output(),
				{output - layers[layers.size()-1].get_output()[0]});

		for (int i=layers.size() - 2; i > 0; --i) { layers[i].update_backprop(layers[i-1].get_output(), layers[i+1].get_backprop_deltas()); }
		layers[0].update_backprop({input}, layers[1].get_backprop_deltas());
	
		if (count % 100 == 0)
			for (auto& layer : layers) { layer.update_params(0.0005); }

		count++;
	}

	return 0;
}
