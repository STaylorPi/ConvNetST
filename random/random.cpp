#include "random.hpp"

namespace Rand {
	std::random_device rand_source{};

	float random()
	{
		return ((float)rand_source() - rand_source.min())/((float)rand_source.max() - rand_source.min());
	}

	float random_bound(float upper)
	{
		return (random() - 0.5) * 2 * upper;
	}
}
