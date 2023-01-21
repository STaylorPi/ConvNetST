#pragma once

#include <random>

namespace Rand {
	// random number between 0.0f and 1.0f
	float random();

	// random number between -upper and upper
	float random_bound(float upper);
}
