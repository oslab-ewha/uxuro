#include "backprop.h"

int
generate_bpnn(int layer_size)
{
	BPNN	*net;

	net = bpnn_create(layer_size, 16, 1);
	bpnn_free(net);

	return 0;
}
