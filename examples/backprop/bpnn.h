#ifndef _BPNN_H_
#define _BPNN_H_

#include "cudaio.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	long	input_n;	/* number of input units */
	long	hidden_n;	/* number of hidden units */
	long	output_n;	/* number of output units */

	cuio_ptr_t	input_units;	/* the input units */
	cuio_ptr_t	hidden_units;	/* the hidden units */
	cuio_ptr_t	output_units;	/* the output units */

	cuio_ptr_t	hidden_delta;	/* storage for hidden unit error */
	cuio_ptr_t	output_delta;	/* storage for output unit error */

	cuio_ptr_t	target;		/* storage for target vector */

	cuio_ptr_t	input_weights;	/* weights from input to hidden layer */
	cuio_ptr_t	hidden_weights;	/* weights from hidden to output layer */
	
	/*** The next two are for momentum ***/
	cuio_ptr_t	input_prev_weights;	/* previous change on input to hidden wgt */
	cuio_ptr_t	hidden_prev_weights;	/* previous change on hidden to output wgt */
	cuio_ptr_t	partial_sum;

	//float	*kernel_input_units;
	//float	*kernel_input_weights;
	//float	*kernel_partial_sum;
	//float	*kernel_hidden_delta;
	//float	*kernel_prev_weights;
} BPNN;

void bpnn_prepare(BPNN *net, unsigned long num_blocks);
void bpnn_update_hidden(BPNN *net, unsigned long num_blocks);

void bpnn_layerforward(BPNN *net);
float bpnn_output_error(BPNN *net);
float bpnn_hidden_error(BPNN *net);
void bpnn_adjust_weights(BPNN *net);

void bpnn_prepare_delta(BPNN *net);
void bpnn_finalize(BPNN *net);

BPNN *bpnn_create(long n_in, long n_hidden, long n_out);
BPNN *bpnn_load(const char *folder);
void bpnn_save(BPNN *net);
void bpnn_free(BPNN *net);

#ifdef __cplusplus
}
#endif
#endif
