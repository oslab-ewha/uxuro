#ifndef _BPNN_H_
#define _BPNN_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
	long	input_n;	/* number of input units */
	long	hidden_n;	/* number of hidden units */
	long	output_n;	/* number of output units */

	float	*input_units;	/* the input units */
	float	*hidden_units;	/* the hidden units */
	float	*output_units;	/* the output units */

	float	*hidden_delta;	/* storage for hidden unit error */
	float	*output_delta;	/* storage for output unit error */

	float	*target;	/* storage for target vector */

	float	*input_weights;		/* weights from input to hidden layer */
	float	*hidden_weights;	/* weights from hidden to output layer */
	
	/*** The next two are for momentum ***/
	float	*input_prev_weights;	/* previous change on input to hidden wgt */
	float	*hidden_prev_weights;	/* previous change on hidden to output wgt */

	float	*kernel_input_units;
	float	*kernel_input_weights;
	float	*partial_sum;
	float	*kernel_partial_sum;
	float	*kernel_hidden_delta;
	float	*kernel_prev_weights;
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
void bpnn_load(BPNN *net, const char *folder);
void bpnn_save(BPNN *net, const char *folder);
void bpnn_free(BPNN *net);

#ifdef __cplusplus
}
#endif
#endif
