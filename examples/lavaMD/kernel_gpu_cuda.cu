static __global__ void
kernel_gpu_cuda(par_str d_par_gpu, dim_str d_dim_gpu, box_str *d_box_gpu,
		FOUR_VECTOR *d_rv_gpu, fp_t *d_qv_gpu, FOUR_VECTOR *d_fv_gpu)
{
	int	bx = blockIdx.x;	// get current horizontal block index (0-n)
	int	tx = threadIdx.x;	// get current horizontal thread index (0-n)
	int	wtx = tx;

	if (bx < d_dim_gpu.number_boxes) {
		// parameters
		fp_t	a2 = 2.0*d_par_gpu.alpha*d_par_gpu.alpha;

		// home box
		int	first_i;
		FOUR_VECTOR	*rA;
		FOUR_VECTOR	*fA;
		__shared__ FOUR_VECTOR	rA_shared[100];

		// nei box
		int	pointer;
		int	k = 0;
		int	first_j;
		FOUR_VECTOR	*rB;
		fp_t	*qB;
		int	j = 0;
		__shared__ FOUR_VECTOR	rB_shared[100];
		__shared__ double	qB_shared[100];

		// common
		fp_t r2, u2, vij, fs, fxij, fyij, fzij;
		THREE_VECTOR	d;

		// home box - box parameters
		first_i = d_box_gpu[bx].offset;

		// home box - distance, force, charge and type parameters
		rA = &d_rv_gpu[first_i];
		fA = &d_fv_gpu[first_i];

		// home box - shared memory
		while (wtx < NUMBER_PAR_PER_BOX) {
			rA_shared[wtx] = rA[wtx];
			wtx = wtx + NUMBER_THREADS;
		}
		wtx = tx;

		// synchronize threads  - not needed, but just to be safe
		__syncthreads();

		//------------------------------
		//	nei box loop
		//------------------------------

		// loop over neiing boxes of home box
		for (k = 0; k < (1 + d_box_gpu[bx].nn); k++) {
			//	nei box - get pointer to the right box
			if (k == 0) {
				pointer = bx;	// set first box to be processed to home box
			}
			else {
				pointer = d_box_gpu[bx].nei[k - 1].number;	// remaining boxes are nei boxes
			}

			//	Setup parameters

			// nei box - box parameters
			first_j = d_box_gpu[pointer].offset;

			// nei box - distance, (force), charge and (type) parameters
			rB = &d_rv_gpu[first_j];
			qB = &d_qv_gpu[first_j];

			//	Setup parameters

			// nei box - shared memory
			while (wtx < NUMBER_PAR_PER_BOX) {
				rB_shared[wtx] = rB[wtx];
				qB_shared[wtx] = qB[wtx];
				wtx = wtx + NUMBER_THREADS;
			}
			wtx = tx;

			// synchronize threads because in next section each thread accesses data brought in by different threads here
			__syncthreads();

			//	Calculation

			// loop for the number of particles in the home box
			while (wtx < NUMBER_PAR_PER_BOX) {
				// loop for the number of particles in the current nei box
				for (j = 0; j < NUMBER_PAR_PER_BOX; j++) {
					r2 = (fp_t)rA_shared[wtx].v + (fp_t)rB_shared[j].v - DOT((fp_t)rA_shared[wtx], (fp_t)rB_shared[j]); 
					u2 = a2 * r2;
					vij = exp(-u2);
					fs = 2*vij;

					d.x = (fp_t)rA_shared[wtx].x  - (fp_t)rB_shared[j].x;
					fxij=fs*d.x;
					d.y = (fp_t)rA_shared[wtx].y  - (fp_t)rB_shared[j].y;
					fyij=fs*d.y;
					d.z = (fp_t)rA_shared[wtx].z  - (fp_t)rB_shared[j].z;
					fzij = fs * d.z;

					fA[wtx].v +=  (double)((fp_t)qB_shared[j] * vij);
					fA[wtx].x +=  (double)((fp_t)qB_shared[j] * fxij);
					fA[wtx].y +=  (double)((fp_t)qB_shared[j] * fyij);
					fA[wtx].z +=  (double)((fp_t)qB_shared[j] * fzij);
				}

				// increment work thread index
				wtx = wtx + NUMBER_THREADS;
			}

			// reset work index
			wtx = tx;

			// synchronize after finishing force contributions from current nei box not to cause conflicts when starting next box
			__syncthreads();
		}
	}
}
