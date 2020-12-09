import numpy as np
import isosplit5_interface

def isosplit5(X,
              isocut_threshold = 1.0,
              min_cluster_size = 10,
              K_init = 200,
              refine_clusters = False,
              max_iterations_per_pass = 500):
	X=X.astype(np.float32,copy=False,order='F') #copies only if type changes, but note that we require fortran order, which is essential for the interface
	M=X.shape[0]
	N=X.shape[1]
	labels_out=np.zeros([N]).astype(np.int32)
	isosplit5_interface.isosplit5_interface(labels_out,X,
                                            isocut_threshold,
                                            min_cluster_size,
                                            K_init,
                                            refine_clusters,
                                            max_iterations_per_pass)
	return labels_out
