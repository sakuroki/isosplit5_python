#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "ndarray.h"
#include "isosplit5.h"

int isosplit5_interface(py::array_t<int> labels_out,
                        py::array_t<float> X,
                        float isocut_threshold,
                        int min_cluster_size,
                        int K_init,
                        bool refine_clusters,
                        int max_iterations_per_pass
                        )
{
    NDArray<float> Xa(X);
    NDArray<int> La(labels_out);
    bigint M=Xa.shape[0];
    bigint N=Xa.shape[1];
    isosplit5_opts opts;
    
    // added by sk 20'/12/04
    opts.isocut_threshold = isocut_threshold;
    opts.min_cluster_size = min_cluster_size;
    opts.K_init = K_init;
    opts.refine_clusters = refine_clusters;
    opts.max_iterations_per_pass = max_iterations_per_pass;
    
    isosplit5(La.ptr,M,N,Xa.ptr,opts);
    return 0;
}

PYBIND11_MODULE(isosplit5_interface, m) {
    m.doc() = "Python interface to isosplit clustering"; // optional module docstring
    
    m.def("isosplit5_interface", &isosplit5_interface, "ISO-SPLIT clustering",
          py::arg("labels_out").noconvert(),
          py::arg("X").noconvert(),
          py::arg("isocut_threshold"),
          py::arg("min_cluster_size"),
          py::arg("K_init"),
          py::arg("refine_clusters"),
          py::arg("max_iterations_per_pass")
    );
}
