#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <vector>
#include <cmath>
#include <cstring>

extern "C" int norm_mi_cuda_impl(
    const int* xb,
    const int* yb,
    int rx,
    int ry,
    int n,
    double* out_value
);

extern "C" int batch_mic_tic_equifreq_cuda_impl(
    const int* ranks_x_host,
    const int* ranks_y_host,
    int nx,
    int ny,
    int n,
    double alpha,
    int symmetric,
    double* mic_out_host,
    double* tic_out_host,
    char* err_msg,
    int err_msg_len
);


extern "C" int high_fidelity_scores_cuda_impl(
    const int* prefix_flat_host,
    const int* prefix_offsets_host,
    const int* rx_list_host,
    const int* ry_list_host,
    const int* ry_index_host,
    const int* cuts_host,
    int prefix_flat_len,
    int n_rys,
    int n_grids,
    int K,
    int n,
    double* scores_out_host,
    char* err_msg,
    int err_msg_len
);

extern "C" int batch_high_fidelity_mic_tic_cuda_impl(
    const int* prefix_flat_host,
    const int* prefix_offsets_host,
    const int* rx_list_host,
    const int* ry_list_host,
    const int* ry_index_host,
    const int* cuts_host,
    int prefix_flat_len,
    int n_pairs,
    int n_rys,
    int n_grids,
    int K,
    int n,
    double* mic_pair_out_host,
    double* tic_pair_out_host,
    char* err_msg,
    int err_msg_len
);

static PyObject* py_norm_mi(PyObject* self, PyObject* args) {
    PyObject* xb_obj = nullptr;
    PyObject* yb_obj = nullptr;
    int rx = 0;
    int ry = 0;
    int n = 0;

    if (!PyArg_ParseTuple(args, "OOiii", &xb_obj, &yb_obj, &rx, &ry, &n)) {
        return nullptr;
    }

    PyArrayObject* xb_arr = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(xb_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY)
    );
    PyArrayObject* yb_arr = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(yb_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY)
    );

    if (xb_arr == nullptr || yb_arr == nullptr) {
        Py_XDECREF(xb_arr);
        Py_XDECREF(yb_arr);
        return nullptr;
    }

    const npy_intp xb_size = PyArray_SIZE(xb_arr);
    const npy_intp yb_size = PyArray_SIZE(yb_arr);

    if (xb_size != n || yb_size != n) {
        Py_DECREF(xb_arr);
        Py_DECREF(yb_arr);
        PyErr_SetString(PyExc_ValueError, "xb and yb must both have length n");
        return nullptr;
    }

    if (rx < 2 || ry < 2 || n < 2) {
        Py_DECREF(xb_arr);
        Py_DECREF(yb_arr);
        PyErr_SetString(PyExc_ValueError, "rx, ry, and n are out of range");
        return nullptr;
    }

    const int* xb = static_cast<const int*>(PyArray_DATA(xb_arr));
    const int* yb = static_cast<const int*>(PyArray_DATA(yb_arr));

    double out_value = 0.0;
    const int status = norm_mi_cuda_impl(xb, yb, rx, ry, n, &out_value);

    Py_DECREF(xb_arr);
    Py_DECREF(yb_arr);

    if (status != 0) {
        PyErr_Format(PyExc_RuntimeError, "CUDA norm_mi failed with status code %d", status);
        return nullptr;
    }

    return PyFloat_FromDouble(out_value);
}

static PyObject* py_batch_mic_tic_equifreq(PyObject* self, PyObject* args) {
    PyObject* ranks_x_obj = nullptr;
    PyObject* ranks_y_obj = nullptr;
    double alpha = 0.6;
    int symmetric = 0;

    if (!PyArg_ParseTuple(args, "OOdi", &ranks_x_obj, &ranks_y_obj, &alpha, &symmetric)) {
        return nullptr;
    }

    PyArrayObject* ranks_x_arr = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(ranks_x_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY)
    );
    PyArrayObject* ranks_y_arr = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(ranks_y_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY)
    );

    if (ranks_x_arr == nullptr || ranks_y_arr == nullptr) {
        Py_XDECREF(ranks_x_arr);
        Py_XDECREF(ranks_y_arr);
        return nullptr;
    }

    if (PyArray_NDIM(ranks_x_arr) != 2 || PyArray_NDIM(ranks_y_arr) != 2) {
        Py_DECREF(ranks_x_arr);
        Py_DECREF(ranks_y_arr);
        PyErr_SetString(PyExc_ValueError, "rank matrices must be 2D arrays shaped (n_variables, n_samples)");
        return nullptr;
    }

    const int nx = static_cast<int>(PyArray_DIM(ranks_x_arr, 0));
    const int ny = static_cast<int>(PyArray_DIM(ranks_y_arr, 0));
    const int n_x = static_cast<int>(PyArray_DIM(ranks_x_arr, 1));
    const int n_y = static_cast<int>(PyArray_DIM(ranks_y_arr, 1));

    if (nx < 1 || ny < 1 || n_x < 2 || n_y < 2 || n_x != n_y) {
        Py_DECREF(ranks_x_arr);
        Py_DECREF(ranks_y_arr);
        PyErr_SetString(PyExc_ValueError, "rank matrices must have matching sample counts >= 2");
        return nullptr;
    }

    if (symmetric && nx != ny) {
        Py_DECREF(ranks_x_arr);
        Py_DECREF(ranks_y_arr);
        PyErr_SetString(PyExc_ValueError, "symmetric batch mode requires nx == ny");
        return nullptr;
    }

    npy_intp dims[2] = {nx, ny};
    PyArrayObject* mic_arr = reinterpret_cast<PyArrayObject*>(
        PyArray_SimpleNew(2, dims, NPY_FLOAT64)
    );
    PyArrayObject* tic_arr = reinterpret_cast<PyArrayObject*>(
        PyArray_SimpleNew(2, dims, NPY_FLOAT64)
    );

    if (mic_arr == nullptr || tic_arr == nullptr) {
        Py_XDECREF(mic_arr);
        Py_XDECREF(tic_arr);
        Py_DECREF(ranks_x_arr);
        Py_DECREF(ranks_y_arr);
        return nullptr;
    }

    double* mic_out = static_cast<double*>(PyArray_DATA(mic_arr));
    double* tic_out = static_cast<double*>(PyArray_DATA(tic_arr));
    const int matrix_size = nx * ny;
    for (int i = 0; i < matrix_size; ++i) {
        mic_out[i] = 0.0;
        tic_out[i] = 0.0;
    }

    char err_msg[512];
    err_msg[0] = '\0';

    const int status = batch_mic_tic_equifreq_cuda_impl(
        static_cast<const int*>(PyArray_DATA(ranks_x_arr)),
        static_cast<const int*>(PyArray_DATA(ranks_y_arr)),
        nx,
        ny,
        n_x,
        alpha,
        symmetric,
        mic_out,
        tic_out,
        err_msg,
        static_cast<int>(sizeof(err_msg))
    );

    Py_DECREF(ranks_x_arr);
    Py_DECREF(ranks_y_arr);

    if (status != 0) {
        Py_DECREF(mic_arr);
        Py_DECREF(tic_arr);
        if (err_msg[0] != '\0') {
            PyErr_Format(PyExc_RuntimeError, "CUDA batch_mic_tic_equifreq failed: %s", err_msg);
        } else {
            PyErr_Format(PyExc_RuntimeError, "CUDA batch_mic_tic_equifreq failed with status code %d", status);
        }
        return nullptr;
    }

    return Py_BuildValue("NN", reinterpret_cast<PyObject*>(mic_arr), reinterpret_cast<PyObject*>(tic_arr));
}


static PyArrayObject* as_int32_array(PyObject* obj, const char* name) {
    PyArrayObject* arr = reinterpret_cast<PyArrayObject*>(
        PyArray_FROM_OTF(obj, NPY_INT32, NPY_ARRAY_IN_ARRAY)
    );
    if (arr == nullptr) {
        return nullptr;
    }
    if (!PyArray_ISCARRAY(arr)) {
        Py_DECREF(arr);
        PyErr_Format(PyExc_ValueError, "%s must be a C-contiguous int32 array", name);
        return nullptr;
    }
    return arr;
}

static PyObject* py_high_fidelity_scores(PyObject* self, PyObject* args) {
    PyObject* prefix_flat_obj = nullptr;
    PyObject* prefix_offsets_obj = nullptr;
    PyObject* rx_list_obj = nullptr;
    PyObject* ry_list_obj = nullptr;
    PyObject* ry_index_obj = nullptr;
    PyObject* cuts_obj = nullptr;
    int n = 0;

    if (!PyArg_ParseTuple(args, "OOOOOOi", &prefix_flat_obj, &prefix_offsets_obj,
                          &rx_list_obj, &ry_list_obj, &ry_index_obj, &cuts_obj, &n)) {
        return nullptr;
    }

    PyArrayObject* prefix_flat = as_int32_array(prefix_flat_obj, "prefix_flat");
    PyArrayObject* prefix_offsets = as_int32_array(prefix_offsets_obj, "prefix_offsets");
    PyArrayObject* rx_list = as_int32_array(rx_list_obj, "rx_list");
    PyArrayObject* ry_list = as_int32_array(ry_list_obj, "ry_list");
    PyArrayObject* ry_index = as_int32_array(ry_index_obj, "ry_index");
    PyArrayObject* cuts = as_int32_array(cuts_obj, "cuts");

    if (!prefix_flat || !prefix_offsets || !rx_list || !ry_list || !ry_index || !cuts) {
        Py_XDECREF(prefix_flat); Py_XDECREF(prefix_offsets); Py_XDECREF(rx_list);
        Py_XDECREF(ry_list); Py_XDECREF(ry_index); Py_XDECREF(cuts);
        return nullptr;
    }

    const int prefix_flat_len = static_cast<int>(PyArray_SIZE(prefix_flat));
    const int n_rys = static_cast<int>(PyArray_SIZE(prefix_offsets));
    const int n_grids = static_cast<int>(PyArray_SIZE(rx_list));
    const int K = static_cast<int>(PyArray_SIZE(cuts));

    if (PyArray_SIZE(ry_list) != n_grids || PyArray_SIZE(ry_index) != n_grids) {
        Py_DECREF(prefix_flat); Py_DECREF(prefix_offsets); Py_DECREF(rx_list);
        Py_DECREF(ry_list); Py_DECREF(ry_index); Py_DECREF(cuts);
        PyErr_SetString(PyExc_ValueError, "rx_list, ry_list, and ry_index must have the same length");
        return nullptr;
    }

    npy_intp dims[1] = {n_grids};
    PyArrayObject* scores_arr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, dims, NPY_FLOAT64));
    if (scores_arr == nullptr) {
        Py_DECREF(prefix_flat); Py_DECREF(prefix_offsets); Py_DECREF(rx_list);
        Py_DECREF(ry_list); Py_DECREF(ry_index); Py_DECREF(cuts);
        return nullptr;
    }

    char err_msg[512]; err_msg[0] = '\0';
    const int status = high_fidelity_scores_cuda_impl(
        static_cast<const int*>(PyArray_DATA(prefix_flat)),
        static_cast<const int*>(PyArray_DATA(prefix_offsets)),
        static_cast<const int*>(PyArray_DATA(rx_list)),
        static_cast<const int*>(PyArray_DATA(ry_list)),
        static_cast<const int*>(PyArray_DATA(ry_index)),
        static_cast<const int*>(PyArray_DATA(cuts)),
        prefix_flat_len,
        n_rys,
        n_grids,
        K,
        n,
        static_cast<double*>(PyArray_DATA(scores_arr)),
        err_msg,
        static_cast<int>(sizeof(err_msg))
    );

    Py_DECREF(prefix_flat); Py_DECREF(prefix_offsets); Py_DECREF(rx_list);
    Py_DECREF(ry_list); Py_DECREF(ry_index); Py_DECREF(cuts);

    if (status != 0) {
        Py_DECREF(scores_arr);
        if (err_msg[0] != '\0') {
            PyErr_Format(PyExc_RuntimeError, "CUDA high_fidelity_scores failed: %s", err_msg);
        } else {
            PyErr_Format(PyExc_RuntimeError, "CUDA high_fidelity_scores failed with status code %d", status);
        }
        return nullptr;
    }

    return reinterpret_cast<PyObject*>(scores_arr);
}

static PyObject* py_batch_high_fidelity_mic_tic(PyObject* self, PyObject* args) {
    PyObject* prefix_flat_obj = nullptr;
    PyObject* prefix_offsets_obj = nullptr;
    PyObject* rx_list_obj = nullptr;
    PyObject* ry_list_obj = nullptr;
    PyObject* ry_index_obj = nullptr;
    PyObject* cuts_obj = nullptr;
    int n_pairs = 0;
    int n_rys = 0;
    int n = 0;

    if (!PyArg_ParseTuple(args, "OOOOOOiii", &prefix_flat_obj, &prefix_offsets_obj,
                          &rx_list_obj, &ry_list_obj, &ry_index_obj, &cuts_obj,
                          &n_pairs, &n_rys, &n)) {
        return nullptr;
    }

    PyArrayObject* prefix_flat = as_int32_array(prefix_flat_obj, "prefix_flat");
    PyArrayObject* prefix_offsets = as_int32_array(prefix_offsets_obj, "prefix_offsets");
    PyArrayObject* rx_list = as_int32_array(rx_list_obj, "rx_list");
    PyArrayObject* ry_list = as_int32_array(ry_list_obj, "ry_list");
    PyArrayObject* ry_index = as_int32_array(ry_index_obj, "ry_index");
    PyArrayObject* cuts = as_int32_array(cuts_obj, "cuts");

    if (!prefix_flat || !prefix_offsets || !rx_list || !ry_list || !ry_index || !cuts) {
        Py_XDECREF(prefix_flat); Py_XDECREF(prefix_offsets); Py_XDECREF(rx_list);
        Py_XDECREF(ry_list); Py_XDECREF(ry_index); Py_XDECREF(cuts);
        return nullptr;
    }

    const int prefix_flat_len = static_cast<int>(PyArray_SIZE(prefix_flat));
    const int n_grids = static_cast<int>(PyArray_SIZE(rx_list));
    const int K = static_cast<int>(PyArray_SIZE(cuts));

    if (PyArray_SIZE(prefix_offsets) != static_cast<npy_intp>(n_pairs) * static_cast<npy_intp>(n_rys) ||
        PyArray_SIZE(ry_list) != n_grids || PyArray_SIZE(ry_index) != n_grids) {
        Py_DECREF(prefix_flat); Py_DECREF(prefix_offsets); Py_DECREF(rx_list);
        Py_DECREF(ry_list); Py_DECREF(ry_index); Py_DECREF(cuts);
        PyErr_SetString(PyExc_ValueError, "invalid high_fidelity batch array sizes");
        return nullptr;
    }

    npy_intp dims[1] = {n_pairs};
    PyArrayObject* mic_arr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, dims, NPY_FLOAT64));
    PyArrayObject* tic_arr = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, dims, NPY_FLOAT64));
    if (mic_arr == nullptr || tic_arr == nullptr) {
        Py_XDECREF(mic_arr); Py_XDECREF(tic_arr);
        Py_DECREF(prefix_flat); Py_DECREF(prefix_offsets); Py_DECREF(rx_list);
        Py_DECREF(ry_list); Py_DECREF(ry_index); Py_DECREF(cuts);
        return nullptr;
    }

    char err_msg[512]; err_msg[0] = '\0';
    const int status = batch_high_fidelity_mic_tic_cuda_impl(
        static_cast<const int*>(PyArray_DATA(prefix_flat)),
        static_cast<const int*>(PyArray_DATA(prefix_offsets)),
        static_cast<const int*>(PyArray_DATA(rx_list)),
        static_cast<const int*>(PyArray_DATA(ry_list)),
        static_cast<const int*>(PyArray_DATA(ry_index)),
        static_cast<const int*>(PyArray_DATA(cuts)),
        prefix_flat_len,
        n_pairs,
        n_rys,
        n_grids,
        K,
        n,
        static_cast<double*>(PyArray_DATA(mic_arr)),
        static_cast<double*>(PyArray_DATA(tic_arr)),
        err_msg,
        static_cast<int>(sizeof(err_msg))
    );

    Py_DECREF(prefix_flat); Py_DECREF(prefix_offsets); Py_DECREF(rx_list);
    Py_DECREF(ry_list); Py_DECREF(ry_index); Py_DECREF(cuts);

    if (status != 0) {
        Py_DECREF(mic_arr); Py_DECREF(tic_arr);
        if (err_msg[0] != '\0') {
            PyErr_Format(PyExc_RuntimeError, "CUDA batch_high_fidelity_mic_tic failed: %s", err_msg);
        } else {
            PyErr_Format(PyExc_RuntimeError, "CUDA batch_high_fidelity_mic_tic failed with status code %d", status);
        }
        return nullptr;
    }

    return Py_BuildValue("NN", reinterpret_cast<PyObject*>(mic_arr), reinterpret_cast<PyObject*>(tic_arr));
}

static PyMethodDef CumineCudaMethods[] = {
    {"norm_mi", py_norm_mi, METH_VARARGS, "Compute normalized MI using the native CUDA histogram core."},
    {"batch_mic_tic_equifreq", py_batch_mic_tic_equifreq, METH_VARARGS,
     "Compute batched MIC/TIC matrices for the fast estimator using native CUDA."},
    {"high_fidelity_scores", py_high_fidelity_scores, METH_VARARGS,
     "Compute high_fidelity characteristic-matrix scores using native CUDA DP."},
    {"batch_high_fidelity_mic_tic", py_batch_high_fidelity_mic_tic, METH_VARARGS,
     "Compute batched high_fidelity MIC/TIC pair scores using native CUDA DP."},
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef CumineCudaModule = {
    PyModuleDef_HEAD_INIT,
    "_cuda_ext",
    "Native CUDA extension for cumine.",
    -1,
    CumineCudaMethods
};

PyMODINIT_FUNC PyInit__cuda_ext(void) {
    import_array();
    return PyModule_Create(&CumineCudaModule);
}
