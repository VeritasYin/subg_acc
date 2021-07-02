#define PY_SSIZE_T_CLEAN

#include </Users/veritas/opt/anaconda3/include/python3.8/Python.h>
#include </Users/veritas/opt/anaconda3/lib/python3.8/site-packages/numpy/core/include/numpy/arrayobject.h>

static PyObject *adds(PyObject *self, PyObject *args) {
    int arg1, arg2;
    if (!(PyArg_ParseTuple(args, "ii", &arg1, &arg2))) {
        return NULL;
    }
    return Py_BuildValue("i", arg1 * 2 + arg2 * 7);
}

static PyObject *exe(PyObject *self, PyObject *args) {
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    return PyLong_FromLong(sts);
}

static PyObject *mask(int size) {
    PyListObject *nm = (PyListObject *) PyList_New(size);
    PyObject **src;
    src = nm->ob_item;
    for (int i = 0; i < size; i++) {
        PyObject *v = PyLong_FromLong(0);
        Py_INCREF(v);
        src[i] = v;
    }
    return (PyObject *) nm;
}

void py_format(PyListObject *PyList) {
    PyObject **item;
    if ((PyList != NULL) && PyList_Check(PyList)) {
        item = PyList->ob_item;
        for (int i = 0; i < Py_SIZE(PyList); i++) {
            printf("%i ", (int) PyLong_AsLong(item[i]));
        }
        printf("\n");
    } else {
        PyErr_SetString(PyExc_TypeError, "Input must be a list.");
    }
}

void f_format(int *CArrays[], int dim0, int dim1) {
    for (int x = 0; x < dim0; x++) {
        printf("idx %d: \n", x);
        for (int y = 0; y < dim1; y++) {
            printf("%d ", CArrays[x][y]);
        }
        printf("\n");
    }
}

static PyObject *gconcat(PyListObject *a, PyObject *bb) {
    Py_ssize_t size;
    Py_ssize_t i;
    PyObject **src, **dest;
    PyListObject *np;
    if (a != NULL) {
        size = Py_SIZE(a);
    } else {
        size = Py_SIZE(bb);
    }

#define b ((PyListObject *)bb)
    if (2 * size > PY_SSIZE_T_MAX)
        return PyErr_NoMemory();
    np = (PyListObject *) PyList_New(2 * size);
    if (np == NULL) {
        return NULL;
    }
    dest = np->ob_item;
    if (a != NULL) {
        src = a->ob_item;
        for (i = 0; i < size; i++) {
            PyObject *v = src[i];
            Py_INCREF(v);
            dest[i] = v;
        }
    } else {
        for (i = 0; i < size; i++) {
            PyObject *v = PyLong_FromLong(0);
            dest[i] = v;
        }
    }
    dest = np->ob_item + size;
    if (b != NULL) {
        src = b->ob_item;
        for (i = 0; i < size; i++) {
            PyObject *v = src[i];
            Py_INCREF(v);
            dest[i] = v;
        }
    } else {
        for (i = 0; i < size; i++) {
            PyObject *v = PyLong_FromLong(0);
            dest[i] = v;
        }
    }
    return (PyObject *) np;
#undef b
}

static PyObject *psearch(PyObject *dict_u, PyObject *dict_v, PyObject *walks, int len) {
    Py_ssize_t size = PyList_Size(walks);
    PyObject **src;
    src = ((PyListObject *) walks)->ob_item;
    int CArrays[size][len * 2];
    memset(CArrays, 0, sizeof(int) * size * len * 2);

#pragma omp parallel for private(i)
    for (int i = 0; i < size; i++) {
        PyObject *pValue1, *pValue2, *pItem;
        pItem = src[i];
        pValue1 = PyDict_GetItem(dict_u, pItem);
        pValue2 = PyDict_GetItem(dict_v, pItem);

        if (pValue1 != NULL) {
            for (int j = 0; j < len; j++) {
                CArrays[i][j] = (int) PyLong_AsLong(PyList_GetItem(pValue1, j));
            }
        }

        if (pValue2 != NULL) {
            for (int j = 0; j < len; j++) {
                CArrays[i][j + len] = (int) PyLong_AsLong(PyList_GetItem(pValue2, j));
            }
        }
    }

    npy_intp Dims[2] = {size, len * 2};
    PyObject *PyArray = PyArray_SimpleNewFromData(2, Dims, NPY_INT, CArrays);
    PyArray_ENABLEFLAGS((PyArrayObject *) PyArray, NPY_ARRAY_OWNDATA);
    return PyArray;
}


static PyObject *search(PyObject *dict_u, PyObject *dict_v, PyObject *walks) {
    Py_ssize_t size = PyList_Size(walks);
    PyListObject *np = (PyListObject *) PyList_New(size);
    PyObject **src, **dest;
    PyObject *pValue1, *pValue2, *pItem;
    src = ((PyListObject *) walks)->ob_item;
    dest = np->ob_item;
    for (int i = 0; i < size; i++) {
        pItem = src[i];
        if (!PyLong_Check(pItem)) {
            PyErr_SetString(PyExc_TypeError, "Keys must be integers.");
            return NULL;
        } else {
            pValue1 = PyDict_GetItem(dict_u, pItem);
            pValue2 = PyDict_GetItem(dict_v, pItem);
            if ((pValue1 == NULL) && (pValue2 == NULL)) {
                PyErr_SetString(PyExc_TypeError, "Keys must be existed in the given dict.");
                return NULL;
            } else {
                PyObject *v = gconcat((PyListObject *) pValue1, pValue2);
                Py_INCREF(v);
                dest[i] = v;
            }
        }
    }
    return (PyObject *) np;
}

static PyObject *assemble(PyObject *dict, PyObject *key1, PyObject *key2, int njobs) {
    if (!PyDict_Check(dict)) {
        PyErr_SetString(PyExc_TypeError, "Input parameter must be two keys.");
        return NULL;
    }

    PyObject *py_v1 = PyDict_GetItem(dict, key1);
    PyObject *py_v2 = PyDict_GetItem(dict, key2);

    if ((py_v1 == NULL) || (py_v2 == NULL)) {
        PyErr_SetString(PyExc_TypeError, "Missed key in given dictionary.");
        return NULL;
    }

    if (PyTuple_Check(py_v1) && (PyTuple_Size(py_v1) == 2) && PyTuple_Check(py_v2) && (PyTuple_Size(py_v2) == 2)) {
        PyObject *py_w1 = PyTuple_GetItem(py_v1, 0);
        PyObject *py_d1 = PyTuple_GetItem(py_v1, 1);
        PyObject *py_w2 = PyTuple_GetItem(py_v2, 0);
        PyObject *py_d2 = PyTuple_GetItem(py_v2, 1);

        if (njobs <= 0) {
            PyObject *du = search(py_d1, py_d2, py_w1);
            PyObject *dv = search(py_d1, py_d2, py_w2);
            return Py_BuildValue("[O,O]", du, dv);
        } else {
            PyObject *du = psearch(py_d1, py_d2, py_w1, njobs);
            PyObject *dv = psearch(py_d1, py_d2, py_w2, njobs);
            return Py_BuildValue("[O,O]", du, dv);
        }
    } else {
        printf("With the tuple size x %i y %i", (int) PyTuple_Size(py_v1), (int) PyTuple_Size(py_v2));
        PyErr_SetString(PyExc_TypeError, "Input must be a tuple with walks and dict.");
        return NULL;
    }
}

// TODO add support to tuples

static PyObject *pgather(PyObject *self, PyObject *args, PyObject *kw) {
    PyObject *dict = NULL;
    PyObject *pair;
    int njobs = -1;
    static char *kwlist[] = {"dict", "pair", "njobs", NULL};
    if (!(PyArg_ParseTupleAndKeywords(args, kw, "O!O|i", kwlist, &PyDict_Type, &dict, &pair, &njobs))) {
        PyErr_SetString(PyExc_TypeError, "Invalid input parameters, must be a pair of keys.");
        return NULL;
    }

    PyObject **key;
    if (PyList_Check(pair)) {
        key = ((PyListObject *) pair)->ob_item;
        return assemble(dict, key[0], key[1], njobs);
    } else {
        PyErr_SetString(PyExc_TypeError, "Type of keys error, must be a list.");
        return NULL;
    }
}

static PyObject *gather(PyObject *self, PyObject *args) {
    PyObject *pair;
    PyObject *dict = NULL;
    PyObject **key;
    if (!(PyArg_ParseTuple(args, "O!O", &PyDict_Type, &dict, &pair))) {
        PyErr_SetString(PyExc_TypeError, "Input parameter must be a pair of keys.");
        return NULL;
    }
    if (PyList_Check(pair)) {
        key = ((PyListObject *) pair)->ob_item;
        return assemble(dict, key[0], key[1], 0);
    } else {
        PyErr_SetString(PyExc_TypeError, "Type of keys error, must be a list.");
        return NULL;
    }
}

static PyObject *gathers(PyObject *self, PyObject *args) {
    PyObject *pairs;
    PyObject *dict = NULL;
    PyObject **src, **dest, **key;
    PyListObject *idx, *np;
    Py_ssize_t size;

    if (!(PyArg_ParseTuple(args, "O!O", &PyDict_Type, &dict, &pairs))) {
        PyErr_SetString(PyExc_TypeError, "Input parameter must be a pair of keys.");
        return NULL;
    }

    if (PyList_Check(pairs)) {
        idx = (PyListObject *) pairs;
        size = Py_SIZE(idx);
        src = idx->ob_item;
        if (!PyList_Check(src[0])) {
            PyErr_SetString(PyExc_TypeError, "The element of the input list must be a list.");
            return NULL;
        } else {
            np = (PyListObject *) PyList_New(size);
            dest = np->ob_item;
            for (int i = 0; i < size; i++) {
                PyObject *keys = src[i];
                key = ((PyListObject *) keys)->ob_item;
                PyObject *v = assemble(dict, key[0], key[1], 0);
                Py_IncRef(v);
                dest[i] = v;
            }
            return (PyObject *) np;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "The input must be a list of pairs.");
        return NULL;
    }
}

static PyObject *gather_key(PyObject *self, PyObject *args) {
    PyObject *key1, *key2;
    PyObject *dict = NULL;

    if (!(PyArg_ParseTuple(args, "O!OO", &PyDict_Type, &dict, &key1, &key2))) {
        PyErr_SetString(PyExc_TypeError, "Input parameter must be two keys.");
        return NULL;
    }

    return assemble(dict, key1, key2, 0);
}

static PyMethodDef GComMethods[] = {
        {"add",        adds,                  METH_VARARGS, "Add ops."},
        {"gather",     gather,                METH_VARARGS, "Gather op with a pair of keys."},
        {"pgather",    (PyCFunction) pgather, METH_VARARGS | METH_KEYWORDS, "Gather op with a list of pairs (openmp)."},
        {"gathers",    gathers,               METH_VARARGS, "Gather op with a list of pairs."},
        {"gather_key", gather_key,            METH_VARARGS, "Gather op with two keys."},
        {"run",        exe,                   METH_VARARGS, "Execute a shell command."},
        {NULL, NULL, 0,                                                     NULL}
};

static char gcom_doc[] = "C extension for gather operation.";

static struct PyModuleDef gcommodule = {
        PyModuleDef_HEAD_INIT,
        "gcom_acc",   /* name of module */
        gcom_doc, /* module documentation, may be NULL */
        -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
        GComMethods
};

PyMODINIT_FUNC PyInit_gcom_acc(void) {
    return PyModule_Create(&gcommodule);
}