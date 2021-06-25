#define PY_SSIZE_T_CLEAN

#include <Python/Python.h>

static PyObject *adds(PyObject *self, PyObject *args) {
    int arg1, arg2;
    if (!(PyArg_ParseTuple(args, "ii", &arg1, &arg2))) {
        return NULL;
    }
    return Py_BuildValue("i", arg1 + arg2 * 10);
}

static PyObject *exe(PyObject *self, PyObject *args) {
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    return PyLong_FromLong(sts);
}

void pformat(PyListObject *a) {
    PyObject **item;
    if ((a != NULL) && PyList_Check(a)) {
        item = a->ob_item;
        for (int i = 0; i < Py_SIZE(a); i++) {
            printf("%li ", PyLong_AsLong(item[i]));
        }
        printf("\n");
    } else {
        PyErr_SetString(PyExc_TypeError, "parameter must be a list.");
    }
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

static PyObject *gconcat(PyListObject *a, PyObject *bb) {
    Py_ssize_t size;
    PyObject **dest;
    PyListObject *np;
    if (a != NULL) {
        size = Py_SIZE(a);
    } else {
        size = Py_SIZE(bb);
    }

//    pformat((PyListObject *) mask((int) size));

    PyObject *defaults = mask((int) size);

#define b ((PyListObject *)bb)
    if (2 * size > PY_SSIZE_T_MAX)
        return PyErr_NoMemory();
    np = (PyListObject *) PyList_New(2);
    if (np == NULL) {
        return NULL;
    }
    dest = np->ob_item;

    PyObject *va;
    if (a != NULL) {
        va = (PyObject *) a;
    } else {
        va = defaults;
    }
    Py_INCREF(va);
    dest[0] = va;

    PyObject *vb;
    if (b != NULL) {
        vb = (PyObject *) b;
    } else {
        vb = defaults;
    }
    Py_INCREF(vb);
    dest[1] = vb;

    return (PyObject *) np;
#undef b
}

static PyObject *pair(PyObject *dict, PyObject *key1, PyObject *key2) {
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


        PyObject *pItem;
        PyObject *pValue1, *pValue2;
        Py_ssize_t n1, n2;
        PyListObject *np;
        PyObject **dest;

        n1 = PyList_Size(py_w1);
        n2 = PyList_Size(py_w2);

        np = (PyListObject *) PyList_New(n1 + n2);
        dest = np->ob_item;
        for (int i = 0; i < n1 + n2; i++) {
            if (i < n1) {
                pItem = PyList_GetItem(py_w1, i);
            } else {
                pItem = PyList_GetItem(py_w2, i - n1);
            }

            if (!PyLong_Check(pItem)) {
                PyErr_SetString(PyExc_TypeError, "Keys must be integers.");
                return NULL;
            } else {
                pValue1 = PyDict_GetItem(py_d1, pItem);
                pValue2 = PyDict_GetItem(py_d2, pItem);
                if ((pValue1 == NULL) && (pValue2 == NULL)) {
                    PyErr_SetString(PyExc_TypeError, "Keys must be existed in one dict.");
                    return NULL;
                } else {
                    PyObject *v = gconcat((PyListObject *) pValue1, pValue2);
                    Py_INCREF(v);
                    dest[i] = v;
//                     pformat(v);
                }
            }
        }
        return (PyObject *) np;
    } else {
        printf("With the tuple size x %i y %i", (int) PyTuple_Size(py_v1), (int) PyTuple_Size(py_v2));
        PyErr_SetString(PyExc_TypeError, "Input must be a tuple with walks and dict.");
        return NULL;
    }
}

// TODO add support to tuples

static PyObject *gather(PyObject *self, PyObject *args) {
    PyObject *pairs;
    PyObject *dict = NULL;
    PyObject **key;
    if (!(PyArg_ParseTuple(args, "O!O", &PyDict_Type, &dict, &pairs))) {
        PyErr_SetString(PyExc_TypeError, "Input parameter must be a pair of keys.");
        return NULL;
    }
    if (PyList_Check(pairs)) {
        key = ((PyListObject *) pairs)->ob_item;
        return pair(dict, key[0], key[1]);
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
        np = (PyListObject *) PyList_New(size);
        dest = np->ob_item;
        if (!PyList_Check(src[0])) {
            PyErr_SetString(PyExc_TypeError, "Input parameter must be a list of pairs.");
            return NULL;
        }
        for (int i = 0; i < size; i++) {
            PyObject *keys = src[i];
            key = ((PyListObject *) keys)->ob_item;
            PyObject *v = pair(dict, key[0], key[1]);
            Py_IncRef(v);
            dest[i] = v;
        }
        return (PyObject *) np;
    } else {
        PyErr_SetString(PyExc_TypeError, "Type of keys error, must be a list.");
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

    return pair(dict, key1, key2);
}

static PyMethodDef GComMethods[] = {
        {"add",        adds,       METH_VARARGS, "Add ops."},
        {"gather",     gather,     METH_VARARGS, "Gather op with a pair of keys."},
        {"gathers",    gathers,    METH_VARARGS, "Gather op with a list of pairs."},
        {"gather_key", gather_key, METH_VARARGS, "Gather op with two keys."},
        {"run",        exe,        METH_VARARGS, "Execute a shell command."},
        {NULL, NULL, 0, NULL}
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