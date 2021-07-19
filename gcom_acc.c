#define PY_SSIZE_T_CLEAN

#include "utils.h"

#include "uthash.h"

#define DEBUG 0

typedef struct item_int {
    int key;
    int val;
    UT_hash_handle hh;
} dict_int;

static int find_key_int(dict_int *maps, int key) {
    struct item_int *s;
    HASH_FIND_INT(maps, &key, s);  /* s: output pointer */
    return s ? s->val : -1;
}

/* hash of hashes */
typedef struct item {
    int key;
    int val;
    struct item *sub;
    UT_hash_handle hh;
} dict_item;

dict_item *items = NULL;

static int find_key(int key) {
    struct item *s;
    HASH_FIND_INT(items, &key, s);  /* s: output pointer */
    return s ? s->val : -1;
}

static int find_idx(int id1, int id2) {
    struct item *s, *p;
    HASH_FIND_INT(items, &id1, s);  /* s: output pointer */
    if (s != NULL) {
        HASH_FIND_INT(s->sub, &id2, p);
        return p ? p->val : 0;
    } else {
        return -1;
    }
}

void delete_all(void) {
    dict_item *item1, *item2, *tmp1, *tmp2;

    /* clean up both hash tables */
    HASH_ITER(hh, items, item1, tmp1) {
        HASH_ITER(hh, item1->sub, item2, tmp2) {
            HASH_DEL(item1->sub, item2);
            free(item2);
        }
        HASH_DEL(items, item1);
        free(item1);
    }
}

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

static void f_format(const npy_intp *dims, int *CArrays) {
    for (int x = 0; x < dims[0]; x++) {
        printf("idx %d: \n", x);
        for (int y = 0; y < dims[1]; y++) {
            printf("%d ", CArrays[x * dims[1] + y]);
        }
        printf("\n");
    }
}

static void
random_walk(int const *ptr, int const *neighs, int const *seq, int n, int num_walks, int num_steps, int seed,
            int nthread, int *walks) {
    /* https://github.com/lkskstlr/rwalk */
    if (DEBUG) {
        printf("get in  with n: %d, num_walks: %d, num_steps: %d, seed: %d, nthread: %d\n", n, num_walks, num_steps,
               seed, nthread);
    }
    if (nthread > 0) {
        omp_set_num_threads(nthread);
    }
#pragma omp parallel
    {
        int thread_num = omp_get_thread_num();
        unsigned int private_seed = (unsigned int) (seed + thread_num);
#pragma omp for
        for (int i = 0; i < n; i++) {
            int offset, num_neighs;
            for (int walk = 0; walk < num_walks; walk++) {
                int curr = seq[i];
                offset = i * num_walks * (num_steps + 1) + walk * (num_steps + 1);
                walks[offset] = curr;
                for (int step = 0; step < num_steps; step++) {
                    num_neighs = ptr[curr + 1] - ptr[curr];
                    if (num_neighs > 0) {
                        curr = neighs[ptr[curr] + (rand_r(&private_seed) % num_neighs)];
                    }
                    walks[offset + step + 1] = curr;
                }
            }
        }
    }
}

static void
random_walk_wo(int const *ptr, int const *neighs, int const *seq, int n, int num_walks, int num_steps, int seed,
               int nthread, int *walks) {
    if (nthread > 0) {
        omp_set_num_threads(nthread);
    }
#pragma omp parallel
    {
        int thread_num = omp_get_thread_num();
        unsigned int private_seed = (unsigned int) (seed + thread_num);

#pragma omp for
        for (int i = 0; i < n; i++) {
            int offset, num_neighs;

            int num_hop1 = ptr[seq[i] + 1] - ptr[seq[i]];
            int rseq[num_hop1];
            if (num_hop1 > num_walks) {
                // https://www.programmersought.com/article/71554044511/
                int s, t;
                for (int j = 0; j < num_hop1; j++)
                    rseq[j] = j;
                for (int k = 0; k < num_walks; k++) {
                    s = rand_r(&private_seed) % (num_hop1 - k) + k;
                    t = rseq[k];
                    rseq[k] = rseq[s];
                    rseq[s] = t;
                }
            }

            for (int walk = 0; walk < num_walks; walk++) {
                int curr = seq[i];
                offset = i * num_walks * (num_steps + 1) + walk * (num_steps + 1);
                walks[offset] = curr;
                if (num_hop1 < 1){
                    walks[offset + 1] = curr;
                }
                else if (num_hop1 <= num_walks) {
                    curr = neighs[ptr[curr] + walk % num_hop1];
                    walks[offset + 1] = curr;
                } else {
                    curr = neighs[ptr[curr] + rseq[walk]];
                    walks[offset + 1] = curr;
                }
                for (int step = 1; step < num_steps; step++) {
                    num_neighs = ptr[curr + 1] - ptr[curr];
                    if (num_neighs > 0) {
                        curr = neighs[ptr[curr] + (rand_r(&private_seed) % num_neighs)];
                    }
                    walks[offset + step + 1] = curr;
                }
            }
        }
    }
}


void dis_encoding(int const *arr, int idx, int num_walks, int num_steps, PyArrayObject **out) {
    PyArrayObject *oarr1 = NULL, *oarr2 = NULL;
    dict_int *mapping = NULL;
    int offset = idx * num_walks * (num_steps + 1);

    // setup root node
    dict_int *root = malloc(sizeof(*root));
    root->key = arr[offset];
    root->val = 0;
    HASH_ADD_INT(mapping, key, root);

    // setup the rest unique node
    int count = 1;
    for (int i = 1; i < num_steps + 1; i++) {
        for (int j = 0; j < num_walks; j++) {
            int token = arr[offset + j * (num_steps + 1) + i];
            if (find_key_int(mapping, token) < 0) {
                dict_int *k = malloc(sizeof(*k));
                // walk starts from each node (main key)
                k->key = token;
                k->val = count;
                HASH_ADD_INT(mapping, key, k);
                count++;
            }
        }
    }
    int num_keys = HASH_COUNT(mapping);
//     printf("There are %u many unique keys.\n", num_keys);

    // create a new array
    npy_intp odims1[2] = {num_keys, num_steps + 1};
    oarr1 = (PyArrayObject *) PyArray_ZEROS(2, odims1, NPY_INT, 0);
    if (!oarr1) PyErr_SetString(PyExc_TypeError, "output error.");
    int *Coarr1 = (int *) PyArray_DATA(oarr1);

    npy_intp odims2[1] = {num_keys};
    oarr2 = (PyArrayObject *) PyArray_SimpleNew(1, odims2, NPY_INT);
    if (!oarr2) PyErr_SetString(PyExc_TypeError, "output error.");
    int *Coarr2 = (int *) PyArray_DATA(oarr2);

    Coarr1[0] = num_walks;

    for (int i = 1; i < num_steps + 1; i++) {
        for (int j = 0; j < num_walks; j++) {
            int anchor = find_key_int(mapping, arr[offset + j * (num_steps + 1) + i]);
            Coarr1[anchor * (num_steps + 1) + i]++;
        }
    }

    // free mem
    dict_int *cur_item, *tmp;
    HASH_ITER(hh, mapping, cur_item, tmp) {
        Coarr2[cur_item->val] = cur_item->key;
        HASH_DEL(mapping, cur_item);  /* delete it (users advances to next) */
        free(cur_item);               /* free it */
    }
    out[2 * idx] = oarr2, out[2 * idx + 1] = oarr1;
}

static PyObject *np_walk(PyObject *self, PyObject *args, PyObject *kws) {
    PyObject *arg1 = NULL, *arg2 = NULL, *query = NULL;
    PyArrayObject *ptr = NULL, *neighs = NULL, *seq = NULL, *oarr = NULL, *obj_arr = NULL;
    int num_walks = 100, num_steps = 3, seed = 111413, nthread = -1, re = -1;
    int n;

    static char *kwlist[] = {"ptr", "neighs", "query", "num_walks", "num_steps", "nthread", "seed", "replacement", NULL};
    if (!(PyArg_ParseTupleAndKeywords(args, kws, "OOO|iiiip", kwlist, &arg1, &arg2, &query, &num_walks, &num_steps,
                                      &nthread, &seed, &re))) {
        PyErr_SetString(PyExc_TypeError, "input parsing error.");
        return NULL;
    }

    /* handle walks (numpy array) */
    ptr = (PyArrayObject *) PyArray_FROM_OTF(arg1, NPY_INT, NPY_ARRAY_IN_ARRAY);
    if (!ptr) return NULL;
    int *Cptr = PyArray_DATA(ptr);

    neighs = (PyArrayObject *) PyArray_FROM_OTF(arg2, NPY_INT, NPY_ARRAY_IN_ARRAY);
    if (!neighs) return NULL;
    int *Cneighs = PyArray_DATA(neighs);

    seq = (PyArrayObject *) PyArray_FROM_OTF(query, NPY_INT, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    if (!seq) return NULL;
    int *Cseq = PyArray_DATA(seq);

    n = (int) PyArray_SIZE(seq);
//     printf("size of input %d\n", n);

    npy_intp odims[2] = {n, num_walks * (num_steps + 1)};
    oarr = (PyArrayObject *) PyArray_SimpleNew(2, odims, NPY_INT);
    if (oarr == NULL) goto fail;
    int *Coarr = (int *) PyArray_DATA(oarr);
//     printf("Dims of output: %d, %d \n", (int) odims[0], (int) odims[1]);

    npy_intp obj_dims[2] = {n, 2};
    obj_arr = (PyArrayObject *) PyArray_SimpleNew(2, obj_dims, NPY_OBJECT);
    if (obj_arr == NULL) goto fail;
    PyArrayObject **Cobj_arr = PyArray_DATA(obj_arr);

    if (re > 0) {
//         printf("Using no replacement sampling for the 1-hop.\n");
        random_walk_wo(Cptr, Cneighs, Cseq, n, num_walks, num_steps, seed, nthread, Coarr);
    } else {
        random_walk(Cptr, Cneighs, Cseq, n, num_walks, num_steps, seed, nthread, Coarr);
    }

    int k;
#pragma omp for private(k)
    for (k = 0; k < n; k++) {
        dis_encoding(Coarr, k, num_walks, num_steps, Cobj_arr);
    }

    Py_DECREF(ptr);
    Py_DECREF(neighs);
    Py_DECREF(seq);
    return Py_BuildValue("[O,O]", PyArray_Return(oarr), PyArray_Return(obj_arr));

    fail:
    Py_XDECREF(ptr);
    Py_XDECREF(neighs);
    Py_XDECREF(seq);
    PyArray_XDECREF_ERR(oarr);
    PyArray_XDECREF_ERR(obj_arr);
    return NULL;
}

static PyObject *np_gather(PyObject *self, PyObject *args, PyObject *kws) {
    PyObject *arg1 = NULL, *arg2 = NULL, *query = NULL, *seq = NULL, **src;
    PyArrayObject *arr = NULL, *iarr = NULL, *oarr = NULL;
    int nthread = -1;

    static char *kwlist[] = {"walk", "key", "query", "nthread", NULL};
    if (!(PyArg_ParseTupleAndKeywords(args, kws, "OOO|i", kwlist, &arg1, &arg2, &query, &nthread))) {
        PyErr_SetString(PyExc_TypeError, "input parsing error.");
        return NULL;
    }

    /* handle walks (numpy array) */
    arr = (PyArrayObject *) PyArray_FROM_OTF(arg1, NPY_INT, NPY_ARRAY_IN_ARRAY);
    if (!arr) return NULL;
    npy_intp *arr1_dims = PyArray_DIMS(arr);
    int stride;
    if (PyArray_NDIM(arr) > 2) {
        stride = (int) (arr1_dims[1] * arr1_dims[2]);
    } else {
        stride = (int) arr1_dims[1];
    }
    int *Carr = (int *) PyArray_DATA(arr);

    /* handle keys (a list of tuple) */
    seq = PySequence_Fast(arg2, "argument must be iterable");
    if (!seq) return NULL;
    if (PySequence_Fast_GET_SIZE(seq) != arr1_dims[0]) {
        PyErr_SetString(PyExc_TypeError, "dims do not match between walks and keys.");
        return NULL;
    }

    /* handle queries (numpy array) */
    iarr = (PyArrayObject *) PyArray_FROM_OTF(query, NPY_INT, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    if (!iarr) return NULL;
    npy_intp *iarr_dims = PyArray_DIMS(iarr);
    int *Ciarr = (int *) PyArray_DATA(iarr);

    if (DEBUG) {
        printf("Dims of query: %d, %d\n", (int) iarr_dims[0], (int) iarr_dims[1]);
    }

    /* initialize the hashtable */
    items = NULL;
    int idx = 1;
    src = PySequence_Fast_ITEMS(seq);

    /* build two level hash table: 1) reindex main keys 2) hash unique node idx associated with each key */
    for (int i = 0; i < arr1_dims[0]; i++) {
        /* make initial element */
        dict_item *k = malloc(sizeof(*k));
        // walk starts from each node (main key)
        k->key = Carr[i * stride];
        k->sub = NULL;
        k->val = i;
        HASH_ADD_INT(items, key, k);

        PyObject *item = PySequence_Fast(src[i], "argument must be iterable");
        int item_size;
        if (!PyArray_CheckExact(item)) {
            item_size = PySequence_Fast_GET_SIZE(item);

            for (int j = 0; j < item_size; j++) {
                /* add a sub hash table off this element */
                dict_item *w = malloc(sizeof(*w));
                w->key = (int) PyLong_AsLong(PySequence_Fast_GET_ITEM(item, j));
                w->sub = NULL;
                w->val = idx;
                HASH_ADD_INT(k->sub, key, w);
                idx++;
            }
        } else {
            item_size = PyArray_Size(item);

            for (int j = 0; j < item_size; j++) {
                /* add a sub hash table off this element */
                dict_item *w = malloc(sizeof(*w));
                w->key = (*(int *) PyArray_GETPTR1((PyArrayObject *)item, j));
                w->sub = NULL;
                w->val = idx;
                HASH_ADD_INT(k->sub, key, w);
                idx++;
            }
        }
//         printf("there are %u items\n", HASH_COUNT(k->sub));
    }

    /* allocate a new return numpy array */
    npy_intp odims[2] = {2, iarr_dims[0] * 2 * stride};
    oarr = (PyArrayObject *) PyArray_SimpleNew(2, odims, NPY_INT);
    if (oarr == NULL) goto fail;
    int *Coarr = (int *) PyArray_DATA(oarr);

    if (DEBUG) {
        printf("Dims of output: %d, %d\n", (int) odims[0], (int) odims[1]);
    }

    if (nthread > 0) {
        omp_set_num_threads(nthread);
    }
    int x;
#pragma omp parallel for private(x)
    for (x = 0; x < iarr_dims[0]; x++) {
        int qid = 2 * x;
        int key1 = Ciarr[qid], key2 = Ciarr[qid + 1];
        int id1 = find_key(key1), id2 = find_key(key2);
//        printf("key1 %d, key2 %d id1 %d, id2 %d\n", key1, key2, id1, id2);
        for (int y = 0; y < 2 * stride; y += 2) {
            Coarr[qid * stride + y] = find_idx(key1, Carr[id1 * stride + y / 2]);
            Coarr[qid * stride + y + 1] = find_idx(key2, Carr[id1 * stride + y / 2]);
            Coarr[odims[1] + qid * stride + y] = find_idx(key1, Carr[id2 * stride + y / 2]);
            Coarr[odims[1] + qid * stride + y + 1] = find_idx(key2, Carr[id2 * stride + y / 2]);
        }
//         for (y = 0; y < 2 * stride; y++) {
//             if (y % 2 == 0) {
//                 Coarr[qid * stride + y] = find_idx(key1, Carr1[id1 * stride + y / 2]);
//                 Coarr[odims[1] + qid * stride + y] = find_idx(key1, Carr1[id2 * stride + y / 2]);
//             } else {
//                 Coarr[qid * stride + y] = find_idx(key2, Carr1[id1 * stride + y / 2]);
//                 Coarr[odims[1] + qid * stride + y] = find_idx(key2, Carr1[id2 * stride + y / 2]);
//             }
//         }
    }

    Py_DECREF(arr);
    Py_DECREF(iarr);
    Py_DECREF(seq);
    delete_all();
    return PyArray_Return(oarr);

    fail:
    Py_XDECREF(arr);
    Py_DECREF(iarr);
    Py_XDECREF(seq);
    delete_all();
    PyArray_XDECREF_ERR(oarr);
    return NULL;
}

static PyMethodDef GComMethods[] = {
        {"add",      adds,                    METH_VARARGS, "Add ops."},
        {"run",      exe,                     METH_VARARGS, "Execute a shell command."},
        {"gather",   (PyCFunction) np_gather, METH_VARARGS | METH_KEYWORDS,
                        "Gather op with a list of pairs (numpy, openmp)."},
        {"ran_walk", (PyCFunction) np_walk,   METH_VARARGS | METH_KEYWORDS,
                        "Random walks with distance encoding (numpy, openmp)."},
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
    import_array();
    return PyModule_Create(&gcommodule);
}