#include <Python.h>
#include "numpy/arrayobject.h"
#include <iostream>
#include <csignal>

static PyObject* spam_system(PyObject* self, PyObject* args) {
    const char* command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command)) {
        return NULL;
    }
    sts = system(command);
    return Py_BuildValue("i", sts);
}

static PyObject* add(PyObject* self, PyObject* args) {
    int a, b;
    if (!PyArg_ParseTuple(args, "ii", &a, &b)) {
      return NULL;
    }
    return Py_BuildValue("i", a + b);
}

static PyObject* trace(PyObject *self, PyObject *args) {
    PyArrayObject *array;
    double sum;
    int i, n;
     

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &array))
        return NULL;
    if (array->nd != 2 || array->descr->type_num != PyArray_DOUBLE) {
        PyErr_SetString(PyExc_ValueError,
            "array must be two-dimensional and of type float");
        return NULL;
    }
     
    n = array->dimensions[0];
    if (n > array->dimensions[1])
        n = array->dimensions[1];
    sum = 0.;
    for (i = 0; i < n; i++)
        sum += *(double *)(array->data + i*array->strides[0] + i*array->strides[1]);

    return PyFloat_FromDouble(sum);
}

static PyObject* trace3D(PyObject *self, PyObject *args) {
    PyArrayObject *array;
    double sum;
    int i, n;
     

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &array))
        return NULL;
    if (array->nd != 3 || array->descr->type_num != PyArray_DOUBLE) {
        PyErr_SetString(PyExc_ValueError,
            "array must be three-dimensional and of type float");
        return NULL;
    }
     
    n = array->dimensions[0];
    if (n > array->dimensions[1])
        n = array->dimensions[1];
    if (n > array->dimensions[2])
        n = array->dimensions[2];
    sum = 0.;
    for (i = 0; i < n; i++)
        sum += *(double *)(array->data + i*array->strides[0] + i*array->strides[1] + i*array->strides[2]);

    return PyFloat_FromDouble(sum);
}

static PyObject* spam_system_plus_1(PyObject* self, PyObject* args) {
    const char* command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command)) {
        return NULL;
    }
    sts = system(command);
    sts ++;
    return Py_BuildValue("i", sts);
}

static PyMethodDef SpamMethods[] = {
    {"add",  add, METH_VARARGS, "Add two integers"},
    {"system",  spam_system, METH_VARARGS, "Execute a shell command."},
    {"systemplus1",  spam_system_plus_1, METH_VARARGS, "Execute a shell command. Adds 1 to return value"},
    {"trace",  trace, METH_VARARGS, ""},
    {"trace3D",  trace3D, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initspam(void) {
    import_array();
    (void) Py_InitModule("spam", SpamMethods);
}

int main(int argc, char *argv[]) {

    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(argv[0]);

    /* Initialize the Python interpreter.  Required. */
    Py_Initialize();

    initspam();

    return 0;
}
