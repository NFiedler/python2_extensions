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

static PyObject* maskImg(PyObject *self, PyObject *args) {
    PyArrayObject *image, *mask;
     

    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &image, &PyArray_Type, &mask))
        return NULL;
    if (image->nd != 3 || image->descr->type_num != PyArray_UBYTE) {
        PyErr_SetString(PyExc_ValueError,
            "image must be three-dimensional and of type numpy.uint8");
        return NULL;
    }
    if (mask->nd != 3 || mask->descr->type_num != PyArray_UBYTE) {
        PyErr_SetString(PyExc_ValueError,
            "mask must be three-dimensional and of type numpy.uint8");
        return NULL;
    }
    

    npy_intp maskDims[] = {image->dimensions[0], image->dimensions[1]};
    int maskDimsNd = 2; // two dimensional output image
    PyArrayObject* maskedImg = (PyArrayObject*) PyArray_SimpleNew(maskDimsNd, maskDims, PyArray_UBYTE);
    unsigned char* pixel; 
    for (int y = 0; y < image->dimensions[0]; y++) {
        for (int x = 0; x < image->dimensions[1]; x++) {
            pixel = (unsigned char*) image->data + y * image->strides[0] + x * image->strides[1];
            if (*(mask->data + pixel[0] * mask->strides[0] + pixel[1] * mask->strides[1] + pixel[2] * mask->strides[2])) {
                *(maskedImg->data + y * maskedImg->strides[0] + x * maskedImg->strides[1]) = 255;
            }
            else {
                *(maskedImg->data + y * maskedImg->strides[0] + x * maskedImg->strides[1]) = 0;
            }
        }
    }

    //std::raise(SIGINT);
    return PyArray_Return(maskedImg);
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

static PyObject* npArray(PyObject* self, PyObject* args) {
    npy_intp dimensions[1] = {3};
    PyArrayObject* pyArray = (PyArrayObject*) PyArray_SimpleNew(1, dimensions, PyArray_INT);
    int* arraydata = (int*) pyArray->data;
    arraydata[0] = 59;
    arraydata[1] = 42;
    arraydata[2] = 7991;

    return PyArray_Return(pyArray);
}

static PyObject* npArray2D(PyObject* self, PyObject* args) {
    npy_intp dimensions[] = {2, 2};
    PyArrayObject* pyArray = (PyArrayObject*) PyArray_SimpleNew(2, dimensions, PyArray_INT);
    int* arraydata = (int*) pyArray->data;
    arraydata[0] = 59;
    arraydata[1] = 42;
    arraydata[2] = 7991;
    arraydata[2] = 7992;

    return PyArray_Return(pyArray);
}

static PyMethodDef SpamMethods[] = {
    {"add",  add, METH_VARARGS, "Add two integers"},
    {"system",  spam_system, METH_VARARGS, "Execute a shell command."},
    {"systemplus1",  spam_system_plus_1, METH_VARARGS, "Execute a shell command. Adds 1 to return value"},
    {"trace",  trace, METH_VARARGS, ""},
    {"trace3D",  trace3D, METH_VARARGS, ""},
    {"getNpArray",  npArray, METH_NOARGS, "returns a simple numpy array."},
    {"getNpArray2D",  npArray2D, METH_NOARGS, "returns a simple 2d numpy array."},
    {"maskImg", maskImg, METH_VARARGS, ""},
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
