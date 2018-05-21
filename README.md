# python2_extensions
a collection of my attempts to extend python and numpy

When i wanted to start writing extensions for python/numpy i had a really hard time finding tutorials, examples and documentation.
Here i try to collect Tutorials and stuff i borrowed from somewhere else and self written snippets. 

## Tutorials 

A list of tutorials i found helpful. Please note that a lot of all the other stuff is copied from there.

* https://docs.python.org/2/extending/extending.html
* https://folk.uio.no/inf3330/scripting/doc/python/NumPy/Numeric/numpy-13.html
* https://www.tutorialspoint.com/python/python_further_extensions.htm
* https://docs.scipy.org/doc/numpy/user/c-info.how-to-extend.html
* https://scipy-cookbook.readthedocs.io/items/C_Extensions_NumPy_arrays.html

## Building a module

To build modules, i use [disutils](https://docs.python.org/2/distutils/). You need to set up a [setup.py](https://docs.python.org/2/distutils/setupscript.html) and then run the build with it.
```python setup.py install --user``` will build and install your module.

## Debugging

The debugging of the modules is pretty straight forward. Simply use gdb. ```gdb python2``` starts the Python2 shell in gdb. Then run it in gdb and use your module in the python shell. 
I used the interrupt signal directly in the code to "set breakpoints" which is hard otherwhise (i don't know how it would work...). I learned [here](https://stackoverflow.com/questions/4326414/set-breakpoint-in-c-or-c-code-programmatically-for-gdb-on-linux#4326474) how to do that.

## Tables

Helpful tables i stole from some tutorial mentioned earlier.

### C constants corresponding to storage types

|Constant              |element data type|
|----------------------|-----------------|
|PyArray_CHAR          |char             |
|PyArray_UBYTE         |unsigned char    |
|PyArray_SBYTE         |signed char      |
|PyArray_SHORT         |short            |
|PyArray_INT           |int              |
|PyArray_LONG          |long             |
|PyArray_FLOAT         |float            |
|PyArray_DOUBLE        |double           |
|PyArray_CFLOAT        |float[2]         |
|PyArray_CDOUBLE       |double[2]        |
|PyArray_OBJECT        |PyObject *       |

These constants can be used to compare them to inputArray->descr->type_num to determine the data type of the numpy array.

### Method Mapping Table Flags

The method mapping table is an array of PyMethodDef structs. They look like this:
```c
struct PyMethodDef {
    char *ml_name;
    PyCFunction ml_meth;
    int ml_flags;
    char *ml_doc;
};
```

|ml_flag              |meaning                    |
|---------------------|---------------------------|
|```METH_VARARGS```   |a fixed number of args     |
|```METH_KEYWORDS```  |args referenced by keywords|
|```METH_NOARGS```    |no arguments               |

```METH_VARARGS | METH_KEYWORDS``` accepts both, some fixed args and keyword args.

## Notes

Stuff i want to note: 
* don't forget to use ```import_array();``` in your module initialization method when you use numpy arrays
