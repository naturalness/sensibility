Model files
===========

Serialized model data resides here.


Structure:

    .
    ├── {model}
    │   ├── {language}.ext
    │   └── ...
    └── ...

For example, the LSTM has two weights files serialized as HDF5, and
a vocabulary, serialized as JSON. Thus, for JavaScript, there resides
the following files:

    .
    └── lstm
        ├── javascript.b.hdf5
        ├── javascript.f.hdf5
        └── javascript.vocabulary.json
