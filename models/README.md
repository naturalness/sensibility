Model files
===========

Serialized model data resides here.


Structure:

    .
    ├── {language}.vocabulary.json
    ├── {model}
    │   ├── {language}.ext
    │   └── ...
    └── ...

For example, the LSTM has two weights files serialized as HDF5. The
JavaScript vocabulary, is serialized as JSON. Thus, for JavaScript,
there resides the following files:

    .
    ├── javascript.vocabulary.json
    └── lstm
        ├── javascript.b.hdf5
        └── javascript.f.hdf5
