Models
======

When you type `make experiments`, the many different model configurations show
up here.

Directory structure:

    .
    └── {language}/
        └── {config-id}.(forwards|backwards)
            ├── intermediate-{val-xentropy}-{epoch}.hdf5
            ├── manifest.json
            └── progress.csv

manifest.json
 ~ Contains pertinent configuration information for replication 

progress.csv
 ~ Contains epoch-by-epoch model accuracy and loss

intermediate-{val-xentropy}-{epoch}.hdf5
 ~ An intermediate model file, created after every epoch
 ~ Intended to be sorted by lowest the validation cross entropy
