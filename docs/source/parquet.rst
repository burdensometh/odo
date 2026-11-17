Parquet Datasets
================

``odo`` can write tabular data to a local directory of Parquet files that is
partitioned by one or more *business columns*.  This is useful for landing the
results of a CSV import or a SQL query on disk once and then reading it back
cheaply (and in parallel) with ordinary ``odo`` conversions.

The backend is implemented on top of :mod:`pyarrow` and lives in
``odo.backends.parquet``.


URIs
----

A dataset is addressed by a directory path ending in ``.parquet``::

    mydataset.parquet
    /path/to/mydataset.parquet

The ``parquet://`` scheme is accepted as well::

    parquet:///path/to/mydataset.parquet

A single Parquet *file* is also readable through the same resource; it is
treated as a one-file dataset.


Creating a dataset
------------------

Pass ``partition_on=`` to select the partition keys.  A schema can be supplied
with ``dshape=`` (or ``schema=`` as an Arrow schema)::

    >>> from odo import resource, odo  # doctest: +SKIP
    >>> ds = resource('sales.parquet',
    ...               dshape='var * {region: string, amount: int64}',
    ...               partition_on=['region'])
    >>> odo('sales.csv', ds)  # doctest: +SKIP

The same works for a SQL query::

    >>> odo(sa.select([table]), 'sales.parquet',  # doctest: +SKIP
    ...     partition_on=['region'])

Each distinct combination of partition values is stored in its own Parquet
file.  The partition columns are **not** duplicated inside the files; their
typed values live in a small manifest and are re-attached on read.


Partition values keep their type
--------------------------------

Partition values are recorded with their Arrow type.  An integer key stays an
integer::

    >>> ds = resource('sales.parquet',
    ...               dshape='var * {year: int64, amount: int64}',
    ...               partition_on=['year'])  # doctest: +SKIP
    >>> ds.read_dataframe(partitions={'year': 2020})  # doctest: +SKIP

``None`` and the empty string are distinct partition values::

    >>> ds.read_dataframe(partitions={'region': None})  # NULL partition
    >>> ds.read_dataframe(partitions={'region': ''})    # empty-string partition


Appending and replacing
-----------------------

``append`` (and therefore ``odo``/``into``) adds rows and publishes a new
version::

    >>> ds.append(frame)  # doctest: +SKIP

``replace_partition`` discards the rows of one or more partitions and writes
new ones.  The replacement data may omit the partition columns; the values are
then supplied by the mapping::

    >>> ds.replace_partition(frame, {'region': 'west'})  # doctest: +SKIP
    >>> ds.replace_partition(frame, [{'region': 'west'},
    ...                              {'region': 'east'}])  # doctest: +SKIP

Replacing a partition with an empty frame is allowed: the partition (and the
dataset schema) remain visible with zero rows.


Atomic writes
-------------

A write either publishes a complete new version or nothing at all.  New
Parquet files are written into a staging area and fsynced before the manifest
that names them is made current.  A failure therefore never exposes half of a
new partition; readers continue to see the previous version.


Consistent reads
----------------

Every read pins one complete manifest up front.  ``read_dataframe`` /
``read_table`` and the ``chunks(DataFrame)`` conversion all iterate over that
snapshot.  Publishing a new version while a reader is streaming does not
change which files the reader sees::

    >>> snapshot = ds.snapshot()          # doctest: +SKIP
    >>> for chunk in snapshot.iter_dataframes():  # doctest: +SKIP
    ...     process(chunk)

Column selection and partition filters are applied while reading::

    >>> ds.read_dataframe(columns=['amount'],
    ...                   partitions={'region': ['west', 'east']})  # doctest: +SKIP

Only files named by the snapshot's manifest are read.  Unrelated files that
happen to live in the directory (``README`` files, hidden files, stray
non-Parquet files) never appear in results.


Adopting existing Parquet directories
-------------------------------------

Pointing ``resource`` at an existing directory of Parquet files adopts those
files as the initial version.  Hive-style ``key=value`` directory names are
understood, and files whose schema is unreadable are ignored.  The adopted
files stay where they are; subsequent appends and replacements publish new
files alongside them and record the full set in a manifest.

As with the other backends, ``drop(dataset)`` removes the dataset directory
(including any files that were adopted into it).


Conversions
-----------

::

    DataFrame  -> ParquetDataset
    ParquetDataset -> DataFrame
    ParquetDataset -> chunks(DataFrame)
    ParquetDataset -> pyarrow.Table

The dataset participates in odo's normal conversion graph, so
``odo(dataset, list)``, ``odo(dataset, np.ndarray)`` and similar work as usual.
