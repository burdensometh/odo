"""Local, partitioned, Parquet-backed datasets.

A dataset is a directory of immutable Parquet files plus a small manifest that
records the exact Arrow schema, the partition keys, and the complete list of
files in every published version.

Design goals
------------

``ParquetDataset`` (also exported as ``Dataset`` and ``LocalDataset``):

* **Atomic publishes** - a write either publishes a complete new version or
  nothing at all.  A failed write never exposes a half-written partition.
* **Snapshot reads** - a reader captures one complete manifest up front.  A
  concurrent publish does not change the files that reader sees.
* **Typed partition keys** - partition values keep their Arrow type, so an
  integer key stays an integer and ``None`` stays distinct from ``""``.
* **Schema for empty data** - creating or replacing a partition with an empty
  frame still records the schema and the partition, so later reads return
  typed zero-row data.
* **Unrelated files ignored** - only files listed in the manifest are read.

Layout::

    mydataset/
        _odo_dataset/
            CURRENT               # name of the published manifest
            LOCK                  # advisory writer lock
            manifests/<id>.json   # immutable, complete version manifests
            staging/              # temporary files during a publish
        data/
            <id>.parquet

Parquet files are never modified or deleted, which is what makes snapshot
reads and atomic publishes possible.
"""

from __future__ import absolute_import, division, print_function

import base64
import contextlib
import datetime
import errno
import json
import os
import shutil
import threading
import time
import uuid

import numpy as np
import pandas as pd

import datashape
from datashape import (
    Categorical,
    DataShape,
    Date,
    DateTime,
    Decimal,
    Null,
    Option,
    Record,
    String,
    Time,
    TimeDelta,
    discover,
)
from datashape.predicates import isrecord

import pyarrow as pa
import pyarrow.parquet as pq

from ..append import append
from ..chunks import Chunks, chunks
from ..convert import convert, ooc_types
from ..drop import drop
from ..resource import resource


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METADATA_DIRNAME = '_odo_dataset'
DATA_DIRNAME = 'data'
MANIFEST_DIRNAME = 'manifests'
STAGING_DIRNAME = 'staging'
CURRENT_FILENAME = 'CURRENT'
LOCK_FILENAME = 'LOCK'
MANIFEST_FORMAT = 1
DEFAULT_COMPRESSION = 'snappy'
DEFAULT_CHUNKSIZE = 2 ** 16
NULL_SENTINEL = '\x00<null>\x00'
#: Prefix used to escape literal values that would otherwise look like
#: ``NULL_SENTINEL`` or an escaped value.
ESCAPE_PREFIX = '\x01'


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def _makedirs(path):
    if path and not os.path.isdir(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def _fsync_dir(path):
    """Best-effort ``fsync`` of a directory so a rename becomes durable."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _replace_file(src, dst):
    """Atomically replace ``dst`` with ``src`` (POSIX-atomic on both Pythons)."""
    replace = getattr(os, 'replace', None)
    if replace is not None:
        replace(src, dst)
    else:
        os.rename(src, dst)


def _atomic_write(path, data):
    """Write ``data`` (bytes) to ``path`` atomically (write-then-rename)."""
    directory = os.path.dirname(path)
    _makedirs(directory)
    tmp = os.path.join(
        directory, '.%s.%s.tmp' % (os.path.basename(path), uuid.uuid4().hex),
    )
    try:
        with open(tmp, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        _replace_file(tmp, path)
        _fsync_dir(directory)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _remove_tree(path):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def _materialise(data):
    """Turn one-shot iterables into a reusable frame.

    ``append`` inspects the data more than once (to infer a schema and then to
    write it), so a plain generator or iterator must be drained up front.
    ``Chunks`` objects are left alone; they are handled by a dedicated
    ``append`` registration.
    """
    if data is None:
        return data
    if isinstance(data, (pd.DataFrame, pa.Table, pa.RecordBatch, Chunks)):
        return data
    if isinstance(data, (dict, list, tuple)):
        return data
    if hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):
        return pd.DataFrame(list(data))
    return data


def _is_null(value):
    if value is None or value is pd.NaT:
        return True
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return True
    return False


# ---------------------------------------------------------------------------
# Datashape <-> Arrow schema
# ---------------------------------------------------------------------------

_PRIMITIVE_ARROW_TYPES = {
    'bool': pa.bool_(),
    'int8': pa.int8(),
    'int16': pa.int16(),
    'int32': pa.int32(),
    'int64': pa.int64(),
    'uint8': pa.uint8(),
    'uint16': pa.uint16(),
    'uint32': pa.uint32(),
    'uint64': pa.uint64(),
    'float32': pa.float32(),
    'float64': pa.float64(),
    'complex[float32]': pa.float32(),
    'complex[float64]': pa.float64(),
}

_TIME_UNITS = frozenset({'s', 'ms', 'us', 'ns'})


def _time_unit(unit):
    return unit if unit in _TIME_UNITS else 'us'


def arrow_type_for(measure):
    """Translate a datashape *measure* into an Arrow type.

    ``Option`` does not change the physical Arrow type; nullability lives on
    the field.  ``string``, ``date``, ``time``, ``datetime``, ``timedelta``,
    ``decimal`` and ``categorical`` map to their natural Arrow counterparts.
    """
    if isinstance(measure, Option):
        return arrow_type_for(measure.ty)
    if isinstance(measure, Null):
        return pa.null()
    if isinstance(measure, String):
        return pa.string()
    if isinstance(measure, Categorical):
        return pa.dictionary(pa.int32(), pa.string())
    if isinstance(measure, DateTime):
        unit = _time_unit(getattr(measure, 'unit', None) or 'us')
        return pa.timestamp(unit, measure.tz)
    if isinstance(measure, Date):
        return pa.date32()
    if isinstance(measure, Time):
        return pa.time64('us')
    if isinstance(measure, TimeDelta):
        return pa.duration(_time_unit(measure.unit))
    if isinstance(measure, Decimal):
        return pa.decimal128(int(measure.precision or 38),
                             int(measure.scale or 0))
    name = getattr(measure, 'name', None)
    if name in _PRIMITIVE_ARROW_TYPES:
        return _PRIMITIVE_ARROW_TYPES[name]
    if name == 'object':
        return pa.string()
    raise TypeError('cannot represent datashape measure %r as Arrow' %
                    (measure,))


def arrow_schema_from_dshape(dshape, partition_on=()):
    """Build an Arrow schema from a datashape.

    Partition keys are always nullable, because a partition may hold a
    ``None`` value.  Keys missing from the datashape are added as strings so
    that ``resource(uri, dshape=..., partition_on=[...])`` works even when the
    caller only describes the business columns.
    """
    dtype = datashape.dshape(dshape)
    measure = dtype.measure
    if isrecord(measure):
        pairs = list(zip(measure.names, measure.types))
    elif len(measure) == 0:
        pairs = []
    else:
        raise ValueError('datasets require a record datashape, got %s' % dtype)

    partition_on = list(partition_on)
    fields = []
    present = set()
    for name, typ in pairs:
        name = str(name)
        present.add(name)
        nullable = isinstance(typ, Option) or name in partition_on
        fields.append(pa.field(name, arrow_type_for(typ), nullable))
    for name in partition_on:
        if name not in present:
            fields.append(pa.field(name, pa.string(), True))
    return pa.schema(fields)


def _measure_from_arrow_type(typ):
    if pa.types.is_null(typ):
        return datashape.null
    if pa.types.is_boolean(typ):
        return datashape.bool_
    if pa.types.is_integer(typ) or pa.types.is_floating(typ):
        return getattr(datashape, str(np.dtype(typ.to_pandas_dtype())))
    if pa.types.is_string(typ) or pa.types.is_large_string(typ):
        return datashape.string
    if pa.types.is_dictionary(typ):
        return Categorical([], type=_measure_from_arrow_type(typ.value_type))
    if pa.types.is_timestamp(typ):
        return DateTime(tz=typ.tz)
    if pa.types.is_date(typ):
        return Date()
    if pa.types.is_time(typ):
        return Time()
    if pa.types.is_duration(typ):
        return TimeDelta(unit='us')
    if pa.types.is_decimal(typ):
        return Decimal(precision=typ.precision, scale=typ.scale)
    return datashape.object_


def dshape_from_arrow_schema(schema):
    """Convert an Arrow schema into a tabular datashape."""
    fields = []
    for field in schema:
        typ = _measure_from_arrow_type(field.type)
        if field.nullable and not isinstance(typ, Null):
            typ = Option(typ)
        fields.append((field.name, typ))
    return datashape.var * Record(fields)


# ---------------------------------------------------------------------------
# Frame <-> Arrow conversion
# ---------------------------------------------------------------------------

def _coerce_python_values(column, typ):
    """Coerce a pandas Series into Python values for ``pa.array``."""
    if pa.types.is_string(typ) or pa.types.is_dictionary(typ):
        return [None if _is_null(v) else str(v) for v in column]
    if pa.types.is_integer(typ):
        out = []
        for v in column:
            if _is_null(v):
                out.append(None)
            elif isinstance(v, str):
                out.append(int(float(v)))
            elif isinstance(v, (float, np.floating)):
                if not float(v).is_integer():
                    raise ValueError(
                        'cannot store non-integral value %r in an integer '
                        'column' % (v,),
                    )
                out.append(int(v))
            else:
                out.append(int(v))
        return out
    if pa.types.is_floating(typ):
        out = []
        for v in column:
            if _is_null(v):
                out.append(None)
            elif isinstance(v, str):
                out.append(float(v))
            else:
                out.append(float(v))
        return out
    if pa.types.is_boolean(typ):
        return [None if _is_null(v) else bool(v) for v in column]
    if pa.types.is_timestamp(typ):
        return [None if _is_null(v) else pd.Timestamp(v) for v in column]
    if pa.types.is_date(typ):
        out = []
        for v in column:
            out.append(None if _is_null(v) else pd.Timestamp(v).date())
        return out
    if pa.types.is_time(typ):
        return [None if _is_null(v) else pd.Timestamp(v).time() for v in column]
    if pa.types.is_duration(typ):
        return [None if _is_null(v) else pd.Timedelta(v) for v in column]
    if pa.types.is_decimal(typ):
        out = []
        for v in column:
            out.append(None if _is_null(v) else Decimal(str(v)))
        return out
    return [None if _is_null(v) else v for v in column]


def _frame_to_table(frame, schema):
    """Convert a pandas ``DataFrame`` to an Arrow table with ``schema``.

    Missing columns are filled with typed nulls.  If pyarrow's fast path
    cannot convert a column (for example numeric strings), we fall back to a
    per-column, value-by-value coercion.
    """
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    names = [field.name for field in schema]
    n = len(frame)
    unnamed = [c for c in frame.columns if c not in names]
    if (len(unnamed) == len(frame.columns) and len(names) == len(frame.columns)
            and not any(c in names for c in frame.columns)):
        # Positional data such as ``[('west', 1), ('east', 2)]``: align the
        # integer column labels with the schema's column names.
        frame = frame.copy()
        frame.columns = names
    data = {}
    for field in schema:
        if field.name in frame.columns:
            column = frame[field.name]
            if len(column) != n:
                raise ValueError('column %r has %d rows but the frame has %d' %
                                 (field.name, len(column), n))
        else:
            column = pd.Series([None] * n, dtype=object)
        data[field.name] = column
    ordered = pd.DataFrame(data, columns=names)
    if len(names) == 0:
        return pa.Table.from_arrays([], schema=pa.schema([]))
    try:
        return pa.Table.from_pandas(ordered, schema=schema,
                                    preserve_index=False)
    except (pa.ArrowInvalid, pa.ArrowTypeError, ValueError, TypeError):
        arrays = []
        for field in schema:
            values = _coerce_python_values(ordered[field.name], field.type)
            arrays.append(pa.array(values, type=field.type))
        return pa.Table.from_arrays(arrays, schema=schema)


def _table_to_dataframe(table):
    if table.num_columns == 0:
        return pd.DataFrame(index=pd.RangeIndex(table.num_rows))
    return table.to_pandas().reset_index(drop=True)


def _as_table(data, schema):
    """Coerce tabular ``data`` into an Arrow table with ``schema``."""
    if isinstance(data, pa.Table):
        table = data
    elif isinstance(data, pa.RecordBatch):
        table = pa.Table.from_batches([data])
    elif isinstance(data, pd.DataFrame):
        table = _frame_to_table(data, schema)
    else:
        table = _frame_to_table(pd.DataFrame(data), schema)
    if table.schema != schema:
        table = _cast_table(table, schema)
    return table


def _cast_table(table, schema):
    """Cast ``table`` to ``schema``, tolerating all-null source columns."""
    if table.schema == schema:
        return table
    arrays = []
    for field in schema:
        if field.name in table.column_names:
            column = table[field.name]
            if column.type != field.type:
                try:
                    column = column.cast(field.type)
                except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
                    column = pa.array(
                        _coerce_python_values(column.to_pandas(), field.type),
                        type=field.type,
                    )
            arrays.append(column)
        else:
            arrays.append(pa.nulls(table.num_rows, type=field.type))
    return pa.Table.from_arrays(arrays, schema=schema)


def _empty_table(schema):
    return pa.Table.from_arrays([pa.array([], type=f.type) for f in schema],
                                schema=schema)


def _zero_column_table(nrows):
    """A table with zero columns but ``nrows`` logical rows.

    Arrow's ``concat_tables`` drops the row count of zero-column tables, so
    ``select([])`` is used here as the only reliable way to keep it.
    """
    if nrows <= 0:
        return pa.Table.from_arrays([], schema=pa.schema([]))
    return pa.table({'__rows__': pa.nulls(nrows)}).select([])


def _schema_from_frame(frame):
    """Derive an Arrow schema from a frame.

    Arrow types come from pyarrow's conversion; nullability follows odo's
    datashape conventions (via ``discover``) so that ``discover(dataset)``
    matches ``discover(frame)``.
    """
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    if len(frame.columns) == 0:
        return pa.schema([])
    table = pa.Table.from_pandas(frame, preserve_index=False)
    measures = _measures_from_frame(frame)
    fields = []
    for field in table.schema:
        measure = measures.get(field.name)
        nullable = (isinstance(measure, Option)
                    if measure is not None
                    else _series_is_nullable(frame[field.name]))
        fields.append(pa.field(field.name, field.type, nullable))
    return pa.schema(fields)


def _measures_from_frame(frame):
    """Map column name -> datashape measure using ``discover``."""
    try:
        measure = discover(frame).measure
    except Exception:
        return {}
    if not isrecord(measure):
        return {}
    return dict(zip(measure.names, measure.types))


def _series_is_nullable(series):
    try:
        return bool(series.isnull().any())
    except Exception:
        return True


def _is_record_dshape(dshape):
    """Whether ``dshape`` names columns (as opposed to bare tuples)."""
    try:
        measure = datashape.dshape(dshape).measure
    except Exception:
        return False
    return isrecord(measure) and len(measure) > 0


def _column_names(data):
    """The column names present in ``data`` before any coercion."""
    if isinstance(data, (pd.DataFrame, pa.Table)):
        return list(data.columns if isinstance(data, pd.DataFrame)
                    else data.column_names)
    if isinstance(data, pa.RecordBatch):
        return list(data.schema.names)
    if isinstance(data, dict):
        return list(data.keys())
    try:
        frame = pd.DataFrame(data)
    except Exception:
        return []
    return list(frame.columns)


def _provided_columns(data, schema):
    """Columns present in ``data``, accounting for positional data.

    A list of tuples (``[('west', 1), ('east', 2)]``) has integer labels; if
    it has exactly as many columns as the schema, the labels are assumed to be
    positional and the schema's names are used instead.
    """
    names = _column_names(data)
    schema_names = [f.name for f in schema]
    frame_columns = _column_names(data)
    if (frame_columns and not any(c in schema_names for c in frame_columns)
            and len(frame_columns) == len(schema_names)):
        return list(schema_names)
    return names


# ---------------------------------------------------------------------------
# Partition value encoding
# ---------------------------------------------------------------------------

def _encode_partition_value(value):
    """Encode a partition value as JSON-friendly text.

    ``None`` becomes a sentinel so it never collides with ``''``.  A literal
    string that happens to equal the sentinel is escaped so it also stays
    distinct from ``NULL``.  The value is decoded using the Arrow type recorded
    in the manifest.
    """
    if _is_null(value):
        return NULL_SENTINEL
    if isinstance(value, (bool, np.bool_)):
        return 'true' if value else 'false'
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    text = str(value)
    if text.startswith(ESCAPE_PREFIX) or text == NULL_SENTINEL:
        return ESCAPE_PREFIX + text
    return text


def _decode_partition_value(text, typ):
    """Decode a manifest value with the Arrow ``typ`` recorded alongside it."""
    if text is None:
        return None
    if isinstance(text, str) and text.startswith(ESCAPE_PREFIX):
        # A literal value that was escaped on the way in.
        return text[len(ESCAPE_PREFIX):]
    if text == NULL_SENTINEL:
        return None
    try:
        if pa.types.is_boolean(typ):
            return text.lower() == 'true'
        if pa.types.is_integer(typ):
            return int(text)
        if pa.types.is_floating(typ):
            return float(text)
        if pa.types.is_timestamp(typ):
            return pd.Timestamp(text)
        if pa.types.is_date(typ):
            return pd.Timestamp(text).date()
        if pa.types.is_time(typ):
            return pd.Timestamp('1970-01-01T' + text).time()
        if pa.types.is_duration(typ):
            return pd.Timedelta(text)
        if pa.types.is_decimal(typ):
            return Decimal(text)
    except (ValueError, TypeError):
        pass
    return text


def _filter_matches(value, wanted):
    """Whether a decoded partition value satisfies a filter specification."""
    if wanted is None:
        return value is None
    if isinstance(wanted, (list, tuple, set, frozenset)):
        return any(_filter_matches(value, w) for w in wanted)
    if value is None:
        return False
    try:
        return value == wanted
    except Exception:
        return str(value) == str(wanted)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _schema_to_text(schema):
    return base64.b64encode(schema.serialize().to_pybytes()).decode('ascii')


def _schema_from_text(text):
    return pa.ipc.read_schema(pa.BufferReader(base64.b64decode(text)))


class Manifest(object):
    """One complete, immutable version of a dataset.

    Attributes
    ----------
    schema : pyarrow.Schema
        Logical schema of the dataset, including partition columns.
    partition_on : tuple of str
        Partition key column names.
    files : list of dict
        ``{'path': relative path, 'partitions': {key: encoded},
        'rows': int}`` for every Parquet file in this version.
    partitions : list of dict
        ``{'values': {key: encoded}}`` for every partition directory in this
        version, including partitions that currently hold zero rows.
    """

    def __init__(self, schema, partition_on, files, partitions):
        self.schema = schema
        self.partition_on = tuple(partition_on)
        self.files = list(files)
        self.partitions = list(partitions)

    # -- serialisation -----------------------------------------------------

    def to_dict(self):
        return {
            'format': MANIFEST_FORMAT,
            'schema': _schema_to_text(self.schema),
            'partition_on': list(self.partition_on),
            'files': self.files,
            'partitions': self.partitions,
        }

    @classmethod
    def from_dict(cls, data):
        if data.get('format') != MANIFEST_FORMAT:
            raise ValueError('unsupported dataset manifest format %r' %
                             (data.get('format'),))
        return cls(
            schema=_schema_from_text(data['schema']),
            partition_on=data.get('partition_on', ()),
            files=data.get('files', []),
            partitions=data.get('partitions', []),
        )

    # -- helpers -----------------------------------------------------------

    def with_schema(self, schema):
        return Manifest(schema, self.partition_on, self.files, self.partitions)

    def public_schema(self):
        """The schema as observed by readers (nullability preserved)."""
        return self.schema


# ---------------------------------------------------------------------------
# Advisory writer lock
# ---------------------------------------------------------------------------

class _FileLock(object):
    """A small cross-process lock built on ``O_CREAT|O_EXCL``."""

    def __init__(self, path, timeout=120.0, poll=0.05):
        self.path = path
        self.timeout = timeout
        self.poll = poll
        self._fd = None

    def __enter__(self):
        deadline = time.time() + self.timeout
        while True:
            try:
                self._fd = os.open(self.path, os.O_CREAT | os.O_EXCL |
                                   os.O_WRONLY, 0o644)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
                if self._stale():
                    try:
                        os.remove(self.path)
                    except OSError:
                        pass
                    continue
                if time.time() > deadline:
                    raise IOError('timed out waiting for dataset lock %s' %
                                  self.path)
                time.sleep(self.poll)
            else:
                os.write(self._fd, str(os.getpid()).encode('ascii'))
                os.fsync(self._fd)
                return self

    def _stale(self):
        try:
            with open(self.path, 'r') as f:
                pid = int(f.read().strip() or '0')
        except (OSError, ValueError):
            return False
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except OSError as e:
            return e.errno == errno.ESRCH
        except Exception:
            return False
        return False

    def __exit__(self, *exc_info):
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None
        try:
            os.remove(self.path)
        except OSError:
            pass
        return False


# ---------------------------------------------------------------------------
# Snapshots (pinned reads)
# ---------------------------------------------------------------------------

class _Snapshot(object):
    """A pinned, complete view of one published manifest.

    Every read goes through a snapshot.  The manifest is captured once, so a
    concurrent publish can never change the set of files this reader sees.
    ``ParquetFile`` objects are opened lazily but pinned: once opened they
    keep working even if the underlying path is later removed.
    """

    def __init__(self, dataset, manifest):
        self.dataset = dataset
        self.manifest = manifest
        self._open_files = {}

    # -- context management -----------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False

    def close(self):
        self._open_files.clear()

    # -- selection ---------------------------------------------------------

    def _selected_files(self, partitions, columns):
        """Files (relative paths) matching the requested partition filters."""
        return [record for record in self.manifest.files
                if self._partition_matches(record, partitions)]

    def _partition_matches(self, record, partitions):
        if not partitions:
            return True
        if not self.manifest.partition_on:
            raise ValueError('dataset has no partition columns to filter on')
        for key, wanted in _normalise_partitions(partitions).items():
            if key not in self.manifest.partition_on:
                raise ValueError('%r is not a partition column' % (key,))
            typ = self.manifest.schema.field(key).type
            value = _decode_partition_value(record['partitions'].get(key), typ)
            if not _filter_matches(value, wanted):
                return False
        return True

    def _output_columns(self, columns):
        if columns is None:
            return [f.name for f in self.manifest.schema]
        if isinstance(columns, str):
            columns = [columns]
        present = set(self.manifest.schema.names)
        missing = [c for c in columns if c not in present]
        if missing:
            raise KeyError('columns %r are not part of the dataset' %
                           (missing,))
        return list(columns)

    def _physical_columns(self, columns):
        """Only the stored columns need to come out of the Parquet files."""
        stored = set(self.manifest.partition_on)
        if columns is None:
            return None
        return [c for c in columns if c not in stored]

    def output_schema(self, columns=None):
        columns = self._output_columns(columns)
        by_name = dict((f.name, f) for f in self.manifest.public_schema())
        return pa.schema([by_name[c] for c in columns])

    # -- iteration ---------------------------------------------------------

    def iter_batches(self, columns=None, partitions=None,
                     batch_size=DEFAULT_CHUNKSIZE, **kwargs):
        """Yield Arrow tables, one Parquet batch at a time."""
        out_columns = self._output_columns(columns)
        physical = self._physical_columns(out_columns)
        records = self._selected_files(partitions, out_columns)
        partition_fields = [
            f for f in self.manifest.schema if f.name in
            set(self.manifest.partition_on) and f.name in set(out_columns)
        ]
        if not records:
            return
        for record in records:
            if physical == []:
                # No stored columns were requested (for example the dataset
                # consists only of partition keys).  Parquet cannot represent
                # a zero-column file with rows, so rebuild the batch from the
                # row count recorded in the manifest.
                yield self._partition_only_table(record, partition_fields,
                                                 out_columns)
                continue
            pf = self._open(record['path'])
            for batch in pf.iter_batches(batch_size=batch_size,
                                         columns=physical):
                base = pa.Table.from_batches([batch])
                yield self._attach_partitions(base, record, partition_fields,
                                              out_columns)

    def _partition_only_table(self, record, partition_fields, out_columns):
        nrows = record.get('rows', 0)
        schema = self.output_schema(out_columns)
        if not schema.names:
            return _zero_column_table(nrows)
        arrays = {}
        for field in partition_fields:
            value = _decode_partition_value(record['partitions'].get(field.name),
                                            field.type)
            arrays[field.name] = pa.repeat(pa.scalar(value, type=field.type),
                                           nrows)
        return pa.Table.from_arrays([arrays[name] for name in schema.names],
                                    schema=schema)

    def _open(self, relpath):
        pf = self._open_files.get(relpath)
        if pf is None:
            pf = pq.ParquetFile(self.dataset._abs(relpath))
            self._open_files[relpath] = pf
        return pf

    def _attach_partitions(self, table, record, partition_fields, out_columns):
        """Add typed partition columns and order columns as requested."""
        arrays = dict(zip(table.column_names, table.columns))
        for field in partition_fields:
            typ = field.type
            value = _decode_partition_value(
                record['partitions'].get(field.name), typ,
            )
            arrays[field.name] = pa.repeat(pa.scalar(value, type=typ),
                                           table.num_rows)
        schema = self.output_schema(out_columns)
        ordered = [arrays[name] for name in schema.names]
        return pa.Table.from_arrays(ordered, schema=schema)

    def iter_tables(self, columns=None, partitions=None, **kwargs):
        for table in self.iter_batches(columns=columns, partitions=partitions,
                                       **kwargs):
            yield table

    def iter_dataframes(self, columns=None, partitions=None, **kwargs):
        for table in self.iter_batches(columns=columns, partitions=partitions,
                                       **kwargs):
            yield _table_to_dataframe(table)

    def read_table(self, columns=None, partitions=None, **kwargs):
        """Read the selected rows as one Arrow table."""
        schema = self.output_schema(columns)
        tables = list(self.iter_batches(columns=columns, partitions=partitions,
                                        **kwargs))
        if not tables:
            return _empty_table(schema)
        if not schema.names:
            # No columns were requested, but the row count must survive.
            return _zero_column_table(sum(t.num_rows for t in tables))
        try:
            return pa.concat_tables(tables)
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            return pa.concat_tables([t.cast(schema) for t in tables])

    def read_dataframe(self, columns=None, partitions=None, **kwargs):
        return _table_to_dataframe(
            self.read_table(columns=columns, partitions=partitions, **kwargs)
        )


def _parquet_row_count(path):
    try:
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return 0


def _normalise_partitions(partitions):
    """Normalise a partition filter into ``{key: value or [values]}``.

    Accepts a mapping (``{'region': 'west'}``), a sequence of mappings (all
    must match), or keyword-style pairs.  Values may be a scalar or a
    collection of candidate values.
    """
    if partitions is None:
        return {}
    if isinstance(partitions, dict):
        return dict(partitions)
    if isinstance(partitions, (list, tuple)):
        result = {}
        for item in partitions:
            if not isinstance(item, dict):
                raise TypeError('partition filters must be mappings')
            for key, value in item.items():
                if key in result:
                    previous = result[key]
                    if not isinstance(previous, (list, tuple, set, frozenset)):
                        previous = [previous]
                    if isinstance(value, (list, tuple, set, frozenset)):
                        previous = list(previous) + list(value)
                    else:
                        previous = list(previous) + [value]
                    result[key] = previous
                else:
                    result[key] = value
        return result
    raise TypeError('partition filters must be a mapping or list of mappings')


# ---------------------------------------------------------------------------
# ParquetDataset
# ---------------------------------------------------------------------------

class ParquetDataset(object):
    """A local, partitioned, Parquet-backed dataset.

    Parameters
    ----------
    path : str
        Directory that holds the dataset.  Created on first write.
    partition_on : str or sequence of str, optional
        Column names to partition by.  Each distinct combination of values is
        stored in its own group of files.  ``None`` and ``''`` are distinct.
    schema : pyarrow.Schema, optional
        The logical schema.  Required to create an empty dataset and otherwise
        inferred from the first write or from an existing directory.
    dshape : DataShape or str, optional
        Alternative to ``schema``.  Partition keys missing from the datashape
        are added as nullable strings.
    compression : str, default ``'snappy'``
        Parquet compression codec.

    Notes
    -----
    A ``ParquetDataset`` object stores only configuration.  It loads the
    manifest on demand, so it always sees the latest published version when a
    read starts.  Each read pins its own snapshot, so concurrent publishes
    never mix files into one result.
    """

    def __init__(self, path, partition_on=(), schema=None, dshape=None,
                 compression=DEFAULT_COMPRESSION, **kwargs):
        if isinstance(partition_on, str):
            partition_on = [partition_on]
        self.path = os.path.abspath(path)
        self.partition_on = tuple(partition_on)
        self.compression = compression
        self.kwargs = kwargs

        if schema is None and dshape is not None:
            schema = arrow_schema_from_dshape(dshape, self.partition_on)
        elif schema is not None and not isinstance(schema, pa.Schema):
            if isinstance(schema, (str, DataShape)):
                schema = arrow_schema_from_dshape(schema, self.partition_on)
            else:
                raise TypeError('schema must be a pyarrow.Schema or a '
                                'datashape')
        self._initial_schema = schema
        self._thread_lock = threading.RLock()

    # -- paths -------------------------------------------------------------

    @property
    def metadata_path(self):
        return os.path.join(self.path, METADATA_DIRNAME)

    @property
    def data_path(self):
        return os.path.join(self.path, DATA_DIRNAME)

    @property
    def manifest_dir(self):
        return os.path.join(self.metadata_path, MANIFEST_DIRNAME)

    @property
    def current_path(self):
        return os.path.join(self.metadata_path, CURRENT_FILENAME)

    @property
    def lock_path(self):
        return os.path.join(self.metadata_path, LOCK_FILENAME)

    @property
    def staging_path(self):
        return os.path.join(self.metadata_path, STAGING_DIRNAME)

    def _abs(self, relpath):
        if os.path.isfile(self.path) and relpath == os.path.basename(self.path):
            # The dataset is a single Parquet file rather than a directory.
            return self.path
        return os.path.join(self.path, relpath)

    def _ensure_layout(self):
        _makedirs(self.manifest_dir)
        _makedirs(self.staging_path)
        _makedirs(self.data_path)

    def __repr__(self):
        return 'ParquetDataset(%r, partition_on=%r)' % (
            self.path, list(self.partition_on),
        )

    # -- manifest access ---------------------------------------------------

    def _current_name(self):
        try:
            with open(self.current_path, 'r') as f:
                name = f.read().strip()
        except (IOError, OSError):
            return None
        return name or None

    def _load(self, name):
        with open(os.path.join(self.manifest_dir, name), 'r') as f:
            return Manifest.from_dict(json.load(f))

    def _empty_manifest(self):
        """A manifest for a dataset that has no published version yet.

        If the directory already holds Parquet files (for example a dataset
        written by another tool), or ``path`` is itself a Parquet file, those
        files are adopted into an in-memory manifest.  Nothing is written to
        disk until the first write.
        """
        if self._initial_schema is None or not self._initial_schema.names:
            adopted = self._adopt_manifest()
            if adopted is not None:
                # Remember the partition keys we discovered so later writes
                # and reads use the same layout.
                if not self.partition_on:
                    self.partition_on = adopted.partition_on
                return adopted
        schema = self._initial_schema
        if schema is None:
            schema = pa.schema([])
        return Manifest(schema, self.partition_on, [], [])

    def manifest(self):
        """The currently published manifest (an empty one if none exists)."""
        name = self._current_name()
        if name is None:
            return self._empty_manifest()
        return self._load(name)

    def snapshot(self):
        """Pin the current version for a consistent multi-chunk read."""
        return _Snapshot(self, self.manifest())

    def _adopt_manifest(self):
        """Build a manifest from Parquet files already on disk.

        Hive-style partition directories (``key=value``) are understood; other
        files are ignored.  Values from directory names are strings, except
        that Hive's ``__HIVE_DEFAULT_PARTITION__`` marker becomes ``None``.
        """
        files = self._disk_parquet_files()
        if not files:
            return None
        schema = None
        records = []
        keys = []
        for relpath in files:
            try:
                file_schema = pq.read_schema(self._abs(relpath))
            except Exception:
                # Not a readable Parquet file: ignore it rather than failing
                # an unrelated read.
                continue
            schema = file_schema if schema is None else schema
            partition_values = {}
            for part in os.path.dirname(relpath).split(os.sep):
                if not part or '=' not in part:
                    continue
                name, _, value = part.partition('=')
                partition_values[name] = (
                    NULL_SENTINEL if value == '__HIVE_DEFAULT_PARTITION__'
                    else value
                )
            records.append({
                'path': relpath,
                'partitions': partition_values,
                'rows': _parquet_row_count(self._abs(relpath)),
            })
            if partition_values:
                keys.append(tuple(partition_values))
        if schema is None:
            return None
        if keys:
            all_keys = []
            for key in keys:
                for name in key:
                    if name not in all_keys:
                        all_keys.append(name)
            partition_on = tuple(all_keys)
        else:
            partition_on = tuple()
        if self.partition_on and tuple(partition_on) != self.partition_on:
            raise ValueError(
                'directory is partitioned by %r but partition_on=%r was '
                'requested' % (partition_on, self.partition_on),
            )
        fields = [pa.field(f.name, f.type, True) for f in schema]
        for name in partition_on:
            if name not in schema.names:
                fields.append(pa.field(name, pa.string(), True))
        manifest_schema = pa.schema(fields)
        for record in records:
            for name in partition_on:
                record['partitions'].setdefault(name, NULL_SENTINEL)
        partitions = []
        seen = set()
        for record in records:
            key = tuple(record['partitions'].get(name) for name in partition_on)
            if key in seen:
                continue
            seen.add(key)
            partitions.append({'values': dict(zip(partition_on, key))})
        return Manifest(manifest_schema, partition_on, records, partitions)

    def _infer_schema(self):
        """Adopt the schema of a pre-existing directory of Parquet files."""
        schema = None
        for relpath in self._disk_parquet_files():
            try:
                candidate = pq.read_schema(self._abs(relpath))
            except Exception:
                continue
            if schema is None:
                schema = candidate
            elif schema != candidate:
                raise ValueError('cannot adopt a Parquet directory with '
                                 'inconsistent schemas: %r differs' % relpath)
        if schema is None:
            return None
        return pa.schema([pa.field(f.name, f.type, True) for f in schema])

    def _disk_parquet_files(self):
        """Parquet files under the dataset root, ignoring our own metadata.

        If ``path`` itself is a Parquet file it is returned (as a bare
        filename) so that a single file can be read as a one-file dataset.
        """
        found = []
        if os.path.isfile(self.path):
            return [os.path.basename(self.path)]
        if not os.path.isdir(self.path):
            return found
        for dirpath, dirnames, filenames in os.walk(self.path):
            rel_dir = os.path.relpath(dirpath, self.path)
            parts = [] if rel_dir == os.curdir else rel_dir.split(os.sep)
            if parts and parts[0] == METADATA_DIRNAME:
                dirnames[:] = []
                continue
            dirnames[:] = sorted(d for d in dirnames if not d.startswith('.'))
            for filename in sorted(filenames):
                if filename.startswith('.') or not filename.endswith('.parquet'):
                    continue
                found.append(os.path.relpath(
                    os.path.join(dirpath, filename), self.path,
                ))
        return found

    def _persist(self, manifest):
        """Write the manifest and atomically mark it current."""
        self._ensure_layout()
        name = 'manifest-%s.json' % uuid.uuid4().hex
        payload = json.dumps(manifest.to_dict(), sort_keys=True).encode('utf-8')
        _atomic_write(os.path.join(self.manifest_dir, name), payload)
        _atomic_write(self.current_path, name.encode('ascii'))

    def _new_relpath(self):
        return os.path.join(DATA_DIRNAME, '%s.parquet' % uuid.uuid4().hex)

    # -- writing -----------------------------------------------------------

    @contextlib.contextmanager
    def _writer(self):
        """Serialise writers within this process and across processes."""
        with self._thread_lock:
            _makedirs(self.metadata_path)
            with _FileLock(self.lock_path):
                yield

    def _write_group(self, table):
        """Write a table to a brand-new Parquet file; return its relpath."""
        relpath = self._new_relpath()
        abspath = self._abs(relpath)
        _makedirs(os.path.dirname(abspath))
        _makedirs(self.staging_path)
        tmp = os.path.join(self.staging_path, '%s.parquet.tmp' %
                           uuid.uuid4().hex)
        try:
            pq.write_table(table, tmp, compression=self.compression)
            with open(tmp, 'rb') as f:
                os.fsync(f.fileno())
            _replace_file(tmp, abspath)
            _fsync_dir(os.path.dirname(abspath))
        except BaseException:
            _remove_tree(tmp)
            raise
        return relpath

    def append(self, data, dshape=None, replace_partition=None, **kwargs):
        """Append ``data`` and publish a new version.

        Parameters
        ----------
        data : DataFrame, Table or iterable
            New rows.  Columns not present in the dataset are rejected.
        dshape : DataShape or str, optional
            Schema to use for the write.  Otherwise inferred from ``data`` or
            taken from the existing dataset.
        replace_partition : mapping or sequence of mappings, optional
            Partitions whose existing rows should be discarded before these
            rows are added.  Values keep their type; ``None`` selects the
            ``NULL`` partition and ``''`` selects the empty-string one.

        Returns
        -------
        self
        """
        if replace_partition is not None:
            return self._replace(data, replace_partition, dshape=dshape,
                                 **kwargs)
        data = _materialise(data)
        with self._writer():
            manifest = self._manifest_for_write()
            if not self.partition_on and manifest.partition_on:
                self.partition_on = manifest.partition_on
            schema = self._write_schema(manifest, dshape, data)
            table, schema = self._prepare(data, schema)
            manifest = manifest.with_schema(schema)
            self._publish(manifest, table, drop_keys=set())
        return self

    def replace_partition(self, data, partition=None, dshape=None, **kwargs):
        """Replace one or more partitions with ``data``.

        ``partition`` is a mapping (``{'region': 'west'}``) or a sequence of
        mappings.  Passing ``None`` replaces the single all-``NULL`` partition
        of an unpartitioned dataset.
        """
        return self._replace(data, partition, dshape=dshape, **kwargs)

    def _replace(self, data, partition, dshape=None, **kwargs):
        data = _materialise(data)
        with self._writer():
            manifest = self._manifest_for_write()
            if not self.partition_on and manifest.partition_on:
                # An adopted directory defines the partition keys.
                self.partition_on = manifest.partition_on
            targets = self._normalise_targets(partition)
            schema = self._write_schema(manifest, dshape, data)
            table, schema = self._prepare(data, schema, targets=targets)
            manifest = manifest.with_schema(schema)
            self._publish(manifest, table, drop_keys=set(targets),
                          keep_keys=set(targets))
        return self

    def _normalise_targets(self, partition):
        if partition is None:
            return [()]
        if isinstance(partition, dict):
            partition = [partition]
        if not isinstance(partition, (list, tuple)):
            raise TypeError('partition must be a mapping or a list of mappings')
        targets = []
        for item in partition:
            if not isinstance(item, dict):
                raise TypeError('each partition must be a mapping')
            unknown = set(item) - set(self.partition_on)
            if unknown:
                raise ValueError('%r are not partition columns' %
                                 (sorted(unknown),))
            missing = [k for k in self.partition_on if k not in item]
            if missing:
                raise ValueError('partition specification is missing keys %r' %
                                 (missing,))
            targets.append(tuple(_encode_partition_value(item[k])
                                 for k in self.partition_on))
        return targets

    def _manifest_for_write(self):
        name = self._current_name()
        if name is None:
            return self._empty_manifest()
        manifest = self._load(name)
        if manifest.partition_on != self.partition_on:
            raise ValueError('dataset was created with partition_on=%r, not '
                             '%r' % (manifest.partition_on, self.partition_on))
        return manifest

    def _write_schema(self, manifest, dshape, data):
        """Decide the schema for this write.

        * The first write defines the schema (from ``dshape`` if given, else
          from the data).
        * Later writes use the dataset's schema as authoritative: incoming
          data is cast to it, so an ``int32`` SQL column can land in an
          ``int64`` column.  Unknown columns are rejected by ``_prepare``.
        """
        if manifest.schema.names:
            schema = manifest.schema
        elif dshape is not None and _is_record_dshape(dshape):
            schema = arrow_schema_from_dshape(dshape, self.partition_on)
        elif dshape is not None:
            # A non-record datashape cannot name columns; derive names from
            # the data instead.
            schema = _schema_from_frame(pd.DataFrame(data))
        elif isinstance(data, pa.Table):
            schema = pa.schema([pa.field(f.name, f.type, f.nullable)
                                for f in data.schema])
        else:
            schema = _schema_from_frame(data)
        schema = self._add_missing_partition_fields(schema)
        # Partition keys may legitimately hold NULL; every other field keeps
        # its inferred nullability.
        partition_on = set(self.partition_on)
        return pa.schema([
            pa.field(f.name, f.type,
                     True if f.name in partition_on else f.nullable)
            for f in schema
        ])

    def _add_missing_partition_fields(self, schema):
        present = set(schema.names)
        fields = list(schema)
        for key in self.partition_on:
            if key not in present:
                fields.append(pa.field(key, pa.string(), True))
        return pa.schema(fields)

    def _prepare(self, data, schema, targets=None):
        """Turn ``data`` into a typed table with all partition columns.

        The returned table keeps the partition columns: they are needed to
        group rows, and they are dropped from each stored file in
        :meth:`_publish` (their values live in the manifest).
        """
        provided = _provided_columns(data, schema)
        known = set(f.name for f in schema)
        extra = [c for c in provided if c not in known]
        if extra:
            raise ValueError('data contains columns %r that are not part of '
                             'the dataset' % (extra,))
        missing = [k for k in self.partition_on if k not in provided]
        table = _as_table(data, schema)
        if missing:
            if not targets:
                raise ValueError('data is missing partition columns %r; use '
                                 'replace_partition to supply them' % missing)
            if len(targets) != 1:
                raise ValueError('data is missing partition columns %r but '
                                 'multiple partitions were requested' %
                                 missing)
            table = self._fill_partition_columns(table, schema, targets[0])
        elif targets:
            self._check_partition_values(table, targets)
        names = [f.name for f in schema]
        table = _cast_table(table.select(names), schema)
        # Parquet silently turns a NULL in a non-nullable column into a
        # zero/empty value, so every column that actually contains a null in
        # this write must be written as nullable.  Widen the manifest schema
        # to match, which also keeps ``discover`` truthful.
        widened = self._widen_nullability(schema, table)
        if widened != schema:
            table = _cast_table(table, widened)
        return table, widened

    def _widen_nullability(self, schema, table):
        fields = []
        changed = False
        for field in schema:
            nullable = field.nullable
            if not nullable and field.name in table.column_names:
                column = table[field.name]
                if column.null_count:
                    nullable = True
                    changed = True
            fields.append(pa.field(field.name, field.type, nullable))
        return pa.schema(fields) if changed else schema

    def _fill_partition_columns(self, table, schema, target):
        arrays = dict(zip(table.column_names, table.columns))
        for key, encoded in zip(self.partition_on, target):
            typ = schema.field(key).type
            value = _decode_partition_value(encoded, typ)
            arrays[key] = pa.repeat(pa.scalar(value, type=typ), table.num_rows)
        names = [f.name for f in schema]
        return pa.Table.from_arrays([arrays[n] for n in names], names=names)

    def _check_partition_values(self, table, targets):
        """Ensure a replacement frame only contains the targeted partitions."""
        if table.num_rows == 0:
            return
        encoded = set()
        for i in range(table.num_rows):
            key = []
            for name in self.partition_on:
                value = table[name][i].as_py()
                key.append(_encode_partition_value(value))
            encoded.add(tuple(key))
        unexpected = encoded - set(targets)
        if unexpected:
            raise ValueError(
                'data contains partition values %r that are not among the '
                'target partitions %r' % (sorted(unexpected), sorted(targets)),
            )

    def _publish(self, manifest, table, drop_keys, keep_keys=None):
        """Write the new data files, then atomically publish a new manifest.

        Existing files are never touched.  Only after all new files are
        written and fsynced is ``CURRENT`` swapped, so a failure leaves the
        previous version fully intact.
        """
        groups = self._group_by_partition(table)
        new_files = []
        written_paths = []
        stored_names = [name for name in table.column_names
                        if name not in set(self.partition_on)]
        try:
            for key in sorted(groups, key=_sort_key):
                group = groups[key]
                if stored_names:
                    stored = group.select(stored_names)
                else:
                    stored = group.select([])
                relpath = self._write_group(stored)
                written_paths.append(self._abs(relpath))
                new_files.append({
                    'path': relpath,
                    'partitions': dict(zip(self.partition_on, key)),
                    'rows': int(group.num_rows),
                })
        except BaseException:
            # Best-effort cleanup so a failed publish does not accumulate
            # unreferenced files.  (Readers never see them regardless: they
            # only read files named by a manifest.)
            for path in written_paths:
                try:
                    os.remove(path)
                except OSError:
                    pass
            raise

        drop_keys = set(drop_keys)
        files = [f for f in manifest.files
                 if tuple(f['partitions'].get(k) for k in self.partition_on)
                 not in drop_keys]
        files.extend(new_files)

        if not self.partition_on:
            # There is exactly one (implicit) partition; listing it would be
            # noise, so the partition list stays empty for unpartitioned data.
            updated = Manifest(manifest.schema, self.partition_on, files, [])
            self._persist(updated)
            return updated

        existing_partitions = [
            tuple(p['values'].get(k) for k in self.partition_on)
            for p in manifest.partitions
        ]
        partitions = [{'values': p['values']} for p, key in
                      zip(manifest.partitions, existing_partitions)
                      if key not in drop_keys]
        seen = set(existing_partitions) - drop_keys
        if keep_keys:
            keep = set(keep_keys)
        else:
            keep = set()
        for key in sorted(groups, key=_sort_key):
            if key in seen:
                continue
            partitions.append({'values': dict(zip(self.partition_on, key))})
            seen.add(key)
        for key in sorted(keep - seen, key=_sort_key):
            # A replacement with zero rows still keeps the partition (and its
            # schema) visible.
            partitions.append({'values': dict(zip(self.partition_on, key))})
            seen.add(key)

        updated = Manifest(manifest.schema, self.partition_on, files,
                           partitions)
        self._persist(updated)
        return updated

    def _group_by_partition(self, table):
        if not self.partition_on:
            return {(): table}
        if table.num_rows == 0:
            return {}
        columns = dict(zip(table.column_names, table.columns))
        keys = []
        for name in self.partition_on:
            keys.append([_encode_partition_value(v)
                         for v in columns[name].to_pylist()])
        groups = {}
        for i in range(table.num_rows):
            key = tuple(column[i] for column in keys)
            groups.setdefault(key, []).append(i)
        return dict(
            (key, table.take(pa.array(indices, type=pa.int64())))
            for key, indices in groups.items()
        )

    # -- reading -----------------------------------------------------------

    def iter_chunks(self, columns=None, partitions=None, chunksize=None,
                    **kwargs):
        """Iterate over Arrow tables (one per Parquet batch)."""
        snapshot = self.snapshot()
        batch_size = chunksize or DEFAULT_CHUNKSIZE
        return snapshot.iter_batches(columns=columns, partitions=partitions,
                                     batch_size=batch_size, **kwargs)

    def read_table(self, columns=None, partitions=None, **kwargs):
        with self.snapshot() as snapshot:
            return snapshot.read_table(columns=columns, partitions=partitions,
                                       **kwargs)

    def read_dataframe(self, columns=None, partitions=None, **kwargs):
        return _table_to_dataframe(
            self.read_table(columns=columns, partitions=partitions, **kwargs)
        )

    def partitions_list(self):
        """The partition values currently present, including empty ones."""
        manifest = self.manifest()
        typ = dict((f.name, f.type) for f in manifest.schema)
        result = []
        for entry in manifest.partitions:
            values = {}
            for key in manifest.partition_on:
                values[key] = _decode_partition_value(
                    entry['values'].get(key), typ[key],
                )
            result.append(values)
        return result

    def partition_data(self, partition):
        """Read one partition (a mapping of partition values)."""
        return self.read_dataframe(partitions=partition)


def _sort_key(key):
    """Order partition tuples deterministically, with NULL first."""
    return tuple((0, '') if k == NULL_SENTINEL else (1, k) for k in key)


# ---------------------------------------------------------------------------
# odo integration: discovery, conversion, append, resource, drop
# ---------------------------------------------------------------------------

@discover.register(ParquetDataset)
def discover_parquet_dataset(dataset, **kwargs):
    """Discover a dataset as ``var * {column: type, ...}``."""
    manifest = dataset.manifest()
    if manifest.schema.names:
        return dshape_from_arrow_schema(manifest.public_schema())
    if dataset._initial_schema is not None:
        return dshape_from_arrow_schema(dataset._initial_schema)
    return datashape.var * Record([])


@convert.register(chunks(pd.DataFrame), ParquetDataset, cost=1.0)
def parquet_dataset_to_chunks_dataframe(dataset, chunksize=DEFAULT_CHUNKSIZE,
                                        columns=None, partitions=None,
                                        **kwargs):
    """Read a dataset as chunks of DataFrames from one pinned snapshot."""
    snapshot = dataset.snapshot()

    def load():
        for table in snapshot.iter_batches(columns=columns,
                                           partitions=partitions,
                                           batch_size=chunksize):
            yield _table_to_dataframe(table)

    return chunks(pd.DataFrame)(load)


@convert.register(pd.DataFrame, ParquetDataset, cost=1.0)
def parquet_dataset_to_dataframe(dataset, columns=None, partitions=None,
                                 **kwargs):
    with dataset.snapshot() as snapshot:
        return snapshot.read_dataframe(columns=columns,
                                       partitions=partitions, **kwargs)


@convert.register(chunks(pd.DataFrame), _Snapshot, cost=1.0)
def snapshot_to_chunks_dataframe(snapshot, chunksize=DEFAULT_CHUNKSIZE,
                                 columns=None, partitions=None, **kwargs):
    def load():
        for table in snapshot.iter_batches(columns=columns,
                                           partitions=partitions,
                                           batch_size=chunksize):
            yield _table_to_dataframe(table)

    return chunks(pd.DataFrame)(load)


@convert.register(pd.DataFrame, _Snapshot, cost=1.0)
def snapshot_to_dataframe(snapshot, columns=None, partitions=None, **kwargs):
    return snapshot.read_dataframe(columns=columns, partitions=partitions,
                                   **kwargs)


@convert.register(pa.Table, ParquetDataset, cost=1.0)
def parquet_dataset_to_table(dataset, columns=None, partitions=None, **kwargs):
    return dataset.read_table(columns=columns, partitions=partitions, **kwargs)


@append.register(ParquetDataset, (pd.DataFrame, pa.Table))
def append_frame_to_parquet_dataset(dataset, data, dshape=None, **kwargs):
    return dataset.append(data, dshape=dshape, **kwargs)


@append.register(ParquetDataset, (list, tuple))
def append_sequence_to_parquet_dataset(dataset, data, dshape=None, **kwargs):
    return dataset.append(pd.DataFrame(data), dshape=dshape, **kwargs)


@append.register(ParquetDataset, Chunks)
def append_chunks_to_parquet_dataset(dataset, data, dshape=None, **kwargs):
    for chunk in data:
        dataset.append(chunk, dshape=dshape, **kwargs)
    return dataset


@append.register(ParquetDataset, object)
def append_object_to_parquet_dataset(dataset, data, dshape=None, **kwargs):
    write_dshape = dshape
    try:
        frame = convert(pd.DataFrame, data, dshape=dshape, **kwargs)
    except (ValueError, KeyError):
        frame = _reconvert_csv_with_header(dataset, data, dshape, kwargs)
        if frame is None:
            raise
    frame = _repair_missing_header(dataset, frame)
    if write_dshape is None:
        write_dshape = _dataset_dshape(dataset)
    return dataset.append(frame, dshape=write_dshape, **kwargs)


def _dataset_dshape(dataset):
    schema = dataset._initial_schema
    if schema is None:
        schema = dataset.manifest().schema
    if not schema.names:
        return None
    return dshape_from_arrow_schema(schema)


def _reconvert_csv_with_header(dataset, data, dshape, kwargs):
    """Re-read a CSV-like source that odo's sniffer mistook for headerless."""
    if not _looks_like_csv(data):
        return None
    read_kwargs = dict(kwargs)
    read_kwargs['has_header'] = True
    target_dshape = dshape
    if target_dshape is None and dataset._initial_schema is not None:
        target_dshape = dshape_from_arrow_schema(dataset._initial_schema)
    if target_dshape is not None:
        read_kwargs['dshape'] = target_dshape
    try:
        return convert(pd.DataFrame, data, **read_kwargs)
    except Exception:
        return None


def _looks_like_csv(data):
    path = getattr(data, 'path', None)
    return isinstance(path, str) and path.lower().endswith(
        ('.csv', '.tsv', '.ssv', '.dat'),
    )


def _repair_missing_header(dataset, frame):
    """Fix a frame whose first row was read as data because the header was
    missed.  Only applies when the dataset already has a schema."""
    schema = dataset._initial_schema
    if schema is None:
        try:
            manifest = dataset.manifest()
        except Exception:
            manifest = None
        if manifest is not None and manifest.schema.names:
            schema = manifest.schema
    if schema is None or not isinstance(frame, pd.DataFrame):
        return frame
    names = [f.name for f in schema]
    if list(frame.columns) == names:
        return frame
    if len(frame.columns) == len(names) and len(frame) > 0:
        first_row = [str(v) for v in frame.iloc[0].tolist()]
        if first_row == [str(n) for n in names]:
            frame = frame.iloc[1:].copy()
            frame.columns = names
            return frame
    if (len(frame.columns) == len(names)
            and len(frame.columns) == len(names)
            and all(isinstance(c, int) or str(c).isdigit()
                    for c in frame.columns)):
        # Positional columns from a headerless read of data whose first row
        # really is data: just relabel.
        frame = frame.copy()
        frame.columns = names
    return frame


@append.register(ParquetDataset, ParquetDataset)
def append_dataset_to_parquet_dataset(dataset, other, **kwargs):
    for chunk in convert(chunks(pd.DataFrame), other, **kwargs):
        append(dataset, chunk, **kwargs)
    return dataset


@resource.register(r'.+\.parquet/*$', priority=14)
@resource.register(r'parquet://.+$', priority=14)
def resource_parquet_dataset(uri, dshape=None, partition_on=(), schema=None,
                             **kwargs):
    """Create or open a local Parquet dataset.

    Accepted forms::

        path/to/data.parquet
        path/to/data.parquet/
        parquet://path/to/dataset.parquet

    A bare ``.parquet`` file that exists and is *not* a dataset directory is
    still treated as a one-file dataset, so plain Parquet files can be read
    through the same resource.
    """
    path = uri
    if path.startswith('parquet://'):
        path = path[len('parquet://'):]
    path = path.rstrip(os.sep) or path
    return ParquetDataset(path, partition_on=partition_on, schema=schema,
                          dshape=dshape, **kwargs)


@drop.register(ParquetDataset)
def drop_parquet_dataset(dataset, **kwargs):
    _remove_tree(dataset.path)


#: Small helper multiline alias for callers that prefer ``Dataset``.
Dataset = ParquetDataset
LocalDataset = ParquetDataset

ooc_types.add(ParquetDataset)
