from __future__ import absolute_import, division, print_function

import os
import shutil
import uuid

import pandas as pd
import pytest

pa = pytest.importorskip('pyarrow')
pq = pytest.importorskip('pyarrow.parquet')

from odo import (
    ParquetDataset,
    append,
    convert,
    discover,
    drop,
    into,
    odo,
    resource,
)
from odo.backends.parquet import NULL_SENTINEL
from odo.chunks import chunks
from odo.utils import tmpfile


@pytest.fixture
def dataset():
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '.%s.parquet' % uuid.uuid4().hex,
    )
    try:
        yield path
    finally:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)


def make_frame():
    return pd.DataFrame({
        'region': ['west', 'east', 'west', None, ''],
        'amount': [1, 2, 3, 4, 5],
        'name': ['a', 'b', 'c', 'd', 'e'],
    })


def to_records(frame, sort='amount'):
    return sorted(frame.to_dict('records'), key=lambda r: r[sort])


# -- basics ---------------------------------------------------------------

def test_resource_and_roundtrip(dataset):
    ds = resource(dataset, dshape='var * {region: string, amount: int64}',
                  partition_on=['region'])
    assert isinstance(ds, ParquetDataset)
    odo([('west', 1), ('east', 2)], ds)
    result = odo(ds, pd.DataFrame)
    assert sorted(map(tuple, result[['region', 'amount']].values.tolist())) == [
        ('east', 2), ('west', 1),
    ]


def test_append_accumulates(dataset):
    ds = ParquetDataset(dataset, partition_on=['region'])
    ds.append(make_frame())
    ds.append(make_frame())
    assert len(ds.read_dataframe()) == 10


def test_discover(dataset):
    ds = ParquetDataset(dataset, partition_on=['region'])
    ds.append(make_frame())
    discovered = discover(ds)
    assert str(discovered.subshape[0].measure if hasattr(
        discovered, 'subshape') else discovered) == str(
            discover(ds.read_dataframe()).measure)
    assert 'region' in discovered.measure.names


def test_convert_to_chunks(dataset):
    ds = ParquetDataset(dataset, partition_on=['region'])
    ds.append(make_frame())
    result = convert(chunks(pd.DataFrame), ds, chunksize=2)
    assert isinstance(result, chunks(pd.DataFrame))
    assert len(list(result)) >= 1
    assert len(pd.concat(list(result))) == len(make_frame())


# -- partition values: type and NULL/'' distinction -----------------------

def test_null_and_empty_string_are_distinct(dataset):
    ds = ParquetDataset(dataset, partition_on=['region'])
    ds.append(make_frame())
    null_rows = ds.read_dataframe(partitions={'region': None})
    empty_rows = ds.read_dataframe(partitions={'region': ''})
    assert null_rows['amount'].tolist() == [4]
    assert empty_rows['amount'].tolist() == [5]


def test_partition_values_keep_type(dataset):
    ds = ParquetDataset(dataset, partition_on=['year'],
                        dshape='var * {year: int64, value: float64}')
    ds.append(pd.DataFrame({'year': [2020, 2020, 2021],
                            'value': [1.0, 2.0, 3.0]}))
    values = [p['year'] for p in ds.partitions_list()]
    assert 2020 in values and 2021 in values
    assert all(isinstance(v, int) or v is None for v in values)
    # An integer filter matches; a string filter does not.
    assert len(ds.read_dataframe(partitions={'year': 2020})) == 2
    assert len(ds.read_dataframe(partitions={'year': '2020'})) == 0


def test_null_partition_keeps_type_for_int_key(dataset):
    ds = ParquetDataset(dataset, partition_on=['year'],
                        dshape='var * {year: ?int64, value: int64}')
    ds.append(pd.DataFrame({'year': [1, None], 'value': [10, 20]}))
    assert ds.read_dataframe(partitions={'year': None})['value'].tolist() == [
        20,
    ]
    assert ds.read_dataframe(partitions={'year': 1})['value'].tolist() == [10]


# -- replace partition ----------------------------------------------------

def test_replace_single_partition(dataset):
    ds = ParquetDataset(dataset, partition_on=['region'])
    ds.append(make_frame())
    ds.replace_partition(pd.DataFrame({'amount': [100, 200], 'name': ['x',
                                                                    'y']}),
                         {'region': 'west'})
    records = to_records(ds.read_dataframe())
    assert [r['amount'] for r in records if r['region'] == 'west'] == [100, 200]
    # Untouched partitions are preserved.
    assert [r['amount'] for r in records if r['region'] == 'east'] == [2]


def test_replace_multiple_partitions(dataset):
    ds = ParquetDataset(dataset, partition_on=['region'])
    ds.append(make_frame())
    ds.replace_partition(
        pd.DataFrame({'region': ['west', 'east'],
                      'amount': [7, 8], 'name': ['w', 'e']}),
        [{'region': 'west'}, {'region': 'east'}],
    )
    records = ds.read_dataframe().to_dict('records')
    amounts = sorted(r['amount'] for r in records)
    assert 7 in amounts and 8 in amounts
    assert 1 not in amounts and 2 not in amounts


def test_replace_partition_rejects_foreign_values(dataset):
    ds = ParquetDataset(dataset, partition_on=['region'])
    ds.append(make_frame())
    with pytest.raises(ValueError):
        ds.replace_partition(
            pd.DataFrame({'region': ['west', 'north'], 'amount': [1, 2]}),
            {'region': 'west'},
        )


def test_replace_null_partition(dataset):
    ds = ParquetDataset(dataset, partition_on=['region'])
    ds.append(make_frame())
    ds.replace_partition(pd.DataFrame({'amount': [42], 'name': ['z']}),
                         {'region': None})
    assert ds.read_dataframe(partitions={'region': None})['amount'].tolist() == [
        42,
    ]
    # The empty-string partition is untouched.
    assert ds.read_dataframe(partitions={'region': ''})['amount'].tolist() == [5]


# -- empty partitions keep schema -----------------------------------------

def test_empty_replacement_keeps_schema(dataset):
    ds = ParquetDataset(dataset, partition_on=['region'])
    ds.append(make_frame())
    empty = pd.DataFrame({
        'amount': pd.Series([], dtype='int64'),
        'name': pd.Series([], dtype=object),
    })
    ds.replace_partition(empty, {'region': 'east'})
    result = ds.read_dataframe(partitions={'region': 'east'})
    assert result.empty
    assert set(result.columns) == {'region', 'amount', 'name'}
    assert 'east' in [p['region'] for p in ds.partitions_list()]
    # Discovered schema still includes all columns.
    assert set(discover(ds).measure.names) == {'region', 'amount', 'name'}


def test_create_empty_dataset_from_schema(dataset):
    ds = ParquetDataset(dataset,
                        dshape='var * {region: string, amount: int64}',
                        partition_on=['region'])
    assert discover(ds) is not None
    result = ds.read_dataframe()
    assert result.empty
    assert list(result.columns) == ['region', 'amount']


# -- atomicity ------------------------------------------------------------

def test_failed_write_exposes_nothing(dataset, monkeypatch):
    ds = ParquetDataset(dataset, partition_on=['region'])
    ds.append(pd.DataFrame({'region': ['a', 'b'], 'amount': [1, 2]}))
    before_files = list(ds.manifest().files)

    original = pq.write_table
    calls = {'n': 0}

    def flaky(table, path, *args, **kwargs):
        calls['n'] += 1
        if calls['n'] == 1:
            raise IOError('simulated disk failure')
        return original(table, path, *args, **kwargs)

    monkeypatch.setattr(pq, 'write_table', flaky)
    with pytest.raises(IOError):
        ds.append(pd.DataFrame({'region': ['c'], 'amount': [3]}))
    monkeypatch.setattr(pq, 'write_table', original)

    assert ds.manifest().files == before_files
    assert sorted(ds.read_dataframe()['amount'].tolist()) == [1, 2]


def test_failed_replacement_keeps_old_partition(dataset, monkeypatch):
    ds = ParquetDataset(dataset, partition_on=['region'])
    ds.append(pd.DataFrame({'region': ['a', 'a'], 'amount': [1, 2]}))

    original = pq.write_table

    def always_fail(*args, **kwargs):
        raise IOError('simulated disk failure')

    monkeypatch.setattr(pq, 'write_table', always_fail)
    with pytest.raises(IOError):
        ds.replace_partition(pd.DataFrame({'amount': [99]}), {'region': 'a'})
    monkeypatch.setattr(pq, 'write_table', original)

    assert sorted(ds.read_dataframe()['amount'].tolist()) == [1, 2]


# -- snapshot isolation ---------------------------------------------------

def test_snapshot_does_not_see_later_versions(dataset):
    ds = ParquetDataset(dataset, partition_on=['region'])
    ds.append(pd.DataFrame({'region': ['a', 'a', 'b'], 'amount': [1, 2, 3]}))
    snapshot = ds.snapshot()
    batches = snapshot.iter_batches()
    first = next(batches)
    # Publish a new version that replaces partition ``a`` *while* the reader
    # is iterating.
    ds.replace_partition(pd.DataFrame({'amount': [10, 20]}), {'region': 'a'})
    rest = list(batches)
    pinned = pa.concat_tables([first] + rest).to_pandas()
    assert sorted(pinned['amount'].tolist()) == [1, 2, 3]
    assert sorted(ds.read_dataframe()['amount'].tolist()) == [3, 10, 20]


def test_snapshot_holds_files_even_if_removed(dataset):
    ds = ParquetDataset(dataset, partition_on=['region'])
    ds.append(pd.DataFrame({'region': ['a', 'b'], 'amount': [1, 2]}))
    snapshot = ds.snapshot()
    snapshot.read_table()  # open the files
    for record in snapshot.manifest.files:
        os.remove(os.path.join(dataset, record['path']))
    assert sorted(snapshot.read_table().to_pandas()['amount'].tolist()) == [1,
                                                                            2]


# -- column selection and partition filters --------------------------------

def test_column_selection(dataset):
    ds = ParquetDataset(dataset, partition_on=['region'])
    ds.append(make_frame())
    result = ds.read_dataframe(columns=['amount', 'region'])
    assert list(result.columns) == ['amount', 'region']


def test_partition_filter_with_list(dataset):
    ds = ParquetDataset(dataset, partition_on=['region'])
    ds.append(make_frame())
    result = ds.read_dataframe(partitions={'region': ['west', 'east']})
    assert sorted(result['amount'].tolist()) == [1, 2, 3]


def test_partition_filter_composes_with_columns(dataset):
    ds = ParquetDataset(dataset, partition_on=['region'])
    ds.append(make_frame())
    result = ds.read_dataframe(columns=['amount'],
                               partitions={'region': 'west'})
    assert list(result.columns) == ['amount']
    assert sorted(result['amount'].tolist()) == [1, 3]


# -- ignoring unrelated files ---------------------------------------------

def test_unrelated_files_are_ignored(dataset):
    ds = ParquetDataset(dataset, partition_on=['region'])
    ds.append(pd.DataFrame({'region': ['a'], 'amount': [1]}))
    with open(os.path.join(dataset, 'README.txt'), 'w') as f:
        f.write('not data')
    os.makedirs(os.path.join(dataset, 'notes'))
    with open(os.path.join(dataset, 'notes', 'extra.parquet'), 'w') as f:
        f.write('also not parquet')
    assert ds.read_dataframe()['amount'].tolist() == [1]


# -- adopting existing Parquet directories --------------------------------

def test_adopts_hive_partitioned_directory(dataset):
    os.makedirs(dataset)
    frame = pd.DataFrame({'region': ['west', 'east', 'west'],
                          'amount': [1, 2, 3]})
    pq.write_to_dataset(
        pa.Table.from_pandas(frame, preserve_index=False),
        root_path=dataset, partition_cols=['region'],
        basename_template='f{i}.parquet',
        existing_data_behavior='overwrite_or_ignore',
    )
    ds = resource(dataset)
    result = ds.read_dataframe()
    assert sorted(result['amount'].tolist()) == [1, 2, 3]
    assert set(ds.manifest().partition_on) == {'region'}
    assert sorted(result['region'].unique()) == ['east', 'west']


# -- drop -----------------------------------------------------------------

def test_drop(dataset):
    ds = ParquetDataset(dataset, partition_on=['region'])
    ds.append(pd.DataFrame({'region': ['a'], 'amount': [1]}))
    drop(ds)
    assert not os.path.exists(dataset)


# -- null preservation in non-partition columns --------------------------

def test_nulls_are_preserved_not_zeroed(dataset):
    ds = ParquetDataset(dataset, partition_on=['region'])
    ds.append(pd.DataFrame({'region': ['a'], 'v': [1]}))
    ds.append(pd.DataFrame({'region': ['b'], 'v': [None]}))
    records = sorted(ds.read_dataframe().to_dict('records'),
                     key=lambda r: r['region'])
    assert records[0]['v'] == 1
    assert pd.isnull(records[1]['v'])
    # A column that gains a null in a later write becomes optional.
    assert str(discover(ds).measure.types[1]) == '?int64'


# -- ingesting CSV and SQL sources ---------------------------------------

def test_csv_into_dataset(tmpdir=None):
    from odo.utils import filetext
    text = 'region,amount\nwest,1\neast,2\nwest,3\n'
    with filetext(text, extension='.csv') as path:
        with tmpfile('.parquet') as dataset_path:
            ds = resource(dataset_path,
                          dshape='var * {region: string, amount: int64}',
                          partition_on=['region'])
            into(ds, path)
            records = sorted(ds.read_dataframe().to_dict('records'),
                             key=lambda r: r['amount'])
            assert [r['amount'] for r in records] == [1, 2, 3]
            assert sorted(r['region'] for r in records) == ['east', 'west',
                                                            'west']


def test_csv_to_dataset_uri():
    from odo.utils import filetext
    text = 'region,amount\nwest,1\neast,2\nwest,3\n'
    with filetext(text, extension='.csv') as path:
        with tmpfile('.parquet') as dataset_path:
            odo(path, dataset_path,
                dshape='var * {region: string, amount: int64}',
                partition_on=['region'])
            ds = resource(dataset_path)
            assert sorted(ds.read_dataframe()['amount'].tolist()) == [1, 2, 3]


def test_sql_select_into_dataset():
    import sqlalchemy as sa
    from odo.backends.sql import create_engine
    with tmpfile('.db') as db:
        engine = create_engine('sqlite:///' + db)
        metadata = sa.MetaData(bind=engine)
        table = sa.Table('accounts', metadata,
                         sa.Column('region', sa.String),
                         sa.Column('amount', sa.Integer))
        metadata.create_all()
        engine.execute(table.insert(), [
            {'region': 'north', 'amount': 10},
            {'region': 'south', 'amount': 20},
        ])
        with tmpfile('.parquet') as dataset_path:
            ds = resource(dataset_path,
                          dshape='var * {region: string, amount: int64}',
                          partition_on=['region'])
            odo(sa.select([table]), ds)
            records = sorted(ds.read_dataframe().to_dict('records'),
                             key=lambda r: r['amount'])
            assert [r['amount'] for r in records] == [10, 20]
            assert [r['region'] for r in records] == ['north', 'south']


def test_manifest_publish_failure_is_atomic(dataset, monkeypatch):
    import odo.backends.parquet as parquet
    ds = ParquetDataset(dataset, partition_on=['region'])
    ds.append(pd.DataFrame({'region': ['a', 'a'], 'v': [1, 2]}))
    before = [dict(f) for f in ds.manifest().files]

    original = parquet._atomic_write

    def fail_current(path, data):
        if os.path.basename(path) == 'CURRENT':
            raise IOError('simulated manifest failure')
        return original(path, data)

    monkeypatch.setattr(parquet, '_atomic_write', fail_current)
    with pytest.raises(IOError):
        ds.append(pd.DataFrame({'region': ['a'], 'v': [99]}))
    monkeypatch.setattr(parquet, '_atomic_write', original)

    assert ds.manifest().files == before
    assert sorted(ds.read_dataframe()['v'].tolist()) == [1, 2]


def test_awkward_partition_values_roundtrip(dataset):
    ds = ParquetDataset(dataset, partition_on=['region'])
    values = ['a/b', 'x y', 'ü', '\\', '=', '__HIVE_DEFAULT_PARTITION__']
    ds.append(pd.DataFrame({'region': values, 'v': list(range(len(values)))}))
    for expected, value in enumerate(values):
        assert ds.read_dataframe(partitions={'region': value})['v'].tolist() \
            == [expected]


def test_partition_named_like_null_sentinel_is_not_null(dataset):
    from odo.backends.parquet import NULL_SENTINEL
    ds = ParquetDataset(dataset, partition_on=['region'])
    ds.append(pd.DataFrame({'region': [NULL_SENTINEL, None], 'v': [1, 2]}))
    assert ds.read_dataframe(partitions={'region': NULL_SENTINEL})[
        'v'].tolist() == [1]
    assert ds.read_dataframe(partitions={'region': None})['v'].tolist() == [2]


def test_chunked_read_honours_columns_and_filters(dataset):
    ds = ParquetDataset(dataset, partition_on=['region'])
    ds.append(pd.DataFrame({'region': ['a', 'b', 'a', 'b'],
                            'v': [1, 2, 3, 4]}))
    result = convert(chunks(pd.DataFrame), ds, columns=['v', 'region'],
                     partitions={'region': 'a'}, chunksize=1)
    frames = list(result)
    combined = pd.concat(frames)
    assert list(combined.columns) == ['v', 'region']
    assert sorted(combined['v'].tolist()) == [1, 3]


def test_snapshot_context_manager(dataset):
    ds = ParquetDataset(dataset, partition_on=['region'])
    ds.append(pd.DataFrame({'region': ['a'], 'v': [1]}))
    with ds.snapshot() as snapshot:
        assert snapshot.read_table().num_rows == 1


def test_generator_input_is_not_consumed_twice(dataset):
    ds = ParquetDataset(dataset, partition_on=['region'])

    def rows():
        for i in range(4):
            yield {'region': 'r%d' % (i % 2), 'v': i}

    ds.append(rows())
    assert sorted(ds.read_dataframe()['v'].tolist()) == [0, 1, 2, 3]


def test_zero_column_selection_keeps_rows(dataset):
    ds = ParquetDataset(dataset, partition_on=['region'])
    ds.append(pd.DataFrame({'region': ['a', 'b'], 'v': [1, 2]}))
    assert ds.read_table(columns=[]).num_rows == 2
    assert ds.read_dataframe(columns=[]).shape == (2, 0)


def test_multiple_partition_keys(dataset):
    ds = ParquetDataset(
        dataset,
        dshape='var * {region: ?string, year: int64, amount: float64}',
        partition_on=['region', 'year'],
    )
    ds.append(pd.DataFrame({
        'region': ['w', 'w', 'e', None, ''],
        'year': [2020, 2020, 2021, 2020, 2020],
        'amount': [1.0, 2.0, 3.0, 4.0, 5.0],
    }))
    assert len(ds.read_dataframe()) == 5
    selected = ds.read_dataframe(partitions={'region': 'w', 'year': 2020})
    assert sorted(selected['amount'].tolist()) == [1.0, 2.0]
    keys = ds.partitions_list()
    assert {'region': 'w', 'year': 2020} in keys
    assert {'region': None, 'year': 2020} in keys
    assert {'region': '', 'year': 2020} in keys
