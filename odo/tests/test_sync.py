from __future__ import absolute_import, division, print_function

import json
import sqlite3
import sys
from datetime import datetime, timedelta

import pytest

sa = pytest.importorskip('sqlalchemy')

from datashape import discover

from odo import odo, resource
from odo.sync import sync, CheckpointMismatchError
from odo.utils import tmpfile

# ``odo.sync`` the attribute is the function (it shadows the submodule),
# so grab the module itself for monkeypatching.
sync_module = sys.modules['odo.sync']


ds_text = ('var * {id: int64, updated_at: datetime, value: float64, '
           'name: ?string}')

t0 = datetime(2020, 1, 1, 0, 0, 0)
t1 = datetime(2020, 1, 1, 1, 0, 0)
t2 = datetime(2020, 1, 1, 2, 0, 0)
t3 = datetime(2020, 1, 1, 3, 0, 0)


def make_rows(n, start=0):
    return [(i, t0 + timedelta(hours=i), float(i), 'name-%d' % i)
            for i in range(start, start + n)]


def make_source(path, rows, table='events'):
    uri = 'sqlite:///%s::%s' % (path, table)
    odo(rows, uri, dshape=ds_text)
    return uri


def read_state(path, sync_id):
    """Read the checkpoint straight from the target file, bypassing odo.

    This is what a restarted process would see.
    """
    conn = sqlite3.connect(path)
    try:
        return conn.execute(
            'SELECT sync_id, fingerprint, position FROM _odo_sync_state '
            'WHERE sync_id = ?', (sync_id,)).fetchone()
    finally:
        conn.close()


def test_sync_copies_all_rows_in_cursor_order():
    with tmpfile('.db') as src, tmpfile('.db') as dst:
        rows = make_rows(5)
        src_uri = make_source(src, rows)
        target = sync(src_uri, 'sqlite:///%s::copy' % dst,
                      cursor=('updated_at', 'id'), batch_size=2)
        assert odo(target, list) == rows


def test_sync_preserves_dshape_and_python_types():
    with tmpfile('.db') as src, tmpfile('.db') as dst:
        rows = make_rows(3)
        src_uri = make_source(src, rows)
        target = sync(src_uri, 'sqlite:///%s::copy' % dst,
                      cursor=('updated_at', 'id'))
        assert str(discover(target)) == str(discover(resource(src_uri)))
        row = odo(target, list)[0]
        assert isinstance(row[0], int)
        assert isinstance(row[1], datetime)
        assert isinstance(row[2], float)
        assert isinstance(row[3], str)


def test_checkpoint_is_persisted_in_target_database():
    with tmpfile('.db') as src, tmpfile('.db') as dst:
        make_source(src, make_rows(3))
        sync('sqlite:///%s::events' % src, 'sqlite:///%s::copy' % dst,
             cursor=('updated_at', 'id'), batch_size=2)
        row = read_state(dst, 'copy')
        assert row is not None
        sync_id, fingerprint, position = row
        assert fingerprint
        position = json.loads(position)
        # the cursor of the last committed row: (updated_at, id)
        assert position[0] == {'__datetime__': t2.isoformat()}
        assert position[1] == 2


def test_second_run_only_copies_new_rows():
    with tmpfile('.db') as src, tmpfile('.db') as dst:
        rows = make_rows(3)
        src_uri = make_source(src, rows)
        dst_uri = 'sqlite:///%s::copy' % dst
        sync(src_uri, dst_uri, cursor=('updated_at', 'id'))

        more = make_rows(2, start=3)
        odo(more, src_uri, dshape=ds_text)
        target = sync(src_uri, dst_uri, cursor=('updated_at', 'id'))

        assert odo(target, list) == rows + more
        position = json.loads(read_state(dst, 'copy')[2])
        assert position[1] == 4


def test_rerun_with_no_new_data_is_a_noop():
    with tmpfile('.db') as src, tmpfile('.db') as dst:
        rows = make_rows(3)
        src_uri = make_source(src, rows)
        dst_uri = 'sqlite:///%s::copy' % dst
        sync(src_uri, dst_uri, cursor=('updated_at', 'id'))
        target = sync(src_uri, dst_uri, cursor=('updated_at', 'id'))
        assert odo(target, list) == rows


def test_empty_source_creates_empty_target_and_pins_definition():
    with tmpfile('.db') as src, tmpfile('.db') as dst:
        src_uri = 'sqlite:///%s::events' % src
        resource(src_uri, dshape=ds_text)  # empty source table
        target = sync(src_uri, 'sqlite:///%s::copy' % dst,
                      cursor=('updated_at', 'id'))
        assert odo(target, list) == []
        assert str(discover(target)) == str(discover(resource(src_uri)))
        row = read_state(dst, 'copy')
        assert row is not None
        assert row[2] is None  # no position committed yet


def test_resume_after_failure_neither_duplicates_nor_skips(monkeypatch):
    with tmpfile('.db') as src, tmpfile('.db') as dst:
        # rows 2 and 3 share a timestamp; batch_size=2 splits them across
        # the commit boundary
        rows = [(1, t0, 1.0, 'a'),
                (2, t1, 2.0, 'b'),
                (3, t1, 3.0, 'c'),
                (4, t2, 4.0, 'd'),
                (5, t3, 5.0, 'e')]
        src_uri = make_source(src, rows)
        dst_uri = 'sqlite:///%s::copy' % dst

        real_insert = sync_module._insert_batch
        calls = []

        def flaky_insert(conn, table, batch):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError('simulated disconnect')
            return real_insert(conn, table, batch)

        monkeypatch.setattr(sync_module, '_insert_batch', flaky_insert)
        with pytest.raises(RuntimeError):
            sync(src_uri, dst_uri, cursor=('updated_at', 'id'), batch_size=2)
        monkeypatch.undo()

        # only the first batch committed, both data and checkpoint
        target = resource(dst_uri)
        assert [r[0] for r in odo(target, list)] == [1, 2]
        position = json.loads(read_state(dst, 'copy')[2])
        assert position == [{'__datetime__': t1.isoformat()}, 2]

        # the next run resumes after the last committed batch: the row
        # sharing the committed timestamp is not skipped and the committed
        # rows are not duplicated
        sync(src_uri, dst_uri, cursor=('updated_at', 'id'), batch_size=2)
        assert odo(target, list) == rows


def test_rows_arriving_during_a_round_wait_for_the_next(monkeypatch):
    with tmpfile('.db') as src, tmpfile('.db') as dst:
        rows = make_rows(3)
        src_uri = make_source(src, rows)
        dst_uri = 'sqlite:///%s::copy' % dst

        real_bound = sync_module._current_upper_bound

        def bound_then_inject(engine, table, cursor_columns):
            upper = real_bound(engine, table, cursor_columns)
            # a concurrent writer appends newer rows right after the
            # upper bound of this round was fixed
            odo(make_rows(2, start=3), src_uri, dshape=ds_text)
            return upper

        monkeypatch.setattr(sync_module, '_current_upper_bound',
                            bound_then_inject)
        target = sync(src_uri, dst_uri, cursor=('updated_at', 'id'),
                      batch_size=2)
        assert [r[0] for r in odo(target, list)] == [0, 1, 2]
        monkeypatch.undo()

        # the late rows are picked up by the next round
        sync(src_uri, dst_uri, cursor=('updated_at', 'id'), batch_size=2)
        assert [r[0] for r in odo(target, list)] == [0, 1, 2, 3, 4]


def test_single_column_cursor_and_string_form():
    with tmpfile('.db') as src, tmpfile('.db') as dst:
        rows = make_rows(4)
        src_uri = make_source(src, rows)
        dst_uri = 'sqlite:///%s::copy' % dst
        sync(src_uri, dst_uri, cursor='updated_at', batch_size=1)
        more = make_rows(2, start=4)
        odo(more, src_uri, dshape=ds_text)
        target = sync(src_uri, dst_uri, cursor='updated_at', batch_size=1)
        assert odo(target, list) == rows + more


def test_source_and_target_accept_table_objects():
    with tmpfile('.db') as src, tmpfile('.db') as dst:
        rows = make_rows(3)
        src_uri = make_source(src, rows)
        target = sync(resource(src_uri), 'sqlite:///%s::copy' % dst,
                      cursor=('updated_at', 'id'))
        assert isinstance(target, sa.Table)
        assert odo(target, list) == rows


def test_different_source_database_is_refused():
    with tmpfile('.db') as src, tmpfile('.db') as src2, tmpfile('.db') as dst:
        rows = make_rows(3)
        src_uri = make_source(src, rows)
        dst_uri = 'sqlite:///%s::copy' % dst
        sync(src_uri, dst_uri, cursor=('updated_at', 'id'))

        other_uri = make_source(src2, rows)
        with pytest.raises(CheckpointMismatchError):
            sync(other_uri, dst_uri, cursor=('updated_at', 'id'))
        # the checkpoint and the already copied data are untouched
        assert [r[0] for r in odo(resource(dst_uri), list)] == [0, 1, 2]


def test_different_source_table_is_refused():
    with tmpfile('.db') as src, tmpfile('.db') as dst:
        rows = make_rows(3)
        src_uri = make_source(src, rows)
        dst_uri = 'sqlite:///%s::copy' % dst
        sync(src_uri, dst_uri, cursor=('updated_at', 'id'))

        other_uri = make_source(src, rows, table='other_events')
        with pytest.raises(CheckpointMismatchError):
            sync(other_uri, dst_uri, cursor=('updated_at', 'id'))


def test_different_source_definition_is_refused():
    with tmpfile('.db') as src, tmpfile('.db') as dst:
        rows = make_rows(3)
        src_uri = make_source(src, rows)
        dst_uri = 'sqlite:///%s::copy' % dst
        sync(src_uri, dst_uri, cursor=('updated_at', 'id'))

        # same table name in the same database, different schema
        conn = sqlite3.connect(src)
        conn.execute('DROP TABLE events')
        conn.execute('CREATE TABLE events (id INTEGER, updated_at DATETIME, '
                     'value FLOAT, name TEXT, extra TEXT)')
        conn.commit()
        conn.close()
        engine = sa.create_engine('sqlite:///' + src)
        redefined = sa.Table('events', sa.MetaData(engine),
                             autoload_with=engine)
        with pytest.raises(CheckpointMismatchError):
            sync(redefined, dst_uri, cursor=('updated_at', 'id'))


def test_different_target_definition_is_refused():
    with tmpfile('.db') as src, tmpfile('.db') as dst:
        rows = make_rows(3)
        src_uri = make_source(src, rows)
        dst_uri = 'sqlite:///%s::copy' % dst
        sync(src_uri, dst_uri, cursor=('updated_at', 'id'))

        # the target table was redefined behind our back (e.g. dropped and
        # recreated with another shape by a different process)
        conn = sqlite3.connect(dst)
        conn.execute('DROP TABLE copy')
        conn.execute('CREATE TABLE copy (id INTEGER, updated_at DATETIME, '
                     'value FLOAT, name TEXT, extra TEXT)')
        conn.commit()
        conn.close()
        engine = sa.create_engine('sqlite:///' + dst)
        redefined = sa.Table('copy', sa.MetaData(engine),
                             autoload_with=engine)
        with pytest.raises(CheckpointMismatchError):
            sync(src_uri, redefined, cursor=('updated_at', 'id'))


def test_different_cursor_is_refused():
    with tmpfile('.db') as src, tmpfile('.db') as dst:
        rows = make_rows(3)
        src_uri = make_source(src, rows)
        dst_uri = 'sqlite:///%s::copy' % dst
        sync(src_uri, dst_uri, cursor=('updated_at', 'id'))
        with pytest.raises(CheckpointMismatchError):
            sync(src_uri, dst_uri, cursor='updated_at')
        # a matching definition still resumes fine afterwards
        more = make_rows(1, start=3)
        odo(more, src_uri, dshape=ds_text)
        target = sync(src_uri, dst_uri, cursor=('updated_at', 'id'))
        assert odo(target, list) == rows + more


def test_independent_syncs_share_one_state_table():
    with tmpfile('.db') as src, tmpfile('.db') as dst:
        rows_a = make_rows(3)
        rows_b = make_rows(2, start=10)
        src_a = make_source(src, rows_a, table='events_a')
        src_b = make_source(src, rows_b, table='events_b')
        dst_a = 'sqlite:///%s::copy_a' % dst
        dst_b = 'sqlite:///%s::copy_b' % dst
        sync(src_a, dst_a, cursor=('updated_at', 'id'))
        sync(src_b, dst_b, cursor=('updated_at', 'id'))
        assert odo(resource(dst_a), list) == rows_a
        assert odo(resource(dst_b), list) == rows_b
        assert read_state(dst, 'copy_a') is not None
        assert read_state(dst, 'copy_b') is not None


def test_unknown_cursor_column_raises():
    with tmpfile('.db') as src, tmpfile('.db') as dst:
        src_uri = make_source(src, make_rows(1))
        with pytest.raises(ValueError):
            sync(src_uri, 'sqlite:///%s::copy' % dst,
                 cursor=('updated_at', 'nope'))


def test_sync_id_separates_progress_for_same_target_database():
    with tmpfile('.db') as src, tmpfile('.db') as dst:
        rows = make_rows(3)
        src_uri = make_source(src, rows)
        dst_uri = 'sqlite:///%s::copy' % dst
        sync(src_uri, dst_uri, cursor=('updated_at', 'id'), sync_id='one')
        assert read_state(dst, 'one') is not None
        assert read_state(dst, 'copy') is None
