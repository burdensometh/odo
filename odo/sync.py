from __future__ import absolute_import, division, print_function

import json
import os
from base64 import b64decode, b64encode
from datetime import date, datetime
from decimal import Decimal
from hashlib import sha256

import sqlalchemy as sa
from datashape import discover

from .backends.sql import getbind
from .compatibility import unicode
from .resource import resource

__all__ = ['sync', 'CheckpointMismatchError']

DEFAULT_STATE_TABLE = '_odo_sync_state'


class CheckpointMismatchError(ValueError):
    """A stored sync checkpoint belongs to a different definition.

    Raised by :func:`sync` when the checkpoint found in the target
    database was written for a different source, target or cursor
    definition, so the stored progress must not be used to continue.
    """


def sync(source, target, cursor, batch_size=1000, sync_id=None,
         state_table=DEFAULT_STATE_TABLE):
    """Resumably synchronize an append-only SQL table into another database.

    ``sync`` copies rows from ``source`` to ``target`` in cursor order and
    is safe to re-run at any time: each call moves only the rows that were
    not committed by previous calls.  It is designed for append-only
    sources whose rows carry a strictly increasing *cursor* -- typically a
    non-null update timestamp plus a unique id, e.g.
    ``cursor=('updated_at', 'id')``.

    Each call performs one round:

    1.  The current maximum cursor of the source is read once and fixed as
        the upper bound of the round.  Rows appended to the source while
        the round is running have larger cursors and are left for the next
        round.
    2.  Only rows with ``last committed cursor < cursor <= upper bound``
        are copied, ordered by the cursor.  A composite cursor is compared
        lexicographically, so rows sharing the same timestamp are never
        skipped or duplicated.
    3.  Rows are copied in batches of ``batch_size``.  Every batch is
        written *together with its new cursor position* in a single
        transaction against the target database, so a crash or disconnect
        can only lose the in-flight batch.  Re-running ``sync`` -- even
        from a restarted process -- resumes after the last successfully
        committed batch.

    Progress is stored in a bookkeeping table (``_odo_sync_state`` by
    default) inside the target database, which is what allows the data
    and the position to commit atomically and to survive restarts.  Next
    to the position, the checkpoint stores a fingerprint of the source,
    target and cursor definitions; if a later call supplies a different
    source, target or cursor for the same ``sync_id``,
    :class:`CheckpointMismatchError` is raised instead of resuming from
    incompatible progress.

    Parameters
    ----------
    source : str or sa.Table
        The append-only source table, either a sqlalchemy table or a URI
        such as ``'sqlite:///src.db::events'``.  Must already exist.
    target : str or sa.Table
        The target table, either a sqlalchemy table or a URI such as
        ``'sqlite:///dst.db::events'``.  When it does not exist yet it is
        created with the dshape discovered from the source, preserving
        column types.
    cursor : str or sequence of str
        Column name(s) forming the strictly increasing cursor, compared
        lexicographically in the given order, e.g.
        ``('updated_at', 'id')``.
    batch_size : int, optional, default 1000
        Number of rows copied per transaction.
    sync_id : str, optional
        Identifier of this synchronization inside the state table.
        Defaults to the target table name.
    state_table : str, optional
        Name of the bookkeeping table created in the target database.

    Returns
    -------
    sa.Table
        The target table.

    Examples
    --------
    >>> from odo import odo
    >>> from odo.sync import sync
    >>> from odo.utils import tmpfile
    >>> from datetime import datetime, timedelta
    >>> ds = 'var * {id: int64, updated_at: datetime, value: float64}'
    >>> rows = [(i, datetime(2016, 1, 1) + timedelta(hours=i), float(i))
    ...         for i in range(4)]
    >>> with tmpfile('.db') as src, tmpfile('.db') as dst:
    ...     _ = odo(rows, 'sqlite:///%s::events' % src, dshape=ds)
    ...     t = sync('sqlite:///%s::events' % src,
    ...              'sqlite:///%s::events' % dst,
    ...              cursor=('updated_at', 'id'), batch_size=2)
    ...     odo(t, list) == rows
    True

    Calling ``sync`` again only moves the rows appended since the last
    committed batch:

    >>> with tmpfile('.db') as src, tmpfile('.db') as dst:
    ...     _ = odo(rows, 'sqlite:///%s::events' % src, dshape=ds)
    ...     t = sync('sqlite:///%s::events' % src,
    ...              'sqlite:///%s::events' % dst,
    ...              cursor=('updated_at', 'id'))
    ...     more = [(4, datetime(2016, 1, 1) + timedelta(hours=4), 4.0)]
    ...     _ = odo(more, 'sqlite:///%s::events' % src, dshape=ds)
    ...     t = sync('sqlite:///%s::events' % src,
    ...              'sqlite:///%s::events' % dst,
    ...              cursor=('updated_at', 'id'))
    ...     odo(t, list) == rows + more
    True
    """
    cursor = _normalize_cursor(cursor)
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer, got %r'
                         % (batch_size,))

    source = _coerce_source(source)
    missing = [name for name in cursor if name not in source.c]
    if missing:
        raise ValueError('cursor column(s) %s not found in source table %r'
                         % (', '.join(map(repr, missing)), source.name))
    source_engine = getbind(source, None)
    if source_engine is None:
        raise ValueError('source table %r is not bound to an engine; pass a '
                         'URI string or a bound table' % source.name)
    source_dshape = discover(source)

    target = _coerce_target(target, source_dshape)
    target_engine = getbind(target, None)
    if target_engine is None:
        raise ValueError('target table %r is not bound to an engine; pass a '
                         'URI string or a bound table' % target.name)
    if not target.exists(bind=target_engine):
        target.create(bind=target_engine)

    sync_id = sync_id if sync_id is not None else target.name
    state = _state_table(state_table)
    state.create(bind=target_engine, checkfirst=True)
    fingerprint = _fingerprint(source, source_engine, target, target_engine,
                               cursor)

    last = None
    with target_engine.begin() as conn:
        row = conn.execute(
            sa.select([state.c.fingerprint, state.c.position])
            .where(state.c.sync_id == sync_id)
        ).fetchone()
        if row is not None:
            if row[state.c.fingerprint] != fingerprint:
                raise CheckpointMismatchError(
                    'checkpoint %r in state table %r was written for a '
                    'different source, target or cursor definition; '
                    'refusing to resume from it. If this change was '
                    'intentional, remove the stale checkpoint row or use a '
                    'different sync_id.' % (sync_id, state_table))
            if row[state.c.position] is not None:
                last = _decode_position(row[state.c.position])
        else:
            # Pin the definition even before the first batch commits, so a
            # later call with a different definition is refused.
            conn.execute(state.insert().values(sync_id=sync_id,
                                               fingerprint=fingerprint))

    cursor_columns = [source.c[name] for name in cursor]
    upper = _current_upper_bound(source_engine, source, cursor_columns)
    if upper is None:
        return target  # empty source, nothing to copy
    if last is not None and not tuple(last) < tuple(upper):
        return target  # no new rows since the last committed batch

    select = _incremental_select(source, cursor_columns, last, upper)
    column_names = [c.name for c in source.columns]
    with source_engine.connect() as source_conn:
        result = source_conn.execute(select)
        while True:
            rows = result.fetchmany(batch_size)
            if not rows:
                break
            batch = [dict(zip(column_names, row)) for row in rows]
            position = _encode_position([rows[-1][name] for name in cursor])
            with target_engine.begin() as target_conn:
                _insert_batch(target_conn, target, batch)
                _write_checkpoint(target_conn, state, sync_id, fingerprint,
                                  position)
    return target


def _normalize_cursor(cursor):
    if isinstance(cursor, (str, unicode)):
        cursor = (cursor,)
    cursor = tuple(cursor)
    if not cursor:
        raise ValueError('cursor must name at least one column')
    return cursor


def _coerce_source(source):
    if isinstance(source, (str, unicode)):
        source = resource(source)
    if not isinstance(source, sa.Table):
        raise TypeError('source must be a sqlalchemy.Table or a URI string '
                        'resolving to one, got %r' % type(source).__name__)
    return source


def _coerce_target(target, source_dshape):
    if isinstance(target, (str, unicode)):
        try:
            target = resource(target)
        except ValueError:
            # The table does not exist yet: create it from the source
            # dshape so column types are preserved.
            target = resource(target, dshape=source_dshape)
    if not isinstance(target, sa.Table):
        raise TypeError('target must be a sqlalchemy.Table or a URI string '
                        'resolving to one, got %r' % type(target).__name__)
    return target


def _state_table(name):
    return sa.Table(
        name, sa.MetaData(),
        sa.Column('sync_id', sa.String(255), primary_key=True),
        sa.Column('fingerprint', sa.String(64), nullable=False),
        sa.Column('position', sa.Text),
        sa.Column('committed_at', sa.DateTime),
    )


def _engine_identity(engine):
    url = engine.url
    if url.drivername.startswith('sqlite'):
        database = url.database
        if database and database != ':memory:':
            database = os.path.abspath(database)
        return ['sqlite', database]
    return [url.drivername, url.host, url.port, url.database]


def _fingerprint(source, source_engine, target, target_engine, cursor):
    payload = {
        'source': {
            'engine': _engine_identity(source_engine),
            'schema': source.schema,
            'table': source.name,
            'dshape': str(discover(source)),
        },
        'target': {
            'engine': _engine_identity(target_engine),
            'schema': target.schema,
            'table': target.name,
            'dshape': str(discover(target)),
        },
        'cursor': list(cursor),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return sha256(blob.encode('utf-8')).hexdigest()


def _tuple_gt(columns, values):
    """(c0, c1, ...) > (v0, v1, ...), lexicographically."""
    if len(columns) == 1:
        return columns[0] > values[0]
    return sa.or_(*[
        sa.and_(*([columns[j] == values[j] for j in range(i)] +
                  [columns[i] > values[i]]))
        for i in range(len(columns))
    ])


def _tuple_lte(columns, values):
    """(c0, c1, ...) <= (v0, v1, ...), lexicographically."""
    if len(columns) == 1:
        return columns[0] <= values[0]
    clauses = [
        sa.and_(*([columns[j] == values[j] for j in range(i)] +
                  [columns[i] < values[i]]))
        for i in range(len(columns))
    ]
    clauses.append(sa.and_(*[c == v for c, v in zip(columns, values)]))
    return sa.or_(*clauses)


def _current_upper_bound(engine, table, cursor_columns):
    """The maximum cursor currently committed in the source, or None."""
    select = (sa.select(list(cursor_columns))
              .order_by(*[c.desc() for c in cursor_columns])
              .limit(1))
    with engine.connect() as conn:
        row = conn.execute(select).fetchone()
    return tuple(row) if row is not None else None


def _incremental_select(table, cursor_columns, last, upper):
    conditions = [_tuple_lte(cursor_columns, upper)]
    if last is not None:
        conditions.append(_tuple_gt(cursor_columns, last))
    return (sa.select(list(table.columns))
            .where(sa.and_(*conditions))
            .order_by(*cursor_columns))


def _insert_batch(conn, table, rows):
    conn.execute(table.insert(), list(rows))


def _write_checkpoint(conn, state, sync_id, fingerprint, position):
    values = dict(fingerprint=fingerprint, position=position,
                  committed_at=datetime.utcnow())
    result = conn.execute(
        state.update().where(state.c.sync_id == sync_id).values(**values))
    if result.rowcount == 0:
        conn.execute(state.insert().values(sync_id=sync_id, **values))


def _encode_position(values):
    return json.dumps([_encode_value(value) for value in values])


def _decode_position(blob):
    return [_decode_value(value) for value in json.loads(blob)]


def _encode_value(value):
    if isinstance(value, datetime):
        return {'__datetime__': value.isoformat()}
    if isinstance(value, date):
        return {'__date__': value.isoformat()}
    if isinstance(value, Decimal):
        return {'__decimal__': str(value)}
    if isinstance(value, bytes):
        return {'__bytes__': b64encode(value).decode('ascii')}
    return value


def _decode_value(value):
    if isinstance(value, dict) and len(value) == 1:
        tag, raw = next(iter(value.items()))
        if tag == '__datetime__':
            return _parse_datetime(raw)
        if tag == '__date__':
            return _parse_date(raw)
        if tag == '__decimal__':
            return Decimal(raw)
        if tag == '__bytes__':
            return b64decode(raw.encode('ascii'))
    return value


def _parse_datetime(text):
    for fmt in ('%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S'):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError('cannot parse checkpoint datetime %r' % (text,))


def _parse_date(text):
    return datetime.strptime(text, '%Y-%m-%d').date()
