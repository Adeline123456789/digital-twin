"""Local content-addressed analysis cache; raw datasets and public reports are untouched."""
import hashlib
import json
import sqlite3
import time
from pathlib import Path


class EvidenceCache:
    def __init__(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(directory / 'evidence.sqlite3')
        self.db.execute('CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, stamp TEXT, digest TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS results (key TEXT PRIMARY KEY, value TEXT)')
        self.bytes_hashed = 0

    def close(self):
        self.db.close()

    def fingerprint(self, path):
        path = Path(path).resolve()
        s = path.stat()
        stamp = json.dumps((s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns))
        row = self.db.execute('SELECT stamp,digest FROM files WHERE path=?', (str(path),)).fetchone()
        if row and row[0] == stamp:
            return row[1]
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(chunk)
                self.bytes_hashed += len(chunk)
        after = path.stat()
        if (s.st_size, s.st_mtime_ns, s.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError('Evidence changed during hashing; retry after recording ends')
        value = h.hexdigest()
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO files VALUES (?,?,?)', (str(path), stamp, value))
        return value

    def get(self, name, dependencies, parameters, compute):
        started = time.monotonic()
        before = self.bytes_hashed
        hashes = {key: self.fingerprint(path) for key, path in sorted(dependencies.items())}
        payload = dict(analyzer=name, dependencies=hashes, parameters=parameters)
        key = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
        row = self.db.execute('SELECT value FROM results WHERE key=?', (key,)).fetchone()
        if row:
            value = json.loads(row[0])
        else:
            value = compute()
            with self.db:
                self.db.execute('INSERT OR REPLACE INTO results VALUES (?,?)',
                                (key, json.dumps(value, allow_nan=False)))
        return dict(value=value, cache=dict(key=key, hit=row is not None,
                    bytes_hashed=self.bytes_hashed-before, dependency_sha256=hashes,
                    wall_seconds=time.monotonic()-started))
