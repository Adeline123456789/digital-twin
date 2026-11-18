"""Cost the common primary inputs under the current OS cache, outside replay timings."""
import csv
import io
import json
from pathlib import Path
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'scripts'))
from learn import DATA


def sample(case):
    ep = DATA / case['episode']
    if case['domain'] == 'real':
        p = ep / 'raw/task'
        paths = [p/n for n in ('commanded_probe.csv', 'execution_report.json',
                 'observed_execution_report.json', 'right_wrist/camera.json', 'external/camera.json')]
    else:
        paths = [DATA / 'self_evolution' / case['id'] / 'trajectory.jsonl',
                 ep / 'proprio_stats/privileged.jsonl']
    read_s = parse_s = size = 0
    for path in paths:
        start = perf_counter()
        raw = path.read_bytes()
        read_s += perf_counter() - start
        size += len(raw)
        start = perf_counter()
        if path.suffix == '.csv':
            rows = list(csv.DictReader(io.StringIO(raw.decode())))
        elif path.suffix == '.jsonl':
            rows = [json.loads(line) for line in raw.splitlines()]
        else:
            rows = json.loads(raw)
        parse_s += perf_counter() - start
        assert rows
    return dict(id=case['id'], bytes=size, file_read_s=read_s, structured_decode_s=parse_s)


if __name__ == '__main__':
    cases = json.loads((HERE / 'design.json').read_text())['cases']
    print(json.dumps([sample(case) for case in cases], indent=2))
