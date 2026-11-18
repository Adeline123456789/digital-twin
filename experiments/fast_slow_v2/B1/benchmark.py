"""Freeze same-input history and re-run the legacy diagnostic without its publishing main."""
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from diagnose_retention import diagnose_episode
from learn import DATA, SIM, digest, version


def measure():
    folder = ROOT / "experiments/research_stage4"
    design = json.loads((folder / "round3/design.json").read_text())
    cases = []
    for split in ("development_episodes", "heldout_success_episodes"):
        for ident in design[split]:
            path, = (ROOT / "experiments").glob(f"*/{ident}/result.json")
            r = json.loads(path.read_text())
            start = time.perf_counter()
            diagnosis = diagnose_episode(r)
            cases.append(dict(split=split, report=str(path.relative_to(ROOT)), sha256=digest(path),
                              profile=r['actions'][0]['profile'], native=r['program_version']['source_sha256']['scripts/try_plug_pick_sim.py'],
                              target=r['initial_state']['target_index'], status=diagnosis['status'],
                              diagnosis_seconds=time.perf_counter()-start, task_success=r['task_success']))
    cli = []
    for _ in range(3):
        start = time.perf_counter()
        subprocess.run(["codex", "--version"], check=True, stdout=subprocess.DEVNULL)
        cli.append(time.perf_counter()-start)
    v = version()
    legacy = folder / 'round4/result.json'
    return dict(cases=cases, historical_planner_report=str(legacy.relative_to(ROOT)), historical_planner_sha256=digest(legacy),
                historical_metrics=json.loads(legacy.read_text())['metrics'],
                source_head=v['git_head'], source_tree_sha256=hashlib.sha256(json.dumps(v['source_sha256'],sort_keys=True).encode()).hexdigest(),
                current_native=digest(SIM/'scripts/try_plug_pick_sim.py'),
                retained_native=digest(DATA/'calibration/offline_workspace_20261110/scripts/try_plug_pick_sim.py'),
                source_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=SIM)),
                source_dirty_diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','HEAD','--binary'],cwd=SIM)).hexdigest(),
                codex_version_process_seconds=cli, cli_startup_median=statistics.median(cli),
                turn_context_overhead='NOT_MEASURED; version startup is only a lower-level process benchmark',
                fresh_generation_simulation='NOT_RUN; comparison uses frozen histories',original_unittest=dict(run=126,failed=3,errors=4),
                corrected_pytest=131, original_sha='f3a0fe62e115930d955a5a74fc9f3629dae6565d')


if __name__ == '__main__':
    print(json.dumps(measure(),indent=2))
