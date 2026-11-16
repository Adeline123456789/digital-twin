"""Observe existing native grasp trials without replacing their physics or motion."""
import json
import sys
from pathlib import Path
from learn import SIM
from grasp_contact_probe import contact_snapshot

sys.path.insert(0, str(SIM / 'scripts'))
import try_plug_pick_sim as native


if __name__ == '__main__':
    original = native.MujocoBackend

    class ObservedBackend(original):
        def save_overview(self, path):
            super().save_overview(path)
            path = Path(path)
            if path.stem in ('grasp', 'lift'):
                path.with_suffix('.contacts.json').write_text(json.dumps(contact_snapshot(self), indent=2))

    native.MujocoBackend = ObservedBackend
    native.main()
