"""Simulation RGB keypoints with separate validation; labels never enter inference."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_agent.keypoint_detector import KeypointNet
from learn import DATA, ROOT, digest


def load(items):
    arrays = [np.load(DATA / row['dataset_relative_path']) for row in items]
    return [np.concatenate([a[key] for a in arrays]) for key in ('rgb', 'keypoints', 'visible')]


def main():
    config = json.loads((ROOT / 'experiments/event_residual_mvp_v2/perception_design.json').read_text())['training']
    manifest = json.loads((ROOT / 'experiments/event_residual_mvp_v2/perception_dataset.json').read_text())
    auxiliary = json.loads((ROOT / 'experiments/event_residual_mvp_v2/perception_auxiliary.json').read_text())
    rgb, points, visible = load([r for r in manifest['sequences'] if r['split'] == 'train'] + [auxiliary])
    torch.manual_seed(config['seed']); torch.set_num_threads(1)
    dataset = TensorDataset(torch.from_numpy(rgb.transpose(0, 3, 1, 2).copy()),
                            torch.from_numpy(points).float(), torch.from_numpy(visible).float())
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    model = KeypointNet().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    y, x = torch.meshgrid(torch.arange(60, device='cuda'), torch.arange(80, device='cuda'), indexing='ij')
    frequencies = torch.as_tensor(visible.mean(axis=0), device='cuda').float()
    positive_weight = (1 - frequencies) / frequencies.clamp_min(.01)
    history = []
    started = time.monotonic()
    for epoch in range(config['epochs']):
        total = 0.
        for images, locations, seen in loader:
            images, locations, seen = images.cuda().float() / 255, locations.cuda() / 4, seen.cuda()
            logits, visibility_logits = model(images)
            gaussian = torch.exp(-((x[None, None] - locations[:, :, 0, None, None]) ** 2 +
                                   (y[None, None] - locations[:, :, 1, None, None]) ** 2) / 2)
            targets = gaussian.flatten(2)
            targets = targets / targets.sum(-1, keepdim=True).clamp_min(1e-9)
            heatmap_loss = -(targets * logits.flatten(2).log_softmax(-1)).sum(-1)
            loss = (heatmap_loss * seen).sum() / seen.sum().clamp_min(1)
            loss += F.binary_cross_entropy_with_logits(visibility_logits, seen, pos_weight=positive_weight)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(images)
        history.append(dict(epoch=epoch + 1, loss=total / len(dataset)))
        if (epoch + 1) % 5 == 0: print('KEYPOINT_TRAINING', history[-1], flush=True)
    destination = DATA / 'self_evolution/event_residual_mvp_v2/perception/keypoint_model.pt'
    torch.save(model.cpu().state_dict(), destination)
    result = dict(config=config, history=history, samples=len(dataset), visible_counts=visible.sum(axis=0).tolist(),
        wall_s=time.monotonic()-started, weights=dict(dataset_relative_path=str(destination.relative_to(DATA)), sha256=digest(destination)),
        heldout_used_for_training=False, validation_used_for_training=False,
        source_hashes={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__), ROOT/'robot_agent/keypoint_detector.py']})
    (ROOT / 'experiments/event_residual_mvp_v2/keypoint_training.json').write_text(json.dumps(result, indent=2) + '\n')
    print('KEYPOINT_COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
