# V2 local runtime

The existing native simulator environment is unchanged. The existing loop_act Python provides Torch; a project-local overlay pins MuJoCo 3.3.7, Shapely 2.1.2 and Stable-Baselines3 2.9.2. The declarative environment is `compute_env_spec.json`.

Run from `/media/midea/F496AC8096AC4548/digital twin`:

```bash
PYTHONPATH=.local/event_residual_mvp_v2/runtime OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl /home/midea/miniconda3/envs/loop_act/bin/python scripts/verify_residual_runtime.py
```

Expected: exit 0, `KERNEL_WITNESS` for an 8×8 CUDA matmul, then `LOCAL_REPLAY_WITNESS` with `is_success: true`, zero correction, zero integration-state error at policy samples and unchanged nominal hash. This executes local frozen-reference physics without hardware or training; it writes only `.local/event_residual_mvp_v2/runtime_witness.json`. Typical wall time is 20 seconds. Labels used for the replay comparison are evaluator-only.
