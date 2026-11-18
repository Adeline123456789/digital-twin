"""Simulation-only falsifiers. A restored snapshot cannot rule out earlier state writes."""
from math import dist


def attachment_flags(model,data,object_name='task/yellow_plug'):
    import mujoco
    obj=model.body(object_name).id
    attached=[]
    node=obj
    while node:
        if model.body_mocapid[node]>=0:
            attached.append('mocap-driven object')
        node=int(model.body_parentid[node])
    if model.body_weldid[obj]!=obj or not model.body_jntnum[obj]:
        attached.append('object fixed to parent')
    for i in range(model.neq):
        if not data.eq_active[i] or model.eq_type[i] not in (mujoco.mjtEq.mjEQ_WELD,mujoco.mjtEq.mjEQ_CONNECT):
            continue
        ids=[int(model.eq_obj1id[i]),int(model.eq_obj2id[i])]
        if model.eq_objtype[i]==mujoco.mjtObj.mjOBJ_SITE:
            ids=[int(model.site_bodyid[j]) for j in ids if j>=0]
        if any(j>=0 and model.body_weldid[j]==model.body_weldid[obj] for j in ids):
            attached.append('active object attachment')
    return attached


def check_grasp(flags,samples,*,direct_pose_write=None):
    if flags or direct_pose_write is True:
        return 'REJECT_AND_ROLLBACK'
    # A sustained, lifted object rigidly following open fingers without support falsifies a carry.
    window=[]
    for s in samples:
        if not (s['phase'] in ('lift','transport') and s['open'] is True and s['unsupported'] is True):
            window=[]
            continue
        window.append(s)
        window=window[-3:]
        if len(window)==3:
            a,b=window[0],window[-1]
            grip=[v-u for u,v in zip(a['grip'],b['grip'])]
            plug=[v-u for u,v in zip(a['plug'],b['plug'])]
            if dist(grip,[0,0,0])>.010 and dist(grip,plug)<.001:
                return 'REJECT_AND_ROLLBACK'
    if flags is None or not samples or direct_pose_write is not False:
        return 'DEFER_UNKNOWN'
    return 'NO_TESTED_CONTRADICTION'
