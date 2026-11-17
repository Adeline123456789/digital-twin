# grasp_correction_01

22mm抓取/抬升/折回，未发送插接动作。阶段保持结果：False。Near-upright object at lift, then visible slip/drop at3.90–3.98s into return, final empty gripper.

命令时长40.83s，无模型暂停。两路各1224帧，原记录器导出及验证通过；原生task_approach诊断purpose、完整插接成功false和训练资格false保留，独立阶段结果在real_result.json，不伪装正式插接成功数据。诊断原始证据留在self_evolution，未混入complete_real成功集。

六电机及夹爪下使能独立验证通过。对照只有一次复位后试次，尚不能证明唯一因果或重复稳定性。控制版本及回滚在本目录versions.json或plan_versions.json，原默认入口未修改。
