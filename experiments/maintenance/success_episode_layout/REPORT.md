# 成功仿真数据存储整理

按用户更正，成功数据采用与失败数据相同的布局：

```text
complete_sim/succeed/<episode_id>/
├── videos/                 # 本条三路相机MP4
├── proprio_stats/          # 本条HDF5、metadata和已有回放证据
├── outcome.json
└── recording_<episode_id>.json  # 新采集记录；原始Episode保留原文件
```

已把15条分散在共享videos/与proprio_stats/中的录像Episode归并为独立目录；原始Episode保留，共16条。没有录像的10条历史headless试跑移到self_evolution/legacy_success_evidence/<id>/，保留全部中间证据，不算完整录像Episode。两条早期记录没有独立录像outcome，依据各自已有metadata.success生成摘要，并保留原source_evidence_outcome.json。

全部既有文件逐个比较迁移前后SHA-256，内容未改变；48段真实文件视频均完整解码，逐条与metadata及HDF5的sim_time帧数核对。失败目录132个文件的路径和SHA-256完全保持。未重新编码、运行新仿真或进行真机运动。

三个旧Episode原本缺少后期要求的recorded_commands.csv，其中一条还缺少回放目录的initial_state.npz；未伪造补齐，逐条限制见result.json。本次验证存储完整性和帧数，不把整理当成新的物理任务验证。

archive_trial.py把新的headless成功证据放在self_evolution/archived_trials/；finalize_recordings.py将完整成功录像直接安装到succeed/<id>/；diagnose_retention.py读取同一位置。AGENTS.md与README.md已更新，取代旧共享视频根目录规则。13项针对性单元测试通过，包含未来保存位置的回归检查。

历史实验报告保持原文和哈希。其旧dataset_relative_path通过migration.json的files[].old/new映射到当前位置；映射同时保留文件哈希及字节数，可用于审计或反向迁移。大型原始数据仍只保存在本地。
