# 当前状态与改动范围

起始提交53c4177；工作分支research/right-arm-20261116。项目工作树最初干净；本地仿真工程已有用户修改，修改前逐文件备份至DATA/self_evolution/right_arm_adapter_v1/native_backup，未重置任何原修改。原始仿真源码、模型、XML、相机标定和图片不上传。

DATA挂载为/dev/nvme0n1p3、ext4、rw；指定仿真/数据/参考路径均存在且可读写。开始时真实失败4目录、真实成功successed为空、仿真成功16目录、失败12目录。外部参考机械臂数据未回放或混入。

读取了最终报告、active policy、原生任务、后端、录像器、归档器、validator与当前单右臂记事本。已确认：--arm A/B/C/D为实验分组；原任务只执行右臂，但状态合同/采集硬编码双臂；原插座是静态夹具，没有左臂支撑逻辑。该静态假设的真实有效性未验证，不能据此启动真机插接。

现场只读枚举得到两台RealSense；另采集两张新鲜RGB，未启动深度流。画面看到右臂、空左侧安装板、插座及插头支架；不能据此推出毫米级坐标、摩擦或真实夹具固定程度。设备序列号/图片仅保存在本地，公开只记相机通道及哈希。未向CAN发送动作，也未启用后台硬件服务。

本批实际调用collect_current_scene.py→benchmark_collection.run_case→learn.simulate→collect_inline→原生try_plug_pick_sim，再经过finalize_recordings/episode_format。硬件合同Command仍要求14值，未改硬件安全控制；新的紧凑7维合同仅存在于仿真后端。仿真接口拒绝14维双臂指令送入右臂模型；本入口不提供硬件动作调用。外部生产Agent与真实采集入口没有自动接管。

原compiled_alignment_v1冻结的是旧原生源码哈希，不能自动迁移到新结构；保留版本和证据，active planner暂回llm。Agent分组入口在PIPER_ACTIVE_ARMS=right时明确指向新的确定性采集入口，不把旧双臂记忆伪装成右臂已验证技能。

## 数据兼容

- 双臂仍是connector_mixed.v1，原左右字段和三视频。
- 右臂是connector_mixed.v2，保持相同HDF5右臂字段、单位、时序与目录组织，仅声明active_arms=[right]和两相机，完全没有伪造左臂状态/动作/左腕视频。
- 同一个episode_format.validate读取v1/v2；正式数据仍一条一目录。
- piper_sim.dataset与lerobot_export处理的是另一种piper.transitions.v1格式，原本也不读取connector_mixed；本批不声称新格式自动适配外部训练消费端。
