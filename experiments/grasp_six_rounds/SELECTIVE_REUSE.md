# 本批只采用能定位当前故障的部分

已读[REFLECT的generate_summary/run_reasoning源码](https://github.com/real-stanford/reflect/blob/c2f9db993e372250321ca74e56ae27519716c934/main/exp.py)：先检查子目标后置条件，找到失败子目标才展开事件摘要。本任务使用现有phase_evidence和可信相机时钟，先读抓取/抬升/搬运端点，再看673–677帧窗口；没有引入其RGB-D、声音、AI2THOR或模型调用链。

已读[AWM的memory.py](https://github.com/zorazrw/agent-workflow-memory/blob/8c0ff8cd11d648c8fceb99e4e42f37e3b75381b1/mind2web/memory.py)和[工作流检索](https://github.com/zorazrw/agent-workflow-memory/blob/8c0ff8cd11d648c8fceb99e4e42f37e3b75381b1/mind2web/workflow/retrieve.py)。前者仍积累观察/动作；不能声称照搬它就会消除历史增长。本任务只保留适用条件明确的NE021诊断/恢复步骤，并把已观察的失败接到grasp_handoff实际执行检查，未安装网页代理或向量库。

[SKILL.state论文](https://aclanthology.org/2026.emnlp-industry.264/)支持显式当前执行状态。current_state改为当前版本、阶段、最新证据、不确定性及下一步；旧状态完整归档。只缩减本项目检查点，不声称改变聊天宿主的历史管理。未核实其官方实现，不列代码地址。

[RoboSimStudio工作规范](https://github.com/robosimstudio/robosimstudio/blob/ba3650c4669fffabb7bd52802f4fdf6cdc66cb19/examples/live/AGENTS.md)及[physics实现](https://github.com/robosimstudio/robosimstudio/blob/ba3650c4669fffabb7bd52802f4fdf6cdc66cb19/robosimstudio_real2sim_agent/physics.py)的冻结/张爪反事实思路，已接到已有grasp_contact_probe；仍使用本机MuJoCo3.3.7。三次独立重放分别为正常闭爪、冻结目标、张爪目标，结果见grasp_correction_02/physics_controls.json。没有复制该库代码或迁移环境；没有运行其完整ctl physics套件，也没有把负对照通过称为真实物理校准。

接触检查复用contact_evidence.inspect_state：有效约束、接触距离/力、碰撞体/可视网格距离、执行器饱和和外力；按具体模型、状态和帧报告。参数解释对照[MuJoCo3.3.7](https://mujoco.readthedocs.io/en/3.3.7/modeling.html#contact-parameters)。目前10N仿真执行器与0.3/1Nm真实电机指令的映射仍未辨识，不通过调摩擦掩盖此缺口。

[增量构建思路](https://www.microsoft.com/en-us/research/publication/build-systems-la-carte/)只用于沿用已有内容哈希缓存和未变的冻结前缀；延伸搬运/折回时仅验证受影响参考。MimicGen物体参考变换、AdaptSim、SysID和新机器人组件尚未在本批执行：当前缺口是已有场景的抓取保持，且真实物体6DoF/力测量前提不足。

是否真的加速以实际耗时与少走的无效动作判断。离线筛查拟合了已见案例，尚无独立误报率；拒绝分支用实际控制循环验证未发送transport目标，但在硬件发生拒绝前，不报告已节省真实运动时间。
