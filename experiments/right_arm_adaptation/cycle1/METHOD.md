# 有界查重与复用：右臂场景适配

本批依据任务文档第十节，只适配现有插座任务的机器人结构与采集通道。六周期是上限，本次一个工程周期、三次完整物理执行；不把教学/未知任务研究重新铺开。academic-search MCP不在本会话可用工具中；使用官方arXiv正文、官方文档和GitHub源码。技能preflight：CrossRef、arXiv可达，PubMed超时；不影响本次机器人来源。

| 研究/实现 | 原问题、实际阅读 | 可复用机制与本次决定 | 边界、版本/许可证 |
|---|---|---|---|
| MuJoCo model editing | [官方文档](https://mujoco.readthedocs.io/en/stable/programming/modeledit.html)、[3.3.7 specs_test.py/test_delete](https://github.com/google-deepmind/mujoco/blob/3.3.7/python/mujoco/specs_test.py) | 用已安装MjSpec删除left/base_link，自动移除引用执行器/关节；保留现实仍在的空安装底板；复用原资产 | 本机3.3.7，tag commit f1d45bd5422c74beddfb0d1deb590a02583d21de，Apache-2.0；使用API，未复制上游代码。必须实际检查导出/重新加载，单次compile不等于记录器兼容 |
| dm_control/PyMJCF | [论文2006.12983v2第3节](https://arxiv.org/html/2006.12983)、named indexing与动态肢体组合示例 | 复用“结构和名称驱动映射”的方法；本机已有MjSpec，故未增加PyMJCF依赖 | 论文不是本机插接方法；本次未采用该项目代码，不声称完整复现 |
| robosuite | [官方Robots模块](https://robosuite.ai/docs/modules/robots.html) | 对照单/双臂模型、控制器和部件一致性要求；不替换现有后端 | 只作接口设计比较，未移植代码或基准结果 |
| CoEnv/CoEx | [v2方法3.1–3.4](https://arxiv.org/html/2604.05484v2)、[code_agent_workflow.py](https://github.com/MARS-EAI/CoEnv/blob/5a232525f5b1845ffe038ac96573fec811def2da/coenv/code_agent_workflow.py)中run、_vlm_review、checkpoint/执行反馈 | 仿真特权状态与执行证据已有等价实现，保留本项目独立物理成功检测；不引入其完整代码生成循环 | 官方main核对5a232525f5b1845ffe038ac96573fec811def2da，MIT。实现依赖Gymnasium任务、工具执行器和Anthropic；论文场景构建需要校准RGB-D，不能当成本机RGB-only重建接口。源码以review.success判定循环完成，不能代替本机物理检测 |
| RHD | [2609.33378v1方法3.1–3.4](https://arxiv.org/html/2609.33378v1) | 接收方执行反馈用于修订外部指导；本次将序列化失败边界写入可复用经验 | 本批没有教师/学生对照、没有VLA干预，不能据此宣称模型迁移收益；未采用代码，未核验新发布实现 |

## 重合判断

| 能力 | 现有重合 / 本批处理 |
|---|---|
| 真实反馈修twin | 仅当前RGB确认左臂缺席与安装板仍在，属于结构/场景配置更新；不辨识摩擦和外参 |
| 仿真程序迁移 | 原任务和原生IK继续执行，关节范围由模型名称查询；没有复制第二套任务程序 |
| 采集与恢复 | 保留完整记录器与validator，新增明确v2右臂通道；保存录制故障，复用已完成双臂录像恢复归档 |
| 自进化 | 实际序列化失败→修复→独立再次完整执行；是否接受由结果决定 |
| 强教弱 | RHD有直接方法重合；本批场景适配不测该能力 |
| 无教师未知探索 | 本批执行的是既有任务，不构成未知任务测试 |
| 总成本 | 测试时间/采集时间可测；本轮重叠运行、故障恢复与研究额度不能遗漏，不能把少一相机当作已证明总成本下降 |

D-World、TinyAstral、VISTA、AdaptSim、SysID、ASID、Real-is-Sim、DataFarm、Recova、EmbodiRSI、F4R、RAX及ASPIRE属于任务总文档后续检索候选。本次第十节限定的结构适配已由现有MuJoCo实现覆盖，因此没有逐一重新全文查证，不把用户提供的简介当作本次已验证事实，也不作整个领域不存在重合的论断。

实用价值按本机完整执行与数据完整性判断；移除机器人子树、名称映射、可配置通道均为既有机制，不主张论文创新。
