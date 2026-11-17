# 录制时间映射修复

实际 Episode `20261117_real_recipient_pair_000001` 的 external 第1170帧、right_wrist 第1169帧出现约 −0.812977s 的 host Unix/monotonic 偏移跳变。相机 global timestamp 步长仍约33.35ms，其 elapsed 与 host monotonic elapsed 全段差异分别不超过0.793ms、0.738ms。该事件发生在插入停止之后，不能解释此前插入失败。

旧导出将每帧相机 global timestamp 加上该帧的 host monotonic−Unix 差；这把连续的相机时间映射向未来推了0.813s。新函数保留第一对 epoch/monotonic 锚点，之后只使用相机 elapsed；校验相机 elapsed 与独立记录的 host receipt elapsed 在一帧以内，设备时钟重置时不静默拟合。未修改机器人轨迹或对齐时间轴来降低机器人误差。

在相同原始图像上，旧→新转换改变了跳钟后的命令配对：J1–J6 最大差为3.828、9.584、7.605、0.340、5.743、3.694°。跳钟前变化只有13.1µs；跳钟后中位变化812.977ms。重新导出1,639帧/相机，恢复24帧，现有完整 schema/HDF5/视频校验通过。旧版1,615帧曾通过格式校验，但时间对应有错，已保存在 `revisions/timing_v1/`。两个版本、原始记录和任务失败标签均保留。

独立本批只读45帧记录复验：external/right_wrist 的 elapsed 差异上限0.155/0.132ms；相对旧换算变化不足1µs。这验证稳定时钟条件下的兼容性，不能代表长时录制或独立曝光同步精度。初始曝光到接收延迟仍取原始 global epoch 关系，没有新增传感器或真机运动。

当前本地 `export_plug_calibration.py` 已调用新增 `recording_clock.py`，本次重导出实际经过该默认代码路径。原导出源码仅留本地备份，公共记录提供前后哈希和新函数；不上传原仿真工程。失败标签、控制器、安全规则和训练资格均未修改。

采用依据是分离系统墙钟和单调时间，复用现有记录而非安装新同步框架。[Python monotonic 文档](https://docs.python.org/3/library/time.html#time.monotonic)明确其不受系统时钟更新影响；已阅读 [RealSense global timestamp 实现](https://github.com/realsenseai/librealsense/blob/master/src/global_timestamp_reader.cpp)中的转换与连续性处理。SDK [已有未来帧时间戳问题记录](https://github.com/realsenseai/librealsense/issues/13122)，但本机证据只证明导出换算错误，不推定同一 SDK 缺陷，也没有升级 SDK。

第一次离线比较用了10µs的过紧断言，13.1µs的正常前段差异使其失败；数据尚未替换。按已测亚毫秒抖动及30fps分辨率修正检查后通过，原检查记录保留。物理成功检测器从未调整。
