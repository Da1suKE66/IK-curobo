# Firefly 双臂 cuRobo IK 控制

给定工作台坐标系中的 TCP 位姿，控制器用 NVIDIA cuRobo 在 GPU 上求出六轴关节角，在 MuJoCo 里驱动对应手臂，并读取实际到达的 TCP 位姿。[已知问题](KNOWN_ISSUES.md) · [修复与回归验收](audit_results/REGRESSION_FIXES.md)

## 安装

使用 Linux x86_64、Python 3.11、Git 和兼容 CUDA 12.8 的 NVIDIA 驱动。运行需要已有的 `firefly-sim-ik-linux-x86_64.tar.gz` 仿真包，其中包含场景、机器人网格和现场标定。GitHub 仓库提供控制器、测试代码和 Markdown 报告。仿真包和 Python 环境需在本地准备，逐步采样在本地运行测试时生成。[cuRobo v0.9.0 安装说明](https://github.com/NVlabs/curobo/blob/v0.9.0/docs/getting-started/installation.rst)

将仿真包复制到仓库根目录后执行以下命令。项目使用 PyTorch 2.9.1+cu128、cuRobo 0.9.0 和 MuJoCo 3.3.7。

```bash
sha256sum -c <<'CHECKSUM'
755b249186d2dfa9830980e47e9f948e7aea296eb13d31dcc41af9f4df68c0f6  firefly-sim-ik-linux-x86_64.tar.gz
CHECKSUM
tar -xzf firefly-sim-ik-linux-x86_64.tar.gz
python3.11 -m venv .venv
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.9.1'
.venv/bin/python -m pip install -r requirements.txt
./ik verify
```

系统缺少 Python 3.11 时，可用仿真包附带的 `firefly-sim-ik/.runtime/python/bin/python3.11` 创建虚拟环境。

## 运行

在本目录执行。每条命令启动一个处于 home 构型的仿真实例。首次求解需要编译 GPU 内核。

```bash
./ik state   # 查看两臂当前的关节角和 TCP 位姿
./ik demo    # 两臂保持各自朝向，一起沿工作台 X 移动 18 mm、沿 Z 抬高 8 mm
./ik verify  # 运行随包 IK 用例及双臂控制验证
./ik audit --trace audit_results/motor_after.npz > audit_results/audit_after.json
```

以下命令驱动左臂到指定位置和朝向。将 `move` 换为 `solve` 可以只求解并检查末端误差。右臂使用 `--right`，也可以在同一条命令中分别传入两侧的目标。

```bash
./ik move --left 0.4408943 0.1278142 0.6614540 0.6387917 0.2031093 0.7198662 -0.1802342
```

七个参数依次为 `X Y Z QW QX QY QZ`。位置相对 `workcell_base`，单位为米。四元数按 wxyz 归一化。左右 TCP 分别位于各自 `link_6` 局部 +Z 方向 0.170 m 处。输出中的 `joints_rad` 按该侧 J1 至 J6 排列，单位为弧度。

## 控制与验收

每侧使用 cuRobo 的 GPU 多初值求解、Firefly 几何分支以及上一帧关节角。求解器在标定的运动链上用双精度细化候选，并复核活动范围和 TCP。点位运动对双臂候选组合尝试同步、先左、先右和不同运动时长；每条轨迹先用完整 MuJoCo 场景按 2 ms 预演。执行端复查接触、关节行程和速度。第 3、5 连杆的碰撞网格由原 STL 分区生成。[硬件配置](hardware.py) · [点位规划](motion.py)

| 输出轴 | 额定力矩 | 仿真硬上限 | 连续超额定时限 |
| --- | ---: | ---: | ---: |
| 每侧 J1–J4 | 9 N·m | 20 N·m | 2.000 s |
| 每侧 J5–J7（含主动夹爪） | 3 N·m | 5 N·m | 2.000 s |

每台电机独立计时。第 1000 个连续超额定物理步后，下一个物理步按额定值限幅；未限幅需求回到额定范围后才重新开放短时余量。守卫计入重力补偿，计时跨命令延续。规划器还检查目标静止时所需的输出力矩。已知集中载荷可以用 `./ik --payload-kg 0.25 0.25 demo` 指定，顺序为左、右 TCP，单位 kg。

`move` 返回 `reached` 时，所有请求手臂的 TCP 位置误差小于 0.4 mm、姿态误差小于 0.004 rad，且最大臂关节速度小于 0.01 rad/s，连续满足三个 100 ms 周期。预演还检查随后 2.4 s 的保持。返回的 `selected_joints_rad` 为几何解，`command_joints_rad` 为实际伺服终点；贴机械限位时，伺服端预留 10 µrad。几何求解失败返回 `ik_failed`，执行前找不到合规动作返回 `motion_rejected`，`first_failure` 和 `rejected_trials` 记录原因。执行中监测到异常则中断并返回 `motion_interrupted`。

`./ik track path.json` 接收带时间戳的 TCP 路点。JSON 字段为 `times_s`、`targets_by_arm` 和可选的 `hold_s`（默认 0.4 s）。每侧按时间给出 `[[x,y,z],[qw,qx,qy,qz]]`，首帧对应当前实测 TCP，从 0 秒起，末端时间落在物理时钟上。控制器按静止起停边界进行三次插值，在每个物理步加入速度和逆动力学前馈。标准输出为 JSON，退出码 0 表示成功，1 表示验收、求解或跟踪失败，2 表示输入或环境错误。[轨迹实现](trajectory.py) · [复跑命令和逐项指标](audit_results/REGRESSION_FIXES.md)

### 在同一个实例中连续控制

在本目录用 `.venv/bin/python` 运行以下代码，后续目标会使用上一次运动结束后的实测关节角。

```python
from contextlib import closing
from control import Controller
from simulation import Simulation

with closing(Simulation()) as sim:
    controller = Controller(sim)
    for dz in (0.008, -0.008):
        xyz, quat = sim.tcp("left")
        result = controller.run({"left": (xyz + [0, 0, dz], quat)}, move=True)
        print(result["status"])
        if result["status"] != "reached":
            break
```

