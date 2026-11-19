# Firefly 双臂 cuRobo IK 控制

给定工作台坐标系中的 TCP 位姿，控制器用 NVIDIA cuRobo 在 GPU 上求出六轴关节角，在 MuJoCo 里驱动对应手臂，并读取实际到达的 TCP 位姿。[已知问题](KNOWN_ISSUES.md) · [完整测试报告](audit_results/REPORT.md)

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
./ik audit --trace audit_results/motor_traces.npz > audit_results/results.json
```

以下命令驱动左臂到指定位置和朝向。将 `move` 换为 `solve` 可以只求解并检查末端误差。右臂使用 `--right`，也可以在同一条命令中分别传入两侧的目标。

```bash
./ik move --left 0.4408943 0.1278142 0.6614540 0.6387917 0.2031093 0.7198662 -0.1802342
```

七个参数依次为 `X Y Z QW QX QY QZ`。位置相对 `workcell_base`，单位为米。四元数按 wxyz 归一化。左右 TCP 分别位于各自 `link_6` 局部 +Z 方向 0.170 m 处。输出中的 `joints_rad` 按该侧 J1 至 J6 排列，单位为弧度。

## 控制与验收

每侧求解器从导出的 URDF 读取运动链和标定基座，从 MuJoCo 场景读取 TCP 变换。[硬件参数](hardware.py) 按厂家数据生成 `hardware_model/` 下的 MJCF 和 URDF，控制器的仿真与求解入口都加载这两个文件。cuRobo 使用多初值求解，并在收敛候选中选择最接近当前实测关节角的解。双臂独立求解，两侧目标均通过 MuJoCo 正运动学复核后，在同一个物理实例中同步执行。单臂运动时，另一臂和夹爪保持原控制设定。当前实现执行关节空间位置伺服，未启用 cuRobo 避障规划。

`move` 的控制周期为 0.1 s，物理步长为 2 ms。每一步用整臂速度规格限制位置指令的斜率，用电机峰值力矩限制关节输出。所有请求手臂的位置误差小于 0.4 mm、姿态误差小于 0.004 rad，且连续保持三个控制周期时，返回 `reached`。这一状态只表示末端到位，运动中仍可能发生接触或短时超额定力矩。`arms` 给出最终实测关节角、TCP 位姿和跟踪误差。`candidate_errors` 给出执行前求解结果的几何误差。返回 `ik_failed` 或 `candidate_fk_failed` 时没有执行本次请求。`tracking_timeout` 表示已执行但在规定时间内未稳定到位。

标准输出为 JSON。退出码 0 表示成功，1 表示求解、跟踪或验证未通过，2 表示输入或运行环境错误。`audit` 检出碰撞、漏解或超限等问题时会写出完整报告并返回 1。[电机与双臂测试结果](audit_results/REPORT.md)

更新硬件参数后，RTX 4090 上重新运行的 14 项基线检查全部通过，包括 10 个可达位姿、2 个不可达目标、双臂请求失败时的状态保持和双臂同步运动。最后一项在 0.7 秒仿真时间内稳定到位，左、右 TCP 的位置误差分别为 0.157 mm 和 0.146 mm。运行 `./ik verify > verification.json` 可保存基线结果。

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

