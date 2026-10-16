# TaOₓ 1T1R 存算一体：首轮仿真结果

## 配置与证据边界
- 数据：scikit-learn 内置 `digits`（8×8 手写数字图像，1347 训练、450 测试）。
- 网络：64→32→10 MLP，数字离线训练；将两层权重映射至两组差分 1T1R 模拟交叉阵列。
- 数字浮点测试准确率：**97.56%**。
- TaOₓ 参数：低电导起点 `9.8 µS`、`35` 状态参考 2023 文献某器件条件；其余 Gmax、Vread、读时间及外围能耗为**未经校准的演示假设**。
- 本模型假设：Gmax=60.0 µS、Vread=0.10 V、tread=10.0 ns、ADC=2.0 pJ/次、DAC=0.2 pJ/次。
- 扫描：器件编程误差 0%/2%/5%/10% 电导跨度（独立高斯扰动），ADC 6/8/10bit及理想无量化，3次重复。
- 本能耗只含交叉阵列电阻耗散 + 指定ADC/DAC模型，**不包含** 1T选通管功耗、驱动时钟、缓冲、线路损耗、控制逻辑、静态漏电、存储编程与供电损耗。
- 这不是经过电路标定的芯片结果，也不能用这些数值声称对 GPU 或 AIDC 的实际节能率。

## 关键结果

| ADC | 器件误差σ | 测试准确率 | 阵列 pJ/次 | ADC pJ/次 | DAC pJ/次 | 局部合计 pJ/次 |
|---|---:|---:|---:|---:|---:|---:|
| 8 | 0% | 97.56% ± 0.00% | 1.73 | 168.00 | 19.20 | 188.93 |
| 8 | 5% | 95.56% ± 1.27% | 1.77 | 168.00 | 19.20 | 188.97 |
| 8 | 10% | 90.74% ± 4.06% | 1.82 | 168.00 | 19.20 | 189.02 |
| ideal | 0% | 97.56% ± 0.00% | 1.73 | 168.00 | 19.20 | 188.93 |

![器件误差与准确率](results/accuracy_vs_variation.png)

![局部模型能耗分解](results/modeled_energy_breakdown.png)

## 数学模型
- 将权重用差分器件编码：`W_ij = scale * (Gplus_ij - Gminus_ij)/(Gmax-Gmin)`。
- 施加 `V_i=Vread*x_i/scale_x`，列电流 `I_j=Σ_i V_i G_ij`，差分读出后还原数字权重尺度。
- 阵列能耗估计：`E_array=tread*Σ_i V_i² Σ_j (Gplus_ij+Gminus_ij)`。
- ADC/DAC 能耗按调用次数乘以假设单次能耗。

## 下一步接入钠电 AIDC
1. 用 CrossSim 替换简化误差模型，校准编程误差、读噪声、互连寄生和真实 ADC 行为。
2. 用 NeuroSim 或 SPICE/电路模型补齐外围面积、功耗、时延，并采用相同工作负载建立数字计算基线。
3. 用钠电池 SOC/放电功率约束驱动任务调度，比较**精度与时延合格的任务数/电池输出能量**。

## 文献与工具
- Abedin M, et al. *Material to system-level benchmarking of CMOS-integrated RRAM with ultra-fast switching for low power on-chip learning*. Scientific Reports 13, 14963 (2023). DOI: https://doi.org/10.1038/s41598-023-42214-x
- Sandia National Laboratories CrossSim: https://cross-sim.sandia.gov/
- NeuroSim: https://github.com/neurosim/NeuroSim
