#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TaOx 1T1R 差分交叉阵列（行为级）AI 推理示例。

运行：python taox_cim_sim.py
依赖：numpy scipy scikit-learn matplotlib pandas

重要说明：
- 35 个状态和约 9.8 uS 的低端电导参考 Abedin et al., Sci Rep 2023。
- Gmax、读电压、ADC/DAC 能耗和时间等为演示假设，不是该论文器件测量值。
- 此程序不是 NeuroSim/CrossSim，也不声称实现晶体管/互连/电池物理级仿真。
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class Config:
    gmin: float = 9.8e-6  # S: paper example: TaOx off-conductance with 35 states
    gmax: float = 60e-6   # S: hypothetical illustrative upper bound
    n_states: int = 35
    v_read: float = 0.10  # V: hypothetical, read voltage
    t_read: float = 10e-9  # s: hypothetical, NOT paper 300 ps write pulse
    e_adc: float = 2e-12  # J/conversion: hypothetical ADC cost
    e_dac: float = 0.2e-12  # J/conversion: hypothetical DAC cost
    adc_bits: int = 8

class AnalogLayer:
    def __init__(self, weights, bias, activation_scale, config, rng, sigma):
        self.w = np.asarray(weights, dtype=np.float64)
        self.b = np.asarray(bias, dtype=np.float64)
        self.cfg = config
        self.scale = float(activation_scale)
        self.wscale = max(float(np.max(abs(self.w))), 1e-12)
        r = self.w / self.wscale
        span = config.gmax - config.gmin
        gp = config.gmin + np.maximum(r,0) * span
        gm = config.gmin + np.maximum(-r,0) * span
        if config.n_states >= 2:
            step = span/(config.n_states-1)
            gp = config.gmin + np.round((gp-config.gmin)/step)*step
            gm = config.gmin + np.round((gm-config.gmin)/step)*step
        self.gp = np.clip(gp + rng.normal(0,sigma*span,gp.shape),config.gmin,config.gmax)
        self.gm = np.clip(gm + rng.normal(0,sigma*span,gm.shape),config.gmin,config.gmax)
        self.refp = None
        self.refm = None

    def currents(self, activ):
        norm=np.clip(activ / self.scale,0,1)
        voltage=self.cfg.v_read*norm
        return voltage @ self.gp, voltage @ self.gm, norm

    def calibrate(self, train_activ):
        p,m,_=self.currents(train_activ)
        # Per-output column full scale; 15% margin. Fixed after training-set calibration.
        self.refp=np.maximum(p.max(axis=0)*1.15,1e-12)
        self.refm=np.maximum(m.max(axis=0)*1.15,1e-12)

    def _adc(self, i, fullscale, bits):
        if bits is None:
            return i
        hi = np.maximum(fullscale,1e-12)
        return np.round(np.clip(i/hi,0,1)*(2**bits-1))/(2**bits-1)*hi

    def forward(self, activ, bits=None):
        p,m,norm=self.currents(activ)
        if bits is not None:
            p=self._adc(p,self.refp,bits)
            m=self._adc(m,self.refm,bits)
        span=self.cfg.gmax-self.cfg.gmin
        logits=(p-m)*self.wscale*self.scale/(self.cfg.v_read*span)+self.b
        # Linear array joule heating, differential paths; V_i I_i per sample.
        conductance=self.gp.sum(axis=1)+self.gm.sum(axis=1)
        e_array=(self.cfg.v_read**2)*self.cfg.t_read*((norm**2)@conductance)
        n_in,n_out=self.w.shape
        e_adc=np.full(activ.shape[0],2*n_out*self.cfg.e_adc)
        e_dac=np.full(activ.shape[0],n_in*self.cfg.e_dac)
        return logits, np.column_stack([e_array,e_adc,e_dac])


def run_model(x, layers, bits):
    total=None
    a=x
    for idx, layer in enumerate(layers):
        a,e=layer.forward(a,bits)
        total=e if total is None else total+e
        if idx<len(layers)-1:
            a=np.maximum(a,0)
    return np.argmax(a,axis=1),total


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'results')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--trials',type=int,default=3)
    args=ap.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    cfg=Config()
    digits=load_digits()
    x=np.asarray(digits.data,dtype=np.float64)/16
    y=np.asarray(digits.target)
    xtr,xte,ytr,yte=train_test_split(x,y,test_size=0.25,stratify=y,random_state=args.seed)
    mlp=MLPClassifier(hidden_layer_sizes=(32,),activation='relu',solver='adam',
                      alpha=0.001,learning_rate_init=0.001,max_iter=550,random_state=args.seed)
    mlp.fit(xtr,ytr)
    baseline=mlp.score(xte,yte)
    train_hidden=np.maximum(xtr @ mlp.coefs_[0]+mlp.intercepts_[0],0)
    # Fixed from training data only, no test-set leakage.
    scale_hidden=max(float(np.max(train_hidden)),1e-6)
    rows=[]
    for bits in [None,6,8,10]:
        for sigma in [0.0,0.02,0.05,0.10]:
            accuracies=[]
            energies=[]
            for trial in range(args.trials):
                rng=np.random.default_rng(args.seed+1000*trial)
                layers=[AnalogLayer(mlp.coefs_[0],mlp.intercepts_[0],1.0,cfg,rng,sigma),
                        AnalogLayer(mlp.coefs_[1],mlp.intercepts_[1],scale_hidden,cfg,rng,sigma)]
                layers[0].calibrate(xtr)
                layers[1].calibrate(train_hidden)
                pred,energy=run_model(xte,layers,bits)
                accuracies.append(np.mean(pred==yte))
                energies.append(energy.mean(axis=0))
            etot=np.mean(energies,axis=0)
            rows.append({'adc_bits':'ideal' if bits is None else bits,
                         'device_sigma_pct':100*sigma,'accuracy_mean':np.mean(accuracies),
                         'accuracy_std':np.std(accuracies),
                         'array_pJ':etot[0]*1e12,'adc_pJ':etot[1]*1e12,
                         'dac_pJ':etot[2]*1e12,'modeled_total_pJ':sum(etot)*1e12})
    keys=rows[0].keys()
    with (args.out/'results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        wr=csv.DictWriter(f,fieldnames=keys);wr.writeheader();wr.writerows(rows)

    # Plot 1: inference accuracy vs device variation.
    fig,ax=plt.subplots(figsize=(7,4.6))
    for bits in ['ideal',6,8,10]:
        a=[v for v in rows if str(v['adc_bits'])==str(bits)]
        ax.errorbar([r['device_sigma_pct'] for r in a],
                    [100*r['accuracy_mean'] for r in a],
                    yerr=[100*r['accuracy_std'] for r in a],marker='o',capsize=2,
                    label=f'ADC {bits}')
    ax.axhline(100*baseline,linestyle='--',label='Digital FP baseline')
    ax.set(xlabel='Device programming variation (% of conductance span)',ylabel='Test accuracy (%)',
           title='TaOx 1T1R behavioral inference (illustrative)')
    ax.legend();ax.grid(alpha=0.25);fig.tight_layout()
    fig.savefig(args.out/'accuracy_vs_variation.png',dpi=170)
    plt.close(fig)

    # Plot 2: energy proxy composition, only modeled partial accelerator consumption.
    r=next(x for x in rows if x['adc_bits']==8 and x['device_sigma_pct']==0)
    fig,ax=plt.subplots(figsize=(6.5,4.2))
    names=['Array resistive','ADC (assumed)','DAC (assumed)']
    ax.bar(names,[r['array_pJ'],r['adc_pJ'],r['dac_pJ']])
    ax.set(ylabel='Modeled energy per inference (pJ)',title='Modeled partial energy — not AIDC energy')
    ax.tick_params(axis='x',rotation=12);fig.tight_layout()
    fig.savefig(args.out/'modeled_energy_breakdown.png',dpi=170)
    plt.close(fig)

    report=[
    '# TaOₓ 1T1R 存算一体：首轮仿真结果',
    '',
    '## 配置与证据边界',
    f'- 数据：scikit-learn 内置 `digits`（8×8 手写数字图像，{len(xtr)} 训练、{len(xte)} 测试）。',
    '- 网络：64→32→10 MLP，数字离线训练；将两层权重映射至两组差分 1T1R 模拟交叉阵列。',
    f'- 数字浮点测试准确率：**{baseline*100:.2f}%**。',
    f'- TaOₓ 参数：低电导起点 `{cfg.gmin*1e6:.1f} µS`、`{cfg.n_states}` 状态参考 2023 文献某器件条件；其余 Gmax、Vread、读时间及外围能耗为**未经校准的演示假设**。',
    f'- 本模型假设：Gmax={cfg.gmax*1e6:.1f} µS、Vread={cfg.v_read:.2f} V、tread={cfg.t_read*1e9:.1f} ns、ADC={cfg.e_adc*1e12:.1f} pJ/次、DAC={cfg.e_dac*1e12:.1f} pJ/次。',
    '- 扫描：器件编程误差 0%/2%/5%/10% 电导跨度（独立高斯扰动），ADC 6/8/10bit及理想无量化，3次重复。',
    '- 本能耗只含交叉阵列电阻耗散 + 指定ADC/DAC模型，**不包含** 1T选通管功耗、驱动时钟、缓冲、线路损耗、控制逻辑、静态漏电、存储编程与供电损耗。',
    '- 这不是经过电路标定的芯片结果，也不能用这些数值声称对 GPU 或 AIDC 的实际节能率。',
    '',
    '## 关键结果',
    '',
    '| ADC | 器件误差σ | 测试准确率 | 阵列 pJ/次 | ADC pJ/次 | DAC pJ/次 | 局部合计 pJ/次 |',
    '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for b,s in [(8,0),(8,5),(8,10),('ideal',0)]:
        r=next(x for x in rows if x['adc_bits']==b and x['device_sigma_pct']==s)
        report.append(f"| {b} | {s:g}% | {100*r['accuracy_mean']:.2f}% ± {100*r['accuracy_std']:.2f}% | {r['array_pJ']:.2f} | {r['adc_pJ']:.2f} | {r['dac_pJ']:.2f} | {r['modeled_total_pJ']:.2f} |")
    report += [
    '',
    '![器件误差与准确率](results/accuracy_vs_variation.png)',
    '',
    '![局部模型能耗分解](results/modeled_energy_breakdown.png)',
    '',
    '## 数学模型',
    '- 将权重用差分器件编码：`W_ij = scale * (Gplus_ij - Gminus_ij)/(Gmax-Gmin)`。',
    '- 施加 `V_i=Vread*x_i/scale_x`，列电流 `I_j=Σ_i V_i G_ij`，差分读出后还原数字权重尺度。',
    '- 阵列能耗估计：`E_array=tread*Σ_i V_i² Σ_j (Gplus_ij+Gminus_ij)`。',
    '- ADC/DAC 能耗按调用次数乘以假设单次能耗。',
    '',
    '## 下一步接入钠电 AIDC',
    '1. 用 CrossSim 替换简化误差模型，校准编程误差、读噪声、互连寄生和真实 ADC 行为。',
    '2. 用 NeuroSim 或 SPICE/电路模型补齐外围面积、功耗、时延，并采用相同工作负载建立数字计算基线。',
    '3. 用钠电池 SOC/放电功率约束驱动任务调度，比较**精度与时延合格的任务数/电池输出能量**。',
    '',
    '## 文献与工具',
    '- Abedin M, et al. *Material to system-level benchmarking of CMOS-integrated RRAM with ultra-fast switching for low power on-chip learning*. Scientific Reports 13, 14963 (2023). DOI: https://doi.org/10.1038/s41598-023-42214-x',
    '- Sandia National Laboratories CrossSim: https://cross-sim.sandia.gov/',
    '- NeuroSim: https://github.com/neurosim/NeuroSim',
    ]
    (args.out/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(f'Baseline digital FP: {100*baseline:.2f}%')
    for b,s in [(8,0),(8,5),(8,10),('ideal',0)]:
        r=next(x for x in rows if x['adc_bits']==b and x['device_sigma_pct']==s)
        print(f"ADC {b:>5}, sigma={s:>4.0f}%: acc={r['accuracy_mean']*100:.2f}% ± {r['accuracy_std']*100:.2f}% ",
              f"partial_energy={r['modeled_total_pJ']:.2f} pJ/inference")
    print('Saved to',args.out)

if __name__=='__main__':
    main()
