import os
import json
from typing import Optional, Literal, Dict, Any, TypedDict, Union, List
from pathlib import Path
from dptb_pilot.tools.init import mcp
import ast


class ConfigResult(TypedDict):
    config_path: str

def auto_basis(material):
    basis = {}
    for e in material:
        basis[e] = "1s"
    return basis

def auto_rmax(material):
    rmax = {}
    for e in material:
        rmax[e] = 5.0
    return rmax

def auto_irreps_hidden(basis):
    from dptb.data import OrbitalMapper

    idp = OrbitalMapper(basis=basis)
    irreps_ess = idp.get_irreps_ess()

    import math

    def next_power_of_2(x):
        if x <= 1:
            return 1
        return 2 ** math.ceil(math.log2(x))

    def adjust_coefficients(s):
        # 解析
        parts = s.split('+')
        parsed = []
        for p in parts:
            num_str, rest = p.split('x')
            idx = int(rest[0])
            letter = rest[1]
            num = int(num_str)
            parsed.append((num, idx, letter))

        # 步骤 1 & 2: 乘 4 并向上取 2 的幂
        coeffs = [next_power_of_2(num * 4) for num, _, _ in parsed]

        # 步骤 3 & 4: 保证非递增且最后 ≥ 16
        n = len(coeffs)
        # 先让最后一项至少 16
        if coeffs[-1] < 16:
            coeffs[-1] = 16
        # 从后往前传播
        for i in range(n - 2, -1, -1):
            if coeffs[i] < coeffs[i + 1]:
                coeffs[i] = coeffs[i + 1]

        # 重新组装
        result_parts = [f"{coeffs[i]}x{idx}{letter}" for i, (_, idx, letter) in enumerate(parsed)]
        return '+'.join(result_parts)

    return adjust_coefficients(str(irreps_ess))

@mcp.tool()
def generate_deeptb_e3_training_config(
    material: str = "Si",
    basis: str = "auto",
    rmax: str = "auto",
    device: str = "cuda",
    seed: int = 42,
    overlap_predicting: bool = True,
    irreps_hidden: str = "auto",
    neurons: str = "[64, 64]",
    num_epoch: int = 500,
    batch_size: int = 1,
    start_lr: float = 0.05,
    lr_scheduler_factor: float = 0.8,
    lr_scheduler_patience: int = 50,
    lr_scheduler_min_lr: float = 1e-6,
    train_data_root: str = "/personal/your_data_path",
    train_data_prefix: str = "your_data_prefix",
    validation_on: bool = False,
    validation_data_root: str = "/personal/your_validation_data_path",
    validation_data_prefix: str = "your_validation_data_prefix",
    work_path: str = ".",
    output_file_name: str = "deeptb_config.json"
) -> ConfigResult:
    """
    生成用于 DeePTB-E3 模型训练的配置文件。

    参数:
        config_output_path: 输出的配置 JSON 文件路径。
        material: 原子类型。多原子格式如：['Be','N']
        basis: 原子基组。格式如：{'Si':'2s2p1d'}需要根据训练DFT数据的基组进行设置；如果设为auto则默认使用1s轨道（注：建议手动配置）。
        rmax: 各个原子所使用的截断半径。格式如：{'Si':7.1}应与DFT计算中的截断半径相同；如果设为auto则自动生成（注：建议手动配置）。
        device: 训练使用的机器类型。可以是cpu或cuda。
        seed: 训练使用的种子。
        overlap_predicting: 是否训练带有overlap矩阵的模型。如果为True，则需要训练集也包含overlap数据。
        irreps_hidden: 隐藏等变不可约表示的尺寸，影响模型预测能力的主要参数。如需手动设置，可参考文档；如果设为auto则根据basis自动生成合适的尺寸（注：建议自动配置）
        num_epoch: 训练的epoch数。一般来说500已足够，可自行根据训练集大小调整设置。
        neurons: 预测层的神经元数量。建议为4的倍数。
        batch_size: 一个batch的大小。一般1就可以，增大加速效果不明显。
        start_lr: 训练开始时的学习率。
        lr_scheduler_factor: rop学习率计划器的因子。
        lr_scheduler_patience: rop Lr scheduler patience.
        lr_scheduler_min_lr: rop Lr scheduler minimum lr.
        train_data_root: 训练集的路径。
        train_data_prefix: 训练集中各数据文件夹名前缀。
        validation_on: 是否使用验证集。
        validation_data_root: 验证集的路径。
        validation_data_prefix: 验证集中各数据文件夹名前缀。
        work_path: 配置文件的保存路径。
        output_file_name: 配置文件的保存文件名。

    返回:
        包含配置文件路径的字典。

    抛出:
        AssumptionError: 某些数据输入不合规。
        RuntimeError: 写入配置文件失败。
    """

    config_output_path = os.path.join(work_path, output_file_name)

    '''    assert train_data_root != "/personal/your_data_path", "训练集路径必须输入"
    assert train_data_prefix != "your_data_prefix", "训练集前缀必须输入"
    if validation_on:
        assert validation_data_root != "/personal/your_validation_data_path", "验证集已启用，验证集路径必须输入"
        assert validation_data_prefix != "your_validation_data_prefix", "验证集已启用，验证集前缀必须输入"'''
    assert device in ['cpu', 'cuda'], "机器只能为cpu或cuda"
    assert num_epoch > 1
    assert batch_size >= 1
    assert start_lr < 1
    assert 1 > lr_scheduler_factor > 0
    assert lr_scheduler_patience > 1
    assert lr_scheduler_min_lr < start_lr

    # Parse material
    if isinstance(material, str):
        material = material.strip()
        if material.startswith("["):
            try:
                material = ast.literal_eval(material)
            except:
                pass # logic below will handle it or fail later
        if isinstance(material, str):
             # It's likely a single element string like "Si"
             material = [material]

    if basis == "auto":
        basis = auto_basis(material)
    elif isinstance(basis, str):
        try:
            basis = ast.literal_eval(basis)
        except:
            pass # Keep as string if eval fails? usually basis should be dict

    if rmax == "auto":
        rmax = auto_rmax(material)
    elif isinstance(rmax, str):
        try:
            rmax = ast.literal_eval(rmax)
        except:
             pass

    if irreps_hidden == "auto":
        irreps_hidden = auto_irreps_hidden(basis)

    if isinstance(neurons, str):
        try:
            neurons = ast.literal_eval(neurons)
        except:
            pass

    config = {
        "common_options": {
            "seed": seed,
            "basis": basis,
            "device": device,
            "dtype": "float32",
            "overlap": overlap_predicting
        },
        "model_options": {
            "embedding": {
                "method": "slem",
                "r_max": rmax,
                "irreps_hidden": irreps_hidden,
                "n_layers": 3,
                "n_radial_basis": 18,
                "env_embed_multiplicity": 10,
                "avg_num_neighbors": 51,
                "latent_dim": 64,
                "latent_channels": [32],
                "tp_radial_emb": True,
                "tp_radial_channels": [32],
                "PolynomialCutoff_p": 6,
                "cutoff_type": "polynomial",
                "res_update": True,
                "res_update_ratios": 0.5,
                "res_update_ratios_learnable": False
            },
            "prediction": {
                "method": "e3tb",
                "scales_trainable": False,
                "shifts_trainable": False,
                "neurons": neurons
            }
        },
        "train_options": {
            "num_epoch": num_epoch,
            "batch_size": batch_size,
            "optimizer": {
                "lr": start_lr,
                "type": "Adam"
            },
            "lr_scheduler": {
                "type": "rop",
                "factor": lr_scheduler_factor,
                "patience": lr_scheduler_patience,
                "min_lr": lr_scheduler_min_lr
            },
            "loss_options": {
                "train": {"method": "hamil_abs"}
            },
            "save_freq": 50,
            "validation_freq": 10,
            "display_freq": 1,
            "use_tensorboard": True
        },
        "data_options": {
            "train": {
                "type": "DefaultDataset",
                "root": train_data_root,
                "prefix": train_data_prefix,
                "get_Hamiltonian": True,
                "get_overlap": overlap_predicting
            }
        }
    }

    if validation_on:
        config["train_options"]["loss_options"]["validation"] = {
            "method": "hamil_abs"
        }
        config["data_options"]["validation"] = {
            "type": "DefaultDataset",
            "root": validation_data_root,
            "prefix": validation_data_prefix,
            "get_Hamiltonian": True,
            "get_overlap": overlap_predicting
        }

    try:
        with open(config_output_path, "w") as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        raise RuntimeError(f"写入配置文件失败: {e}")

    return {"config_path": str(Path(config_output_path).absolute())}
