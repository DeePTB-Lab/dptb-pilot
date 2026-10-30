* # DeePTB Tutorial 1: deeptb-sk baseline model  [v2.2] 

<div style="color:black; background-color:#FFF3E9; border: 1px solid #FFE0C3; border-radius: 10px; margin-bottom:1rem">
    <p style="margin:1rem; padding-left: 1rem; line-height: 2.5;">
        Author: <a style="font-weight:normal" href="mailto:guqq@ustc.edu.cn">Gu, Qiangqiang 顾强强 📨 </a></b></i><br/>
        Date: 2025-04-20<br/>
        Protocol：<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.</a></i><br/>
        Quick Start：Click the <span style="background-color:rgb(85, 91, 228); color:white; padding: 3px; border-radius: 5px;box-shadow: 2px 2px 3px rgba(0, 0, 0, 0.3); font-size:0.75rem;">Start Connection</span> button，then wait a moment to begin.
    </p>
</div>

## Introduction

**DeePTB** is a method that uses deep learning to accelerate first-principles electronic structure simulations.

### Version Features
- **v1**: Constructed tight-binding (TB) models with first-principles accuracy (DeePTB-SK)
- **v2**: Added E3 equivariant networks to represent single-electron operators (Hamiltonian, density matrix, and overlap matrix) (DeePTB-E3)
- **v2.2**: Incorporated built-in SK empirical parameters covering commonly used elements across the periodic table

Through these capabilities, DeePTB provides multiple approaches to accelerate electronic structure simulations of materials.

### Learning Objectives

In this tutorial, you will:
1. Learn how to use built-in base model to plot band structure for given crystal structure
2. Learn how to generate a empirical sk model in deeptb-sk format for target system

## 1. Calculating Band Structure for a Given Structure

The deeptb-sk module now [since v2.2] has built-in empirical SK parameter models covering elements across the periodic table. 

These can be directly used to obtain empirical SKTB models for given structures. It also supports directly plotting band structures for a given structure.


```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os
workdir='/root/soft/DeePTB/examples/base_model/'
os.chdir(f"{workdir}/structures")
!tree -L 1 ./
```

    [01;34m./[00m
    ├── [01;34mband_gaas[00m
    ├── [01;34mband_hBN[00m
    ├── [01;34mband_si[00m
    ├── gaas.vasp
    ├── hBN.vasp
    └── silicon.vasp
    
    3 directories, 3 files


Run the band structure plotting command.
**Note** that the selection of high-symmetry paths in the Brillouin zone is based on the seekpath.get_path_orig_cell function, which has the following characteristics to be aware of:
1. It does not support 2D materials and will treat 2D materials as 3D materials
   
2. If the input cell is a non-standard primitive unit cell, the returned k path is equivalent to the k path for the standard cell. For example, the band structure calculated along the k path for the standard and non-standard unit cells will be the same up to numerical errors.
   
3. If the input cell is a supercell of a smaller primitive cell, the returned k path is that of the associated primitive cell, in the basis of supercell reciprocal lattice. In this case, the k points are not the high-symmetry points of the first Brillouin zone of the given supercell, but the high-symmetry points of the Brillouin zone of the associated primitive cell.

The command for plotting the band structure is as follows: 


```python
# 1. Silicon
!dptb run band -i poly4 -stu silicon.vasp -o band_si

plt.figure(figsize=(10, 8))
img = mpimg.imread('./band_si/results/band.png')
plt.imshow(img)
plt.axis('off')
plt.show()
```

    TBPLaS is not installed. Thus the TBPLaS is not available, Please install it first.
     
     
    #################################################################################
    #                                                                               #
    #                                                                               #
    #      ██████████                     ███████████  ███████████ ███████████      #
    #     ░░███░░░░███                   ░░███░░░░░███░█░░░███░░░█░░███░░░░░███     #
    #      ░███   ░░███  ██████   ██████  ░███    ░███░   ░███  ░  ░███    ░███     #
    #      ░███    ░███ ███░░███ ███░░███ ░██████████     ░███     ░██████████      #
    #      ░███    ░███░███████ ░███████  ░███░░░░░░      ░███     ░███░░░░░███     #
    #      ░███    ███ ░███░░░  ░███░░░   ░███            ░███     ░███    ░███     #
    #      ██████████  ░░██████ ░░██████  █████           █████    ███████████      #
    #     ░░░░░░░░░░    ░░░░░░   ░░░░░░  ░░░░░           ░░░░░    ░░░░░░░░░░░       #
    #                                                                               #
    #                         Version: 2.0.4.dev93+ea00a42                          #
    #                                                                               #
    #################################################################################
     
     
    /opt/mamba/lib/python3.10/site-packages/seekpath/hpkot/__init__.py:172: DeprecationWarning: dict interface is deprecated. Use attribute interface instead
      conv_lattice = dataset["std_lattice"]
    DEEPTB INFO    The structure space group is: Fd-3m (No. 227)
    DEEPTB INFO    dtype is not provided in the input json, set to the value torch.float32 in model ckpt.
    DEEPTB INFO    device is not provided in the input json, set to the value cpu in model ckpt.
    DEEPTB INFO    overlap is not provided in the input json, set to the value True in model ckpt.
    /opt/mamba/lib/python3.10/site-packages/torch/nested/__init__.py:107: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. (Triggered internally at ../aten/src/ATen/NestedTensorImpl.cpp:178.)
      return torch._nested_tensor_from_tensor_list(ts, dtype, None, device, None)
    DEEPTB INFO    No Fermi energy available, setting to 0.0 eV
    Figure(640x560)
    DEEPTB INFO    band calculation successfully completed.



    
![png](tu1_files/tu1_6_1.png)
    



```python
# 2. GaAs
!dptb run band -i poly4 -stu gaas.vasp -o band_gaas

# display the band plot:
plt.figure(figsize=(10, 8))
img = mpimg.imread('./band_gaas/results/band.png')
plt.imshow(img)
plt.axis('off')
plt.show()
```

    TBPLaS is not installed. Thus the TBPLaS is not available, Please install it first.
     
     
    #################################################################################
    #                                                                               #
    #                                                                               #
    #      ██████████                     ███████████  ███████████ ███████████      #
    #     ░░███░░░░███                   ░░███░░░░░███░█░░░███░░░█░░███░░░░░███     #
    #      ░███   ░░███  ██████   ██████  ░███    ░███░   ░███  ░  ░███    ░███     #
    #      ░███    ░███ ███░░███ ███░░███ ░██████████     ░███     ░██████████      #
    #      ░███    ░███░███████ ░███████  ░███░░░░░░      ░███     ░███░░░░░███     #
    #      ░███    ███ ░███░░░  ░███░░░   ░███            ░███     ░███    ░███     #
    #      ██████████  ░░██████ ░░██████  █████           █████    ███████████      #
    #     ░░░░░░░░░░    ░░░░░░   ░░░░░░  ░░░░░           ░░░░░    ░░░░░░░░░░░       #
    #                                                                               #
    #                         Version: 2.0.4.dev93+ea00a42                          #
    #                                                                               #
    #################################################################################
     
     
    /opt/mamba/lib/python3.10/site-packages/seekpath/hpkot/__init__.py:172: DeprecationWarning: dict interface is deprecated. Use attribute interface instead
      conv_lattice = dataset["std_lattice"]
    DEEPTB INFO    The structure space group is: F-43m (No. 216)
    DEEPTB INFO    dtype is not provided in the input json, set to the value torch.float32 in model ckpt.
    DEEPTB INFO    device is not provided in the input json, set to the value cpu in model ckpt.
    DEEPTB INFO    overlap is not provided in the input json, set to the value True in model ckpt.
    /opt/mamba/lib/python3.10/site-packages/torch/nested/__init__.py:107: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. (Triggered internally at ../aten/src/ATen/NestedTensorImpl.cpp:178.)
      return torch._nested_tensor_from_tensor_list(ts, dtype, None, device, None)
    DEEPTB INFO    No Fermi energy available, setting to 0.0 eV
    Figure(640x560)
    DEEPTB INFO    band calculation successfully completed.



    
![png](tu1_files/tu1_7_1.png)
    



```python
# 3. hBN 2D 
!dptb run band -i poly4 -stu hBN.vasp -o band_hBN

# display the band plot:
plt.figure(figsize=(10, 8))
img = mpimg.imread('./band_hBN/results/band.png')
plt.imshow(img)
plt.axis('off')
plt.show()

```

    TBPLaS is not installed. Thus the TBPLaS is not available, Please install it first.
     
     
    #################################################################################
    #                                                                               #
    #                                                                               #
    #      ██████████                     ███████████  ███████████ ███████████      #
    #     ░░███░░░░███                   ░░███░░░░░███░█░░░███░░░█░░███░░░░░███     #
    #      ░███   ░░███  ██████   ██████  ░███    ░███░   ░███  ░  ░███    ░███     #
    #      ░███    ░███ ███░░███ ███░░███ ░██████████     ░███     ░██████████      #
    #      ░███    ░███░███████ ░███████  ░███░░░░░░      ░███     ░███░░░░░███     #
    #      ░███    ███ ░███░░░  ░███░░░   ░███            ░███     ░███    ░███     #
    #      ██████████  ░░██████ ░░██████  █████           █████    ███████████      #
    #     ░░░░░░░░░░    ░░░░░░   ░░░░░░  ░░░░░           ░░░░░    ░░░░░░░░░░░       #
    #                                                                               #
    #                         Version: 2.0.4.dev93+ea00a42                          #
    #                                                                               #
    #################################################################################
     
     
    /opt/mamba/lib/python3.10/site-packages/seekpath/hpkot/__init__.py:172: DeprecationWarning: dict interface is deprecated. Use attribute interface instead
      conv_lattice = dataset["std_lattice"]
    DEEPTB INFO    The structure space group is: P-6m2 (No. 187)
    DEEPTB INFO    dtype is not provided in the input json, set to the value torch.float32 in model ckpt.
    DEEPTB INFO    device is not provided in the input json, set to the value cpu in model ckpt.
    DEEPTB INFO    overlap is not provided in the input json, set to the value True in model ckpt.
    /opt/mamba/lib/python3.10/site-packages/torch/nested/__init__.py:107: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. (Triggered internally at ../aten/src/ATen/NestedTensorImpl.cpp:178.)
      return torch._nested_tensor_from_tensor_list(ts, dtype, None, device, None)
    DEEPTB INFO    No Fermi energy available, setting to 0.0 eV
    Figure(640x560)
    DEEPTB INFO    band calculation successfully completed.



    
![png](tu1_files/tu1_8_1.png)
    


## 2. Extracting SK Parameter Files for a Given System

Since there is a built-in baseline model covering the periodic table, for the target research system, you can extract the empirical parameter model for your target system from this built-in baseline model.


```python
os.chdir(f"{workdir}/confs")
!tree -L 1 ./
```

    [01;34m./[00m
    ├── [01;34mband_hBN[00m
    ├── gaas.json
    ├── hbn_sp.json
    ├── [01;34mhbn_sp_model[00m
    ├── hbn_spd.json
    ├── [01;34mhbn_spd_model[00m
    └── silicon.json
    
    3 directories, 4 files


For the target system, we first need to define the basis set configuration and save it in a JSON file. Below is the configuration we use for hBN.


```python
!cat hbn_spd.json
```

    {
        "common_options": {
            "basis": {
                "B": ["s","p","d"],
                "N": ["s","p","d"]
            }
        }
    }

Run the following command to extract the empirical model settings and parameters for the target system from the built-in empirical model covering the periodic table, and save them in the sktb.json file.


```python
!dptb esk hbn_spd.json -m poly4 -o hbn_spd_model
```

    TBPLaS is not installed. Thus the TBPLaS is not available, Please install it first.
     
     
    #################################################################################
    #                                                                               #
    #                                                                               #
    #      ██████████                     ███████████  ███████████ ███████████      #
    #     ░░███░░░░███                   ░░███░░░░░███░█░░░███░░░█░░███░░░░░███     #
    #      ░███   ░░███  ██████   ██████  ░███    ░███░   ░███  ░  ░███    ░███     #
    #      ░███    ░███ ███░░███ ███░░███ ░██████████     ░███     ░██████████      #
    #      ░███    ░███░███████ ░███████  ░███░░░░░░      ░███     ░███░░░░░███     #
    #      ░███    ███ ░███░░░  ░███░░░   ░███            ░███     ░███    ░███     #
    #      ██████████  ░░██████ ░░██████  █████           █████    ███████████      #
    #     ░░░░░░░░░░    ░░░░░░   ░░░░░░  ░░░░░           ░░░░░    ░░░░░░░░░░░       #
    #                                                                               #
    #                         Version: 2.0.4.dev93+ea00a42                          #
    #                                                                               #
    #################################################################################
     
     
    DEEPTB INFO    Extracting empirical SK parameters for BN
    DEEPTB INFO    dtype is not provided in the input json, set to the value torch.float32 in model ckpt.
    DEEPTB INFO    device is not provided in the input json, set to the value cpu in model ckpt.
    DEEPTB INFO    overlap is not provided in the input json, set to the value True in model ckpt.
    DEEPTB INFO    Empirical SK parameters are saved in hbn_spd_model/sktb.json
    DEEPTB INFO    If you want to further train the model, please use `dptb config` command to generate input template.


The above command will create an hbn_spd_model folder and save the sktb.json model file in it.


```python
!cat hbn_spd_model/sktb.json
```

    {
        "version": 2,
        "unit": "eV",
        "model_options": {
            "nnsk": {
                "onsite": {
                    "method": "uniform_noref"
                },
                "hopping": {
                    "method": "poly4pow",
                    "rs": {
                        "B-B": 4.22,
                        "B-N": 4.04,
                        "N-B": 4.04,
                        "N-N": 3.85
                    },
                    "w": 0.2
                },
                "soc": {},
                "freeze": false,
                "push": false,
                "std": 0.01,
                "atomic_radius": "cov"
            }
        },
        "common_options": {
            "basis": {
                "B": [
                    "s",
                    "p",
                    "d"
                ],
                "N": [
                    "s",
                    "p",
                    "d"
                ]
            },
            "dtype": "float32",
            "device": "cuda",
            "overlap": true
        },
        "model_params": {
            "onsite": {
                "B-s-0": [
                    -9.436724662780762
                ],
                "B-p-0": [
                    -3.6036229133605957
                ],
                "B-d-0": [
                    0.0
                ],
                "N-s-0": [
                    -18.575620651245117
                ],
                "N-p-0": [
                    -7.092767238616943
                ],
                "N-d-0": [
                    0.0
                ]
            },
            "hopping": {
                "B-B-s-s-0": [
                    -4.163856506347656,
                    3.1430838108062744,
                    3.5406551361083984,
                    -13.840292930603027,
                    4.28679084777832,
                    0.0003413402009755373
                ],
                "B-B-s-p-0": [
                    -4.2632155418396,
                    1.855499505996704,
                    5.719208240509033,
                    -12.824378967285156,
                    2.3826582431793213,
                    0.00027443747967481613
                ],
                "B-B-s-d-0": [
                    0.00014130362251307815,
                    -0.00012631429126486182,
                    0.0001882282958831638,
                    -0.0005035149515606463,
                    0.00022772896045353264,
                    -0.40316706895828247
                ],
                "B-B-p-p-0": [
                    4.029307842254639,
                    -0.4571390151977539,
                    -5.698226451873779,
                    5.24634313583374,
                    1.8383373022079468,
                    0.00098962162155658
                ],
                "B-B-p-p-1": [
                    -1.7277371883392334,
                    1.7380834817886353,
                    0.2498142123222351,
                    -4.095367431640625,
                    1.5244688987731934,
                    0.6229994893074036
                ],
                "B-B-p-d-0": [
                    -3.374056541360915e-05,
                    0.00013169155863579363,
                    -0.00022330175852403045,
                    -8.510611951351166e-05,
                    -6.722987745888531e-05,
                    -0.3973250985145569
                ],
                "B-B-p-d-1": [
                    -7.808039663359523e-05,
                    3.817861215793528e-05,
                    -5.0371396355330944e-05,
                    -0.000102539241197519,
                    -1.8067279597744346e-05,
                    -0.3949469327926636
                ],
                "B-B-d-d-0": [
                    -6.081238097976893e-05,
                    9.681181109044701e-05,
                    -9.043539466802031e-05,
                    5.160560249350965e-05,
                    -0.0002392987225903198,
                    0.3956696689128876
                ],
                "B-B-d-d-1": [
                    0.00015222658112179488,
                    6.237948400666937e-05,
                    0.0002462207048665732,
                    -0.0003655400942079723,
                    0.0004969655419699848,
                    -0.39649420976638794
                ],
                "B-B-d-d-2": [
                    -6.792173371650279e-05,
                    0.00012900785077363253,
                    1.0697433026507497e-05,
                    -0.00010661676060408354,
                    -9.13233234314248e-05,
                    -0.3952234387397766
                ],
                "N-B-s-s-0": [
                    -5.198118686676025,
                    4.442707061767578,
                    4.672472953796387,
                    -22.44223403930664,
                    8.299448013305664,
                    -0.27995532751083374
                ],
                "N-B-s-p-0": [
                    -6.1210198402404785,
                    4.5004191398620605,
                    6.178765296936035,
                    -23.72483253479004,
                    7.701132774353027,
                    -0.17325514554977417
                ],
                "N-B-p-s-0": [
                    -4.536301136016846,
                    2.089078426361084,
                    7.104267597198486,
                    -18.932785034179688,
                    5.050920486450195,
                    -0.08721348643302917
                ],
                "N-B-s-d-0": [
                    -7.312803063541651e-05,
                    0.00010872803977690637,
                    0.00023205213074106723,
                    -0.0005027204751968384,
                    0.0001486523833591491,
                    -0.43798455595970154
                ],
                "N-B-d-s-0": [
                    0.00018420522974338382,
                    -0.00017437210772186518,
                    -0.00016033451538532972,
                    0.0007334152469411492,
                    -8.776571485213935e-05,
                    0.4466051459312439
                ],
                "N-B-p-p-0": [
                    4.69775390625,
                    -0.6529765129089355,
                    -6.943027973175049,
                    8.185653686523438,
                    1.3665649890899658,
                    -0.2770684063434601
                ],
                "N-B-p-p-1": [
                    -1.963960886001587,
                    1.9514763355255127,
                    0.5242166519165039,
                    -5.668179988861084,
                    2.2336957454681396,
                    -0.8325421810150146
                ],
                "N-B-p-d-0": [
                    -6.0136051615700126e-05,
                    8.034883649088442e-05,
                    -0.0002142499142792076,
                    0.00013417207810562104,
                    3.860515425913036e-05,
                    -0.44638267159461975
                ],
                "N-B-d-p-0": [
                    -0.00015385006554424763,
                    9.536759898765013e-05,
                    -0.00010434392606839538,
                    2.4020744604058564e-05,
                    -0.00022013107081875205,
                    0.44291141629219055
                ],
                "N-B-p-d-1": [
                    -4.726650513475761e-05,
                    8.19785927888006e-05,
                    0.00011910804460057989,
                    -0.0003437807899899781,
                    0.00014456789358519018,
                    0.4415556490421295
                ],
                "N-B-d-p-1": [
                    9.281534585170448e-05,
                    -0.000124435915495269,
                    0.00014234631089493632,
                    -0.0006778020178899169,
                    0.00030078168492764235,
                    -0.4489174485206604
                ],
                "N-B-d-d-0": [
                    -0.0001388048694934696,
                    0.00010272378858644515,
                    -0.00019081367645412683,
                    3.532186383381486e-07,
                    -0.00017708863015286624,
                    0.45609885454177856
                ],
                "N-B-d-d-1": [
                    -0.0001369681558571756,
                    0.0001792228576960042,
                    -0.00011874250776600093,
                    -0.00022725776943843812,
                    -5.964709271211177e-05,
                    -0.45647943019866943
                ],
                "N-B-d-d-2": [
                    0.00014195215771906078,
                    -7.780492160236463e-05,
                    8.625858754385263e-05,
                    0.0001418764004483819,
                    3.1707189918961376e-05,
                    -0.45413458347320557
                ],
                "N-N-s-s-0": [
                    -6.06509256362915,
                    4.933748245239258,
                    7.251394271850586,
                    -31.393178939819336,
                    11.564867973327637,
                    0.6752002239227295
                ],
                "N-N-s-p-0": [
                    -6.407330513000488,
                    4.119426727294922,
                    8.804641723632812,
                    -30.208786010742188,
                    9.762639999389648,
                    0.4003536105155945
                ],
                "N-N-s-d-0": [
                    0.00011224352056160569,
                    0.00017669328371994197,
                    2.3582368157804012e-05,
                    -0.0004224574367981404,
                    0.000269725191174075,
                    0.5320302248001099
                ],
                "N-N-p-p-0": [
                    5.50653076171875,
                    -1.4403566122055054,
                    -8.006013870239258,
                    13.777935981750488,
                    -0.6363162994384766,
                    0.36886876821517944
                ],
                "N-N-p-p-1": [
                    -2.2638466358184814,
                    2.1415014266967773,
                    1.0750678777694702,
                    -7.8354573249816895,
                    3.0920794010162354,
                    -1.063719391822815
                ],
                "N-N-p-d-0": [
                    7.20867101335898e-05,
                    0.00036340532824397087,
                    0.00024355569621548057,
                    -0.0004975462798029184,
                    0.0004125885898247361,
                    0.5261558294296265
                ],
                "N-N-p-d-1": [
                    0.00015687875566072762,
                    4.641729174181819e-05,
                    1.3238663086667657e-05,
                    0.00046566742821596563,
                    -0.00020128212054260075,
                    -0.533370852470398
                ],
                "N-N-d-d-0": [
                    8.857469947542995e-05,
                    0.00029952620388939977,
                    0.00024031809880398214,
                    -0.00025129984715022147,
                    0.0003825896419584751,
                    0.5308742523193359
                ],
                "N-N-d-d-1": [
                    3.192495569237508e-05,
                    -0.0001347306970274076,
                    -0.00010394358105259016,
                    -0.00015763661940582097,
                    -0.0002910229086410254,
                    0.527771532535553
                ],
                "N-N-d-d-2": [
                    4.771947715198621e-05,
                    7.819013262633234e-05,
                    0.00018762303807307035,
                    0.00027266336837783456,
                    0.0002987241605296731,
                    -0.5279330611228943
                ]
            },
            "overlap": {
                "B-B-s-s-0": [
                    0.20907151699066162,
                    -0.23900975286960602,
                    -0.06029646843671799,
                    0.805722713470459,
                    -0.34798508882522583,
                    -0.07375287264585495
                ],
                "B-B-s-p-0": [
                    0.25513431429862976,
                    -0.22145505249500275,
                    -0.26228660345077515,
                    1.159956932067871,
                    -0.4085614085197449,
                    -0.0003635674365796149
                ],
                "B-B-s-d-0": [
                    -0.00010135513730347157,
                    0.00012300396338105202,
                    -0.00016775091353338212,
                    1.0182542609982193e-05,
                    -0.0003550578549038619,
                    -0.3522164523601532
                ],
                "B-B-p-p-0": [
                    -0.2968706786632538,
                    0.1348806917667389,
                    0.5835362076759338,
                    -1.4401586055755615,
                    0.34514784812927246,
                    0.00011754724982893094
                ],
                "B-B-p-p-1": [
                    0.0973825603723526,
                    -0.14244243502616882,
                    0.023147646337747574,
                    0.47471076250076294,
                    -0.2704032361507416,
                    -0.667027473449707
                ],
                "B-B-p-d-0": [
                    -0.00011588518100325018,
                    0.00018029639613814652,
                    -9.216874605044723e-05,
                    -0.00048209051601588726,
                    4.212089697830379e-05,
                    0.3581341505050659
                ],
                "B-B-p-d-1": [
                    -0.00011588512279558927,
                    0.00018029661441687495,
                    -9.216976468451321e-05,
                    -0.000482094066683203,
                    4.2125670006498694e-05,
                    0.35813406109809875
                ],
                "B-B-d-d-0": [
                    -0.0001013551518553868,
                    0.0001230038469657302,
                    -0.00016775041876826435,
                    1.018380862660706e-05,
                    -0.0003550600085873157,
                    -0.3522164225578308
                ],
                "B-B-d-d-1": [
                    0.00010135525371879339,
                    -0.0001230034977197647,
                    0.0001677492109593004,
                    -1.018853799905628e-05,
                    0.0003550658584572375,
                    -0.35221636295318604
                ],
                "B-B-d-d-2": [
                    9.815972589422017e-05,
                    -0.00016340948059223592,
                    0.0001705800968920812,
                    0.0002916558878496289,
                    -6.5286149038001895e-06,
                    -0.3533816933631897
                ],
                "N-B-s-s-0": [
                    0.1997109055519104,
                    -0.2244972437620163,
                    -0.11343343555927277,
                    0.9763136506080627,
                    -0.4251488149166107,
                    -0.2723999619483948
                ],
                "N-B-s-p-0": [
                    0.2620687186717987,
                    -0.2704774737358093,
                    -0.23450201749801636,
                    1.4295225143432617,
                    -0.593266487121582,
                    -0.0280486810952425
                ],
                "N-B-p-s-0": [
                    0.2336491048336029,
                    -0.18477198481559753,
                    -0.32471197843551636,
                    1.258323073387146,
                    -0.4374508261680603,
                    -3.666881821118295e-05
                ],
                "N-B-s-d-0": [
                    -2.9759947210550308e-05,
                    9.752172627486289e-05,
                    -0.0001534644834464416,
                    0.00047143836854957044,
                    1.3361132005229592e-05,
                    0.40016892552375793
                ],
                "N-B-d-s-0": [
                    -0.00019644841086119413,
                    0.0001292879751417786,
                    -0.00011083389108534902,
                    -0.00012051903468091041,
                    -0.00022029990213923156,
                    -0.4031250476837158
                ],
                "N-B-p-p-0": [
                    -0.280412495136261,
                    0.11416203528642654,
                    0.6316858530044556,
                    -1.5773518085479736,
                    0.4028981328010559,
                    -0.00018129641830455512
                ],
                "N-B-p-p-1": [
                    0.09812135994434357,
                    -0.14181673526763916,
                    0.005639200564473867,
                    0.5506149530410767,
                    -0.3099479675292969,
                    0.6836565732955933
                ],
                "N-B-p-d-0": [
                    8.620692824479192e-05,
                    7.003633072599769e-05,
                    0.00014002059469930828,
                    1.5960773453116417e-05,
                    0.0003343636344652623,
                    0.39526382088661194
                ],
                "N-B-d-p-0": [
                    -0.00017928905435837805,
                    0.00014779999037273228,
                    -6.557474262081087e-05,
                    0.00033531803637742996,
                    -1.5598576283082366e-05,
                    -0.4068489074707031
                ],
                "N-B-p-d-1": [
                    3.969529643654823e-05,
                    -0.0001192440977320075,
                    0.00016035677981562912,
                    -0.0005162757006473839,
                    -1.6432837583124638e-05,
                    -0.3948792815208435
                ],
                "N-B-d-p-1": [
                    0.0002542896254453808,
                    -4.0774408262223005e-05,
                    0.00016710262570995837,
                    0.0006886226474307477,
                    7.42142292438075e-05,
                    0.4113677740097046
                ],
                "N-B-d-d-0": [
                    -0.0001716944680083543,
                    2.248838427476585e-05,
                    -0.00012059728032909334,
                    0.0001464982924517244,
                    -6.761669646948576e-05,
                    -0.4281955659389496
                ],
                "N-B-d-d-1": [
                    -0.0001716944680083543,
                    2.248838427476585e-05,
                    -0.00012059728032909334,
                    0.0001464982924517244,
                    -6.761669646948576e-05,
                    0.4281955659389496
                ],
                "N-B-d-d-2": [
                    -9.511156531516463e-05,
                    0.0001717804989311844,
                    -1.6953305021161214e-05,
                    0.0001179392565973103,
                    9.918229625327513e-05,
                    0.4204893112182617
                ],
                "N-N-s-s-0": [
                    0.1918010711669922,
                    -0.2127472460269928,
                    -0.1606387048959732,
                    1.155442476272583,
                    -0.5151748657226562,
                    0.48609858751296997
                ],
                "N-N-s-p-0": [
                    0.24681918323040009,
                    -0.23861254751682281,
                    -0.31103551387786865,
                    1.6039279699325562,
                    -0.660601019859314,
                    -0.07287845760583878
                ],
                "N-N-s-d-0": [
                    -6.115203723311424e-05,
                    -0.0002562953741289675,
                    -0.00012758496450260282,
                    0.0004416516749188304,
                    -0.00025469131651334465,
                    0.488688588142395
                ],
                "N-N-p-p-0": [
                    -0.2732810080051422,
                    0.10842154920101166,
                    0.699445903301239,
                    -1.881728172302246,
                    0.5495500564575195,
                    0.00020055289496667683
                ],
                "N-N-p-p-1": [
                    0.10150457173585892,
                    -0.1420295238494873,
                    -0.021784797310829163,
                    0.6614340543746948,
                    -0.36609184741973877,
                    -0.7560092806816101
                ],
                "N-N-p-d-0": [
                    -1.3360753655433655e-05,
                    -0.00014160110731609166,
                    0.00010747101623564959,
                    -0.00040430951048620045,
                    0.0002254340797662735,
                    -0.4823491871356964
                ],
                "N-N-p-d-1": [
                    5.713030986953527e-05,
                    0.00020219954603817314,
                    -1.679777051322162e-05,
                    -0.00015270523726940155,
                    0.0001418725005351007,
                    -0.4920305609703064
                ],
                "N-N-d-d-0": [
                    2.117294025083538e-05,
                    -0.00019494653679430485,
                    9.384578152094036e-05,
                    -0.00038998579839244485,
                    -7.752966484986246e-05,
                    0.487053245306015
                ],
                "N-N-d-d-1": [
                    -3.1081654014997184e-06,
                    -0.00021633837604895234,
                    3.2148207537829876e-05,
                    -3.418952110223472e-05,
                    -0.00027515843976289034,
                    0.4870029091835022
                ],
                "N-N-d-d-2": [
                    -8.452979091089219e-05,
                    -0.00021817802917212248,
                    -5.2256466005928814e-05,
                    -0.0005801816005259752,
                    -6.465519254561514e-05,
                    -0.4823436141014099
                ]
            }
        }
    }

**We can also load the generated sktb.json model file to plot the band structure:**


```python
!dptb run band -i ./hbn_spd_model/sktb.json -stu ../structures/hBN.vasp -o band_hBN

# display the band plot:
plt.figure(figsize=(10, 8))
img = mpimg.imread('./band_hBN/results/band.png')
plt.imshow(img)
plt.axis('off')
plt.show()


```

    TBPLaS is not installed. Thus the TBPLaS is not available, Please install it first.
     
     
    #################################################################################
    #                                                                               #
    #                                                                               #
    #      ██████████                     ███████████  ███████████ ███████████      #
    #     ░░███░░░░███                   ░░███░░░░░███░█░░░███░░░█░░███░░░░░███     #
    #      ░███   ░░███  ██████   ██████  ░███    ░███░   ░███  ░  ░███    ░███     #
    #      ░███    ░███ ███░░███ ███░░███ ░██████████     ░███     ░██████████      #
    #      ░███    ░███░███████ ░███████  ░███░░░░░░      ░███     ░███░░░░░███     #
    #      ░███    ███ ░███░░░  ░███░░░   ░███            ░███     ░███    ░███     #
    #      ██████████  ░░██████ ░░██████  █████           █████    ███████████      #
    #     ░░░░░░░░░░    ░░░░░░   ░░░░░░  ░░░░░           ░░░░░    ░░░░░░░░░░░       #
    #                                                                               #
    #                         Version: 2.0.4.dev93+ea00a42                          #
    #                                                                               #
    #################################################################################
     
     
    /opt/mamba/lib/python3.10/site-packages/seekpath/hpkot/__init__.py:172: DeprecationWarning: dict interface is deprecated. Use attribute interface instead
      conv_lattice = dataset["std_lattice"]
    DEEPTB INFO    The structure space group is: P-6m2 (No. 187)
    /opt/mamba/lib/python3.10/site-packages/torch/nested/__init__.py:107: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. (Triggered internally at ../aten/src/ATen/NestedTensorImpl.cpp:178.)
      return torch._nested_tensor_from_tensor_list(ts, dtype, None, device, None)
    DEEPTB INFO    No Fermi energy available, setting to 0.0 eV
    Figure(640x560)
    DEEPTB INFO    band calculation successfully completed.



    
![png](tu1_files/tu1_18_1.png)
    


We can see that the band structure is the same as before. 

Here we can choose different basis settings, e.g. `hbn_sp.json` as the input config:
```json
{
    "common_options": {
        "basis": {
            "B": ["s","p"],
            "N": ["s","p"]
        }
    }
}
```


```python
!dptb esk hbn_sp.json -m poly4 -o hbn_sp_model
```

    TBPLaS is not installed. Thus the TBPLaS is not available, Please install it first.
     
     
    #################################################################################
    #                                                                               #
    #                                                                               #
    #      ██████████                     ███████████  ███████████ ███████████      #
    #     ░░███░░░░███                   ░░███░░░░░███░█░░░███░░░█░░███░░░░░███     #
    #      ░███   ░░███  ██████   ██████  ░███    ░███░   ░███  ░  ░███    ░███     #
    #      ░███    ░███ ███░░███ ███░░███ ░██████████     ░███     ░██████████      #
    #      ░███    ░███░███████ ░███████  ░███░░░░░░      ░███     ░███░░░░░███     #
    #      ░███    ███ ░███░░░  ░███░░░   ░███            ░███     ░███    ░███     #
    #      ██████████  ░░██████ ░░██████  █████           █████    ███████████      #
    #     ░░░░░░░░░░    ░░░░░░   ░░░░░░  ░░░░░           ░░░░░    ░░░░░░░░░░░       #
    #                                                                               #
    #                         Version: 2.0.4.dev93+ea00a42                          #
    #                                                                               #
    #################################################################################
     
     
    DEEPTB INFO    Extracting empirical SK parameters for BN
    DEEPTB INFO    dtype is not provided in the input json, set to the value torch.float32 in model ckpt.
    DEEPTB INFO    device is not provided in the input json, set to the value cpu in model ckpt.
    DEEPTB INFO    overlap is not provided in the input json, set to the value True in model ckpt.
    DEEPTB INFO    Empirical SK parameters are saved in hbn_sp_model/sktb.json
    DEEPTB INFO    If you want to further train the model, please use `dptb config` command to generate input template.



```python
!dptb run band -i ./hbn_sp_model/sktb.json -stu ../structures/hBN.vasp -o band_hBN

# display the band plot:
plt.figure(figsize=(10, 8))
img = mpimg.imread('./band_hBN/results/band.png')
plt.imshow(img)
plt.axis('off')
plt.show()

```

    TBPLaS is not installed. Thus the TBPLaS is not available, Please install it first.
     
     
    #################################################################################
    #                                                                               #
    #                                                                               #
    #      ██████████                     ███████████  ███████████ ███████████      #
    #     ░░███░░░░███                   ░░███░░░░░███░█░░░███░░░█░░███░░░░░███     #
    #      ░███   ░░███  ██████   ██████  ░███    ░███░   ░███  ░  ░███    ░███     #
    #      ░███    ░███ ███░░███ ███░░███ ░██████████     ░███     ░██████████      #
    #      ░███    ░███░███████ ░███████  ░███░░░░░░      ░███     ░███░░░░░███     #
    #      ░███    ███ ░███░░░  ░███░░░   ░███            ░███     ░███    ░███     #
    #      ██████████  ░░██████ ░░██████  █████           █████    ███████████      #
    #     ░░░░░░░░░░    ░░░░░░   ░░░░░░  ░░░░░           ░░░░░    ░░░░░░░░░░░       #
    #                                                                               #
    #                         Version: 2.0.4.dev93+ea00a42                          #
    #                                                                               #
    #################################################################################
     
     
    /opt/mamba/lib/python3.10/site-packages/seekpath/hpkot/__init__.py:172: DeprecationWarning: dict interface is deprecated. Use attribute interface instead
      conv_lattice = dataset["std_lattice"]
    DEEPTB INFO    The structure space group is: P-6m2 (No. 187)
    /opt/mamba/lib/python3.10/site-packages/torch/nested/__init__.py:107: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. (Triggered internally at ../aten/src/ATen/NestedTensorImpl.cpp:178.)
      return torch._nested_tensor_from_tensor_list(ts, dtype, None, device, None)
    DEEPTB INFO    No Fermi energy available, setting to 0.0 eV
    Figure(640x560)
    DEEPTB INFO    band calculation successfully completed.



    
![png](tu1_files/tu1_21_1.png)
    


It can be clearly seen that the bands near 0 eV are missing. This is because for the hBN system, our built-in empirical model parameters only include sp orbitals. The d orbital parameters are all set to 0 just to maintain a consistent format.

Similarly, we can obtain the corresponding model parameters for individual Si and GaAs systems. Readers are invited to explore this themselves.

<div style="color:black; background-color:#FFF3E9; border: 1px solid #FFE0C3; border-radius: 10px; margin-bottom:1rem">
    <p style="margin:1rem; padding-left: 1rem; line-height: 2.5;">
        Author: <a style="font-weight:normal" href="mailto:guqq@ustc.edu.cn">Gu, Qiangqiang : guqq@ustc.edu.cn</a>
    </p>
    <p style="margin:1rem; padding-left: 1rem; line-height: 2.5;">
        Thank you for reading!
    </p>
</div>
