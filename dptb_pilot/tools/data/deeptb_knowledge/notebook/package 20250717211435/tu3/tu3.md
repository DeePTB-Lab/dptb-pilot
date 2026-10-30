# DeePTB Tutorial 3: Training  deeptb-sk model for Silicon  [v2.2]

<div style="color:black; background-color:#FFF3E9; border: 1px solid #FFE0C3; border-radius: 10px; margin-bottom:1rem">
    <p style="margin:1rem; padding-left: 1rem; line-height: 2.5;">
        Author: <a style="font-weight:normal" href="mailto:guqq@ustc.edu.cn">Gu, Qiangqiang 顾强强 📨 </a></b></i><br/>
        Date: 2025-04-04<br/>
        Protocol：<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.</a></i><br/>
        Quick Start：Click the <span style="background-color:rgb(85, 91, 228); color:white; padding: 3px; border-radius: 5px;box-shadow: 2px 2px 3px rgba(0, 0, 0, 0.3); font-size:0.75rem;">Start Connection</span> button，then wait a moment to begin.
    </p>
</div>

## Introduction
**DeePTB** is a method that uses deep learning to accelerate first-principles electronic structure simulations.

### Version Features
- **v1**: Constructed tight-binding (TB) models with first-principles accuracy (DeePTB-SK)
- **v2.0-2.1**: Added E3 equivariant networks to represent single-electron operators (Hamiltonian, density matrix, and overlap matrix) (DeePTB-E3)
- **v2.2**: Incorporated built-in SK empirical parameters covering commonly used elements across the periodic table

Through these capabilities, DeePTB provides multiple approaches to accelerate electronic structure simulations of materials.

### Learning Objectives

This tutorial mainly introduces the basic operations of constructing TB models using the DeePTB-SK module.

Reading this tutorial will help yo
u:
1. Familiarize yourself with the training process of DeePTB models
2. Obtain a complete DeePTB model for silicon crystal with high accuracy
3. Familiarize yourself with the usage of DeePTB property calculation module

## Method Practice <a id ='practice'></a>


```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os
os.chdir("/root/soft/DeePTB/examples/silicon/tutorial_v2.2")
```

### **1. data preparation** <a id ='dataprepare'></a>
The data used to train the model and plot the verification data is in the `data` folder:

```bash
deeptb/examples/silicon/data/
|-- kpath.0                 # train data of primary cell. (k-path bands)
|-- kpathmd25.0             # train data of 10 MD snapshots at T=25K   (k-path bands)
|-- kpathmd100.0            # train data of 10 MD snapshots at T=100K  (k-path bands)
|-- kpathmd300.0            # train data of 10 MD snapshots at T=300K  (k-path bands)
|-- kpt.0                   # kmesh samples of primary cell  (k-mesh bands)
|-- kpath_spk.0
|-- silicon.vasp            # structure of primary cell
``` 
The meaning of the datasets in this folder is as follows:
- `kpath.0`: Band data of the primitive cell
- `kpathmd25.0`: Band data of 10 MD snapshots at 25K
- `kpathmd100.0`: Band data of 10 MD snapshots at 100K
- `kpathmd300.0`: Band data of 10 MD snapshots at 300K
- `kpt.0`: K-point mesh sampling data of the primitive cell
- `silicon.vasp`: Structure data of the primitive cell
- `kpath_spk.0`: Band data of the primitive cell, spare k points.

Each dataset contains **DeePTB** data files, such as `kpath.0`:
```bash
deeptb/examples/silicon/data/kpath.0/
-- info.json # defining the training objective and edge cutoff of atomic data
-- eigenvalues.npy # numpy array of shape [num_frame, num_kpoint, num_band]
-- kpoints.npy # numpy array of shape [num_kpoint, 3]
-- xdat.traj # ase trajectory file with num_frame
```
Where:
- `info.json`: The filename of this file is fixed and provides information about the dataset loaded in the DeePTB model.
```json
{
    "nframes": 1,
    "natoms": 2,
    "pos_type": "ase",
    "pbc": true,
    "bandinfo": {
        "band_min": 0,
        "band_max": 6,
        "emin": null,
        "emax": null
    }
}
```
> `nframes` marks the number of trajectory snapshots, `natoms` marks the number of atoms in each snapshot, `pos_type` marks the coordinate type, and `pbc` marks whether periodic boundary conditions are applied. The `bandinfo` contains information about the band window, which can be set according to the needs of the user. The band window information can be sorted by band index or divided according to energy size. Note that the value of emin is relative to min(eig[band_min]). Taking min(eig[band_min]) as 0 point.
- `eigenvalues.npy`: This file has a fixed name and contains the original band data, with shape `[n_frames, nkpoints, nbands]`
- `kpoints.npy`: This file has a fixed name and contains the original k-point data, with shape `[nkpoints, 3]`
- `xdat.traj`: This file can have any prefix, but must have the fixed suffix ".traj", and contains trajectory structure data that can be read using ase.
> In addition to providing ase trajectory data with the `.traj` suffix, you can also choose to provide three text files: `positions.dat`, `cell.dat`, and `atomic_numbers.dat` to load the trajectory. The coordinate type provided by the user is specified in `info.json`: it can be fractional coordinates `frac`, actual coordinates `cart`, or ase trajectory file `ase`.




### **2. Model Training ** <a id ='train'></a>
#### **2.1 Extract Initial Experience sktb Model**
Extract the initial experience parameters from the built-in experience parameters. Here is the initial experience parameter model for Si. For details, please refer to the tutorial-1.

Prepare an input file `sk_in.json` for parameter extraction, as follows:
```json
{
    "common_options": {
        "basis": {
            "Si": ["3s","3p","d*"]
        }
    }
}
```
note: the basis can also be ['s','p','d'].

First, run the command to generate an initial sk model:
```bash
dptb esk sk_in.json -m poly4
```
After running, you can see a sktb.json model file.


```python
!dptb esk sk_in.json -m poly4 
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
     
     
    DEEPTB INFO    Extracting empirical SK parameters for Si
    DEEPTB INFO    dtype is not provided in the input json, set to the value torch.float32 in model ckpt.
    DEEPTB INFO    device is not provided in the input json, set to the value cpu in model ckpt.
    DEEPTB INFO    overlap is not provided in the input json, set to the value True in model ckpt.
    DEEPTB INFO    Empirical SK parameters are saved in ./sktb.json
    DEEPTB INFO    If you want to further train the model, please use `dptb config` command to generate input template.


We can compare the band structure of the initial model with DFT results. 


```python
!dptb run band.json -i sktb.json -o  band -stu ../data/silicon.vasp
# display the band plot:
image_path = f'./band/results/band.png'
img = mpimg.imread(image_path)
plt.figure(figsize=(10, 8))
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
     
     
    /opt/mamba/lib/python3.10/site-packages/torch/nested/__init__.py:107: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. (Triggered internally at ../aten/src/ATen/NestedTensorImpl.cpp:178.)
      return torch._nested_tensor_from_tensor_list(ts, dtype, None, device, None)
    DEEPTB INFO    KPOINTS  klist: 302 kpoints
    DEEPTB INFO    The eigenvalues are already in data. will use them.
    DEEPTB INFO    Calculating Fermi energy in the case of spin-degeneracy.
    DEEPTB INFO    Fermi energy converged after 25 iterations.
    DEEPTB INFO    q_cal: 8.00000000014317, total_electrons: 8.0, diff q: 1.4317080854198139e-10
    DEEPTB INFO    Estimated E_fermi: -3.666769308434781 based on the valence electrons setting nel_atom : {'Si': 4} .
    DEEPTB INFO    Using input Fermi energy: -4.7220 eV (estimated: -3.6668 eV)
    Figure(640x560)
    DEEPTB INFO    band calculation successfully completed.



    
![png](tu3_files/tu3_9_1.png)
    


#### **2.2 Generate Training Input Control Parameters**
The developers have provided a template for generating training sk models to facilitate user use. The command to obtain the template is as follows:
```bash
dptb config ./ -tr -sk -m ./sktb.json 
```
**Note**: Here I loaded the sktb.json model file generated in the previous step, so some parameters will be set according to the model.
After running the above command, you will get `./input_templete.json`.

Note: the template cannot be used directly and needs to be modified according to the situation. For example, the parameters in `train_options` and `data_options`. 
You should also ensure that the paths and options are correctly set for your specific use case.


```python
!dptb config ./ -tr -sk -m ./sktb.json 
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
     
     
    DEEPTB INFO    Writing full config for train_SK to ./input_templete.json


Donot forget to modify the `data_options` and `train_options` parameters in the input file. 

We suggest copying the template and modifying the copied input parameter file. For example, we have already prepared the first training parameter file `input_1.json` in the case folder, which can be used for training the model for perfect crystal Si.

#### **2.3.1 Training the DeePTB-SK model for perfect lattice**


```python
# v100 1m45s
!dptb train input_1.json -i sktb.json -o nnsk1
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
     
     
    DEEPTB INFO    ------------------------------------------------------------------
    DEEPTB INFO         Cutoff options:                                            
    DEEPTB INFO                                                                    
    DEEPTB INFO         r_max            : {'Si-Si': 6.24}                         
    DEEPTB INFO         er_max           : None                                    
    DEEPTB INFO         oer_max          : None                                    
    DEEPTB INFO    ------------------------------------------------------------------
    /opt/mamba/lib/python3.10/site-packages/torch/nested/__init__.py:107: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. (Triggered internally at ../aten/src/ATen/NestedTensorImpl.cpp:178.)
      return torch._nested_tensor_from_tensor_list(ts, dtype, None, device, None)
    DEEPTB WARNING The cutoffs in data and model are not checked. be careful!
    DEEPTB INFO    The ['overlap_param'] are frozen!
    DEEPTB INFO    The ['overlap_param'] are frozen!
    DEEPTB INFO    The ['overlap_param'] are frozen!
    DEEPTB INFO    iteration:1	train_loss: 1.713108  (0.513932)	lr: 0.01
    DEEPTB INFO    checkpoint saved as nnsk.iter1
    DEEPTB INFO    Epoch 1 summary:	train_loss: 1.713108	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep1
    DEEPTB INFO    iteration:2	train_loss: 0.789610  (0.596636)	lr: 0.00999
    DEEPTB INFO    checkpoint saved as nnsk.iter2
    DEEPTB INFO    Epoch 2 summary:	train_loss: 0.789610	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep2
    DEEPTB INFO    iteration:3	train_loss: 0.344749  (0.521070)	lr: 0.00998
    DEEPTB INFO    checkpoint saved as nnsk.iter3
    DEEPTB INFO    Epoch 3 summary:	train_loss: 0.344749	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep3
    DEEPTB INFO    iteration:4	train_loss: 0.299568  (0.454619)	lr: 0.00997
    DEEPTB INFO    checkpoint saved as nnsk.iter4
    DEEPTB INFO    Epoch 4 summary:	train_loss: 0.299568	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep4
    DEEPTB INFO    iteration:5	train_loss: 0.454800  (0.454673)	lr: 0.00996
    DEEPTB INFO    checkpoint saved as nnsk.iter5
    DEEPTB INFO    Epoch 5 summary:	train_loss: 0.454800	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    iteration:6	train_loss: 0.573379  (0.490285)	lr: 0.00995
    DEEPTB INFO    checkpoint saved as nnsk.iter6
    DEEPTB INFO    Epoch 6 summary:	train_loss: 0.573379	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    iteration:7	train_loss: 0.560844  (0.511453)	lr: 0.00994
    DEEPTB INFO    checkpoint saved as nnsk.iter7
    DEEPTB INFO    Epoch 7 summary:	train_loss: 0.560844	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    iteration:8	train_loss: 0.454261  (0.494295)	lr: 0.00993
    DEEPTB INFO    checkpoint saved as nnsk.iter8
    DEEPTB INFO    Epoch 8 summary:	train_loss: 0.454261	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    iteration:9	train_loss: 0.327401  (0.444227)	lr: 0.00992
    DEEPTB INFO    checkpoint saved as nnsk.iter9
    DEEPTB INFO    Epoch 9 summary:	train_loss: 0.327401	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    iteration:10	train_loss: 0.235823  (0.381706)	lr: 0.00991
    DEEPTB INFO    checkpoint saved as nnsk.iter10
    DEEPTB INFO    Epoch 10 summary:	train_loss: 0.235823	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep10
    DEEPTB INFO    iteration:11	train_loss: 0.203891  (0.328361)	lr: 0.0099 
    DEEPTB INFO    checkpoint saved as nnsk.iter11
    DEEPTB INFO    Epoch 11 summary:	train_loss: 0.203891	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep11
    DEEPTB INFO    iteration:12	train_loss: 0.223001  (0.296753)	lr: 0.009891
    DEEPTB INFO    checkpoint saved as nnsk.iter12
    DEEPTB INFO    Epoch 12 summary:	train_loss: 0.223001	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    iteration:13	train_loss: 0.262046  (0.286341)	lr: 0.009881
    DEEPTB INFO    checkpoint saved as nnsk.iter13
    DEEPTB INFO    Epoch 13 summary:	train_loss: 0.262046	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    iteration:14	train_loss: 0.288180  (0.286893)	lr: 0.009871
    DEEPTB INFO    checkpoint saved as nnsk.iter14
    DEEPTB INFO    Epoch 14 summary:	train_loss: 0.288180	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    iteration:15	train_loss: 0.282944  (0.285708)	lr: 0.009861
    DEEPTB INFO    checkpoint saved as nnsk.iter15
    DEEPTB INFO    Epoch 15 summary:	train_loss: 0.282944	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    iteration:16	train_loss: 0.247870  (0.274356)	lr: 0.009851
    DEEPTB INFO    checkpoint saved as nnsk.iter16
    DEEPTB INFO    Epoch 16 summary:	train_loss: 0.247870	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    iteration:17	train_loss: 0.196793  (0.251087)	lr: 0.009841
    DEEPTB INFO    checkpoint saved as nnsk.iter17
    DEEPTB INFO    Epoch 17 summary:	train_loss: 0.196793	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep17
    DEEPTB INFO    iteration:18	train_loss: 0.149260  (0.220539)	lr: 0.009831
    DEEPTB INFO    checkpoint saved as nnsk.iter18
    DEEPTB INFO    Epoch 18 summary:	train_loss: 0.149260	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep18
    DEEPTB INFO    iteration:19	train_loss: 0.119433  (0.190207)	lr: 0.009822
    DEEPTB INFO    checkpoint saved as nnsk.iter19
    DEEPTB INFO    Epoch 19 summary:	train_loss: 0.119433	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep19
    DEEPTB INFO    iteration:20	train_loss: 0.112715  (0.166960)	lr: 0.009812
    DEEPTB INFO    checkpoint saved as nnsk.iter20
    DEEPTB INFO    Epoch 20 summary:	train_loss: 0.112715	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep20
    DEEPTB INFO    iteration:21	train_loss: 0.123057  (0.153789)	lr: 0.009802
    DEEPTB INFO    checkpoint saved as nnsk.iter21
    DEEPTB INFO    Epoch 21 summary:	train_loss: 0.123057	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    iteration:22	train_loss: 0.138459  (0.149190)	lr: 0.009792
    DEEPTB INFO    checkpoint saved as nnsk.iter22
    DEEPTB INFO    Epoch 22 summary:	train_loss: 0.138459	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    iteration:23	train_loss: 0.147751  (0.148758)	lr: 0.009782
    DEEPTB INFO    checkpoint saved as nnsk.iter23
    DEEPTB INFO    Epoch 23 summary:	train_loss: 0.147751	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    iteration:24	train_loss: 0.145198  (0.147690)	lr: 0.009773
    DEEPTB INFO    checkpoint saved as nnsk.iter24
    DEEPTB INFO    Epoch 24 summary:	train_loss: 0.145198	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    iteration:25	train_loss: 0.131092  (0.142711)	lr: 0.009763
    DEEPTB INFO    checkpoint saved as nnsk.iter25
    DEEPTB INFO    Epoch 25 summary:	train_loss: 0.131092	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    iteration:26	train_loss: 0.111268  (0.133278)	lr: 0.009753
    DEEPTB INFO    checkpoint saved as nnsk.iter26
    DEEPTB INFO    Epoch 26 summary:	train_loss: 0.111268	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep26
    DEEPTB INFO    iteration:27	train_loss: 0.092944  (0.121178)	lr: 0.009743
    DEEPTB INFO    checkpoint saved as nnsk.iter27
    DEEPTB INFO    Epoch 27 summary:	train_loss: 0.092944	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep27
    DEEPTB INFO    iteration:28	train_loss: 0.082383  (0.109539)	lr: 0.009733
    DEEPTB INFO    checkpoint saved as nnsk.iter28
    DEEPTB INFO    Epoch 28 summary:	train_loss: 0.082383	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep28
    DEEPTB INFO    iteration:29	train_loss: 0.080075  (0.100700)	lr: 0.009724
    DEEPTB INFO    checkpoint saved as nnsk.iter29
    DEEPTB INFO    Epoch 29 summary:	train_loss: 0.080075	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep29
    DEEPTB INFO    iteration:30	train_loss: 0.082685  (0.095295)	lr: 0.009714
    DEEPTB INFO    checkpoint saved as nnsk.iter30
    DEEPTB INFO    Epoch 30 summary:	train_loss: 0.082685	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    iteration:31	train_loss: 0.085411  (0.092330)	lr: 0.009704
    DEEPTB INFO    checkpoint saved as nnsk.iter31
    DEEPTB INFO    Epoch 31 summary:	train_loss: 0.085411	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    iteration:32	train_loss: 0.084484  (0.089976)	lr: 0.009695
    DEEPTB INFO    checkpoint saved as nnsk.iter32
    DEEPTB INFO    Epoch 32 summary:	train_loss: 0.084484	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    iteration:33	train_loss: 0.078903  (0.086654)	lr: 0.009685
    DEEPTB INFO    checkpoint saved as nnsk.iter33
    DEEPTB INFO    Epoch 33 summary:	train_loss: 0.078903	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep33
    DEEPTB INFO    iteration:34	train_loss: 0.070559  (0.081826)	lr: 0.009675
    DEEPTB INFO    checkpoint saved as nnsk.iter34
    DEEPTB INFO    Epoch 34 summary:	train_loss: 0.070559	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep34
    DEEPTB INFO    iteration:35	train_loss: 0.062745  (0.076101)	lr: 0.009666
    DEEPTB INFO    checkpoint saved as nnsk.iter35
    DEEPTB INFO    Epoch 35 summary:	train_loss: 0.062745	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep35
    DEEPTB INFO    iteration:36	train_loss: 0.058192  (0.070729)	lr: 0.009656
    DEEPTB INFO    checkpoint saved as nnsk.iter36
    DEEPTB INFO    Epoch 36 summary:	train_loss: 0.058192	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep36
    DEEPTB INFO    iteration:37	train_loss: 0.057665  (0.066810)	lr: 0.009646
    DEEPTB INFO    checkpoint saved as nnsk.iter37
    DEEPTB INFO    Epoch 37 summary:	train_loss: 0.057665	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep37
    DEEPTB INFO    iteration:38	train_loss: 0.059696  (0.064675)	lr: 0.009637
    DEEPTB INFO    checkpoint saved as nnsk.iter38
    DEEPTB INFO    Epoch 38 summary:	train_loss: 0.059696	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    iteration:39	train_loss: 0.061716  (0.063787)	lr: 0.009627
    DEEPTB INFO    checkpoint saved as nnsk.iter39
    DEEPTB INFO    Epoch 39 summary:	train_loss: 0.061716	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    iteration:40	train_loss: 0.061624  (0.063138)	lr: 0.009617
    DEEPTB INFO    checkpoint saved as nnsk.iter40
    DEEPTB INFO    Epoch 40 summary:	train_loss: 0.061624	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    iteration:41	train_loss: 0.058877  (0.061860)	lr: 0.009608
    DEEPTB INFO    checkpoint saved as nnsk.iter41
    DEEPTB INFO    Epoch 41 summary:	train_loss: 0.058877	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    iteration:42	train_loss: 0.054544  (0.059665)	lr: 0.009598
    DEEPTB INFO    checkpoint saved as nnsk.iter42
    DEEPTB INFO    Epoch 42 summary:	train_loss: 0.054544	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep42
    DEEPTB INFO    iteration:43	train_loss: 0.050468  (0.056906)	lr: 0.009588
    DEEPTB INFO    checkpoint saved as nnsk.iter43
    DEEPTB INFO    Epoch 43 summary:	train_loss: 0.050468	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep43
    DEEPTB INFO    iteration:44	train_loss: 0.048096  (0.054263)	lr: 0.009579
    DEEPTB INFO    checkpoint saved as nnsk.iter44
    DEEPTB INFO    Epoch 44 summary:	train_loss: 0.048096	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep44
    DEEPTB INFO    iteration:45	train_loss: 0.047749  (0.052309)	lr: 0.009569
    DEEPTB INFO    checkpoint saved as nnsk.iter45
    DEEPTB INFO    Epoch 45 summary:	train_loss: 0.047749	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep45
    DEEPTB INFO    iteration:46	train_loss: 0.048553  (0.051182)	lr: 0.00956 
    DEEPTB INFO    checkpoint saved as nnsk.iter46
    DEEPTB INFO    Epoch 46 summary:	train_loss: 0.048553	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    iteration:47	train_loss: 0.049181  (0.050582)	lr: 0.00955 
    DEEPTB INFO    checkpoint saved as nnsk.iter47
    DEEPTB INFO    Epoch 47 summary:	train_loss: 0.049181	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    iteration:48	train_loss: 0.048701  (0.050018)	lr: 0.009541
    DEEPTB INFO    checkpoint saved as nnsk.iter48
    DEEPTB INFO    Epoch 48 summary:	train_loss: 0.048701	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    iteration:49	train_loss: 0.047034  (0.049122)	lr: 0.009531
    DEEPTB INFO    checkpoint saved as nnsk.iter49
    DEEPTB INFO    Epoch 49 summary:	train_loss: 0.047034	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep49
    DEEPTB INFO    iteration:50	train_loss: 0.044856  (0.047842)	lr: 0.009522
    DEEPTB INFO    checkpoint saved as nnsk.iter50
    DEEPTB INFO    Epoch 50 summary:	train_loss: 0.044856	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep50
    DEEPTB INFO    iteration:51	train_loss: 0.043061  (0.046408)	lr: 0.009512
    DEEPTB INFO    checkpoint saved as nnsk.iter51
    DEEPTB INFO    Epoch 51 summary:	train_loss: 0.043061	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep51
    DEEPTB INFO    iteration:52	train_loss: 0.042162  (0.045134)	lr: 0.009503
    DEEPTB INFO    checkpoint saved as nnsk.iter52
    DEEPTB INFO    Epoch 52 summary:	train_loss: 0.042162	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep52
    DEEPTB INFO    iteration:53	train_loss: 0.042025  (0.044202)	lr: 0.009493
    DEEPTB INFO    checkpoint saved as nnsk.iter53
    DEEPTB INFO    Epoch 53 summary:	train_loss: 0.042025	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep53
    DEEPTB INFO    iteration:54	train_loss: 0.042054  (0.043557)	lr: 0.009484
    DEEPTB INFO    checkpoint saved as nnsk.iter54
    DEEPTB INFO    Epoch 54 summary:	train_loss: 0.042054	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    iteration:55	train_loss: 0.041654  (0.042986)	lr: 0.009474
    DEEPTB INFO    checkpoint saved as nnsk.iter55
    DEEPTB INFO    Epoch 55 summary:	train_loss: 0.041654	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep55
    DEEPTB INFO    iteration:56	train_loss: 0.040619  (0.042276)	lr: 0.009465
    DEEPTB INFO    checkpoint saved as nnsk.iter56
    DEEPTB INFO    Epoch 56 summary:	train_loss: 0.040619	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep56
    DEEPTB INFO    iteration:57	train_loss: 0.039222  (0.041360)	lr: 0.009455
    DEEPTB INFO    checkpoint saved as nnsk.iter57
    DEEPTB INFO    Epoch 57 summary:	train_loss: 0.039222	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep57
    DEEPTB INFO    iteration:58	train_loss: 0.037962  (0.040341)	lr: 0.009446
    DEEPTB INFO    checkpoint saved as nnsk.iter58
    DEEPTB INFO    Epoch 58 summary:	train_loss: 0.037962	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep58
    DEEPTB INFO    iteration:59	train_loss: 0.037204  (0.039400)	lr: 0.009436
    DEEPTB INFO    checkpoint saved as nnsk.iter59
    DEEPTB INFO    Epoch 59 summary:	train_loss: 0.037204	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep59
    DEEPTB INFO    iteration:60	train_loss: 0.036955  (0.038666)	lr: 0.009427
    DEEPTB INFO    checkpoint saved as nnsk.iter60
    DEEPTB INFO    Epoch 60 summary:	train_loss: 0.036955	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep60
    DEEPTB INFO    iteration:61	train_loss: 0.036908  (0.038139)	lr: 0.009417
    DEEPTB INFO    checkpoint saved as nnsk.iter61
    DEEPTB INFO    Epoch 61 summary:	train_loss: 0.036908	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep61
    DEEPTB INFO    iteration:62	train_loss: 0.036699  (0.037707)	lr: 0.009408
    DEEPTB INFO    checkpoint saved as nnsk.iter62
    DEEPTB INFO    Epoch 62 summary:	train_loss: 0.036699	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep62
    DEEPTB INFO    iteration:63	train_loss: 0.036143  (0.037238)	lr: 0.009399
    DEEPTB INFO    checkpoint saved as nnsk.iter63
    DEEPTB INFO    Epoch 63 summary:	train_loss: 0.036143	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep63
    DEEPTB INFO    iteration:64	train_loss: 0.035332  (0.036666)	lr: 0.009389
    DEEPTB INFO    checkpoint saved as nnsk.iter64
    DEEPTB INFO    Epoch 64 summary:	train_loss: 0.035332	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep64
    DEEPTB INFO    iteration:65	train_loss: 0.034522  (0.036023)	lr: 0.00938 
    DEEPTB INFO    checkpoint saved as nnsk.iter65
    DEEPTB INFO    Epoch 65 summary:	train_loss: 0.034522	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep65
    DEEPTB INFO    iteration:66	train_loss: 0.033935  (0.035396)	lr: 0.00937 
    DEEPTB INFO    checkpoint saved as nnsk.iter66
    DEEPTB INFO    Epoch 66 summary:	train_loss: 0.033935	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep66
    DEEPTB INFO    iteration:67	train_loss: 0.033608  (0.034860)	lr: 0.009361
    DEEPTB INFO    checkpoint saved as nnsk.iter67
    DEEPTB INFO    Epoch 67 summary:	train_loss: 0.033608	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep67
    DEEPTB INFO    iteration:68	train_loss: 0.033400  (0.034422)	lr: 0.009352
    DEEPTB INFO    checkpoint saved as nnsk.iter68
    DEEPTB INFO    Epoch 68 summary:	train_loss: 0.033400	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep68
    DEEPTB INFO    iteration:69	train_loss: 0.033135  (0.034036)	lr: 0.009342
    DEEPTB INFO    checkpoint saved as nnsk.iter69
    DEEPTB INFO    Epoch 69 summary:	train_loss: 0.033135	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep69
    DEEPTB INFO    iteration:70	train_loss: 0.032723  (0.033642)	lr: 0.009333
    DEEPTB INFO    checkpoint saved as nnsk.iter70
    DEEPTB INFO    Epoch 70 summary:	train_loss: 0.032723	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep70
    DEEPTB INFO    iteration:71	train_loss: 0.032213  (0.033213)	lr: 0.009324
    DEEPTB INFO    checkpoint saved as nnsk.iter71
    DEEPTB INFO    Epoch 71 summary:	train_loss: 0.032213	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep71
    DEEPTB INFO    iteration:72	train_loss: 0.031730  (0.032768)	lr: 0.009314
    DEEPTB INFO    checkpoint saved as nnsk.iter72
    DEEPTB INFO    Epoch 72 summary:	train_loss: 0.031730	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep72
    DEEPTB INFO    iteration:73	train_loss: 0.031372  (0.032349)	lr: 0.009305
    DEEPTB INFO    checkpoint saved as nnsk.iter73
    DEEPTB INFO    Epoch 73 summary:	train_loss: 0.031372	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep73
    DEEPTB INFO    iteration:74	train_loss: 0.031137  (0.031986)	lr: 0.009296
    DEEPTB INFO    checkpoint saved as nnsk.iter74
    DEEPTB INFO    Epoch 74 summary:	train_loss: 0.031137	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep74
    DEEPTB INFO    iteration:75	train_loss: 0.030939  (0.031672)	lr: 0.009286
    DEEPTB INFO    checkpoint saved as nnsk.iter75
    DEEPTB INFO    Epoch 75 summary:	train_loss: 0.030939	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep75
    DEEPTB INFO    iteration:76	train_loss: 0.030683  (0.031375)	lr: 0.009277
    DEEPTB INFO    checkpoint saved as nnsk.iter76
    DEEPTB INFO    Epoch 76 summary:	train_loss: 0.030683	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep76
    DEEPTB INFO    iteration:77	train_loss: 0.030332  (0.031062)	lr: 0.009268
    DEEPTB INFO    checkpoint saved as nnsk.iter77
    DEEPTB INFO    Epoch 77 summary:	train_loss: 0.030332	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep77
    DEEPTB INFO    iteration:78	train_loss: 0.029930  (0.030722)	lr: 0.009259
    DEEPTB INFO    checkpoint saved as nnsk.iter78
    DEEPTB INFO    Epoch 78 summary:	train_loss: 0.029930	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep78
    DEEPTB INFO    iteration:79	train_loss: 0.029552  (0.030371)	lr: 0.009249
    DEEPTB INFO    checkpoint saved as nnsk.iter79
    DEEPTB INFO    Epoch 79 summary:	train_loss: 0.029552	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep79
    DEEPTB INFO    iteration:80	train_loss: 0.029249  (0.030035)	lr: 0.00924 
    DEEPTB INFO    checkpoint saved as nnsk.iter80
    DEEPTB INFO    Epoch 80 summary:	train_loss: 0.029249	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep80
    DEEPTB INFO    iteration:81	train_loss: 0.029012  (0.029728)	lr: 0.009231
    DEEPTB INFO    checkpoint saved as nnsk.iter81
    DEEPTB INFO    Epoch 81 summary:	train_loss: 0.029012	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep81
    DEEPTB INFO    iteration:82	train_loss: 0.028791  (0.029447)	lr: 0.009222
    DEEPTB INFO    checkpoint saved as nnsk.iter82
    DEEPTB INFO    Epoch 82 summary:	train_loss: 0.028791	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep82
    DEEPTB INFO    iteration:83	train_loss: 0.028539  (0.029174)	lr: 0.009212
    DEEPTB INFO    checkpoint saved as nnsk.iter83
    DEEPTB INFO    Epoch 83 summary:	train_loss: 0.028539	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep83
    DEEPTB INFO    iteration:84	train_loss: 0.028245  (0.028896)	lr: 0.009203
    DEEPTB INFO    checkpoint saved as nnsk.iter84
    DEEPTB INFO    Epoch 84 summary:	train_loss: 0.028245	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep84
    DEEPTB INFO    iteration:85	train_loss: 0.027937  (0.028608)	lr: 0.009194
    DEEPTB INFO    checkpoint saved as nnsk.iter85
    DEEPTB INFO    Epoch 85 summary:	train_loss: 0.027937	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep85
    DEEPTB INFO    iteration:86	train_loss: 0.027651  (0.028321)	lr: 0.009185
    DEEPTB INFO    checkpoint saved as nnsk.iter86
    DEEPTB INFO    Epoch 86 summary:	train_loss: 0.027651	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep86
    DEEPTB INFO    iteration:87	train_loss: 0.027403  (0.028046)	lr: 0.009176
    DEEPTB INFO    checkpoint saved as nnsk.iter87
    DEEPTB INFO    Epoch 87 summary:	train_loss: 0.027403	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep87
    DEEPTB INFO    iteration:88	train_loss: 0.027178  (0.027785)	lr: 0.009166
    DEEPTB INFO    checkpoint saved as nnsk.iter88
    DEEPTB INFO    Epoch 88 summary:	train_loss: 0.027178	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep88
    DEEPTB INFO    iteration:89	train_loss: 0.026951  (0.027535)	lr: 0.009157
    DEEPTB INFO    checkpoint saved as nnsk.iter89
    DEEPTB INFO    Epoch 89 summary:	train_loss: 0.026951	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep89
    DEEPTB INFO    iteration:90	train_loss: 0.026704  (0.027286)	lr: 0.009148
    DEEPTB INFO    checkpoint saved as nnsk.iter90
    DEEPTB INFO    Epoch 90 summary:	train_loss: 0.026704	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep90
    DEEPTB INFO    iteration:91	train_loss: 0.026443  (0.027033)	lr: 0.009139
    DEEPTB INFO    checkpoint saved as nnsk.iter91
    DEEPTB INFO    Epoch 91 summary:	train_loss: 0.026443	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep91
    DEEPTB INFO    iteration:92	train_loss: 0.026188  (0.026779)	lr: 0.00913 
    DEEPTB INFO    checkpoint saved as nnsk.iter92
    DEEPTB INFO    Epoch 92 summary:	train_loss: 0.026188	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep92
    DEEPTB INFO    iteration:93	train_loss: 0.025954  (0.026532)	lr: 0.009121
    DEEPTB INFO    checkpoint saved as nnsk.iter93
    DEEPTB INFO    Epoch 93 summary:	train_loss: 0.025954	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep93
    DEEPTB INFO    iteration:94	train_loss: 0.025741  (0.026295)	lr: 0.009112
    DEEPTB INFO    checkpoint saved as nnsk.iter94
    DEEPTB INFO    Epoch 94 summary:	train_loss: 0.025741	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep94
    DEEPTB INFO    iteration:95	train_loss: 0.025535  (0.026067)	lr: 0.009102
    DEEPTB INFO    checkpoint saved as nnsk.iter95
    DEEPTB INFO    Epoch 95 summary:	train_loss: 0.025535	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep95
    DEEPTB INFO    iteration:96	train_loss: 0.025321  (0.025843)	lr: 0.009093
    DEEPTB INFO    checkpoint saved as nnsk.iter96
    DEEPTB INFO    Epoch 96 summary:	train_loss: 0.025321	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep96
    DEEPTB INFO    iteration:97	train_loss: 0.025096  (0.025619)	lr: 0.009084
    DEEPTB INFO    checkpoint saved as nnsk.iter97
    DEEPTB INFO    Epoch 97 summary:	train_loss: 0.025096	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep97
    DEEPTB INFO    iteration:98	train_loss: 0.024869  (0.025394)	lr: 0.009075
    DEEPTB INFO    checkpoint saved as nnsk.iter98
    DEEPTB INFO    Epoch 98 summary:	train_loss: 0.024869	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep98
    DEEPTB INFO    iteration:99	train_loss: 0.024652  (0.025171)	lr: 0.009066
    DEEPTB INFO    checkpoint saved as nnsk.iter99
    DEEPTB INFO    Epoch 99 summary:	train_loss: 0.024652	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep99
    DEEPTB INFO    iteration:100	train_loss: 0.024448  (0.024955)	lr: 0.009057
    DEEPTB INFO    checkpoint saved as nnsk.iter100
    DEEPTB INFO    Epoch 100 summary:	train_loss: 0.024448	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep100
    DEEPTB INFO    iteration:101	train_loss: 0.024254  (0.024744)	lr: 0.009048
    DEEPTB INFO    checkpoint saved as nnsk.iter101
    DEEPTB INFO    Epoch 101 summary:	train_loss: 0.024254	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep101
    DEEPTB INFO    iteration:102	train_loss: 0.024060  (0.024539)	lr: 0.009039
    DEEPTB INFO    checkpoint saved as nnsk.iter102
    DEEPTB INFO    Epoch 102 summary:	train_loss: 0.024060	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep102
    DEEPTB INFO    iteration:103	train_loss: 0.023862  (0.024336)	lr: 0.00903 
    DEEPTB INFO    checkpoint saved as nnsk.iter103
    DEEPTB INFO    Epoch 103 summary:	train_loss: 0.023862	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep103
    DEEPTB INFO    iteration:104	train_loss: 0.023663  (0.024134)	lr: 0.009021
    DEEPTB INFO    checkpoint saved as nnsk.iter104
    DEEPTB INFO    Epoch 104 summary:	train_loss: 0.023663	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep104
    DEEPTB INFO    iteration:105	train_loss: 0.023468  (0.023934)	lr: 0.009012
    DEEPTB INFO    checkpoint saved as nnsk.iter105
    DEEPTB INFO    Epoch 105 summary:	train_loss: 0.023468	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep105
    DEEPTB INFO    iteration:106	train_loss: 0.023280  (0.023738)	lr: 0.009003
    DEEPTB INFO    checkpoint saved as nnsk.iter106
    DEEPTB INFO    Epoch 106 summary:	train_loss: 0.023280	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep106
    DEEPTB INFO    iteration:107	train_loss: 0.023099  (0.023546)	lr: 0.008994
    DEEPTB INFO    checkpoint saved as nnsk.iter107
    DEEPTB INFO    Epoch 107 summary:	train_loss: 0.023099	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep107
    DEEPTB INFO    iteration:108	train_loss: 0.022920  (0.023359)	lr: 0.008985
    DEEPTB INFO    checkpoint saved as nnsk.iter108
    DEEPTB INFO    Epoch 108 summary:	train_loss: 0.022920	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep108
    DEEPTB INFO    iteration:109	train_loss: 0.022740  (0.023173)	lr: 0.008976
    DEEPTB INFO    checkpoint saved as nnsk.iter109
    DEEPTB INFO    Epoch 109 summary:	train_loss: 0.022740	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep109
    DEEPTB INFO    iteration:110	train_loss: 0.022560  (0.022989)	lr: 0.008967
    DEEPTB INFO    checkpoint saved as nnsk.iter110
    DEEPTB INFO    Epoch 110 summary:	train_loss: 0.022560	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep110
    DEEPTB INFO    iteration:111	train_loss: 0.022383  (0.022807)	lr: 0.008958
    DEEPTB INFO    checkpoint saved as nnsk.iter111
    DEEPTB INFO    Epoch 111 summary:	train_loss: 0.022383	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep111
    DEEPTB INFO    iteration:112	train_loss: 0.022211  (0.022628)	lr: 0.008949
    DEEPTB INFO    checkpoint saved as nnsk.iter112
    DEEPTB INFO    Epoch 112 summary:	train_loss: 0.022211	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep112
    DEEPTB INFO    iteration:113	train_loss: 0.022044  (0.022453)	lr: 0.00894 
    DEEPTB INFO    checkpoint saved as nnsk.iter113
    DEEPTB INFO    Epoch 113 summary:	train_loss: 0.022044	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep113
    DEEPTB INFO    iteration:114	train_loss: 0.021880  (0.022281)	lr: 0.008931
    DEEPTB INFO    checkpoint saved as nnsk.iter114
    DEEPTB INFO    Epoch 114 summary:	train_loss: 0.021880	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep114
    DEEPTB INFO    iteration:115	train_loss: 0.021718  (0.022112)	lr: 0.008922
    DEEPTB INFO    checkpoint saved as nnsk.iter115
    DEEPTB INFO    Epoch 115 summary:	train_loss: 0.021718	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep115
    DEEPTB INFO    iteration:116	train_loss: 0.021555  (0.021945)	lr: 0.008913
    DEEPTB INFO    checkpoint saved as nnsk.iter116
    DEEPTB INFO    Epoch 116 summary:	train_loss: 0.021555	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep116
    DEEPTB INFO    iteration:117	train_loss: 0.021394  (0.021780)	lr: 0.008904
    DEEPTB INFO    checkpoint saved as nnsk.iter117
    DEEPTB INFO    Epoch 117 summary:	train_loss: 0.021394	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep117
    DEEPTB INFO    iteration:118	train_loss: 0.021236  (0.021616)	lr: 0.008895
    DEEPTB INFO    checkpoint saved as nnsk.iter118
    DEEPTB INFO    Epoch 118 summary:	train_loss: 0.021236	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep118
    DEEPTB INFO    iteration:119	train_loss: 0.021082  (0.021456)	lr: 0.008886
    DEEPTB INFO    checkpoint saved as nnsk.iter119
    DEEPTB INFO    Epoch 119 summary:	train_loss: 0.021082	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep119
    DEEPTB INFO    iteration:120	train_loss: 0.020931  (0.021299)	lr: 0.008878
    DEEPTB INFO    checkpoint saved as nnsk.iter120
    DEEPTB INFO    Epoch 120 summary:	train_loss: 0.020931	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep120
    DEEPTB INFO    iteration:121	train_loss: 0.020781  (0.021143)	lr: 0.008869
    DEEPTB INFO    checkpoint saved as nnsk.iter121
    DEEPTB INFO    Epoch 121 summary:	train_loss: 0.020781	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep121
    DEEPTB INFO    iteration:122	train_loss: 0.020633  (0.020990)	lr: 0.00886 
    DEEPTB INFO    checkpoint saved as nnsk.iter122
    DEEPTB INFO    Epoch 122 summary:	train_loss: 0.020633	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep122
    DEEPTB INFO    iteration:123	train_loss: 0.020487  (0.020839)	lr: 0.008851
    DEEPTB INFO    checkpoint saved as nnsk.iter123
    DEEPTB INFO    Epoch 123 summary:	train_loss: 0.020487	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep123
    DEEPTB INFO    iteration:124	train_loss: 0.020343  (0.020690)	lr: 0.008842
    DEEPTB INFO    checkpoint saved as nnsk.iter124
    DEEPTB INFO    Epoch 124 summary:	train_loss: 0.020343	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep124
    DEEPTB INFO    iteration:125	train_loss: 0.020202  (0.020544)	lr: 0.008833
    DEEPTB INFO    checkpoint saved as nnsk.iter125
    DEEPTB INFO    Epoch 125 summary:	train_loss: 0.020202	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep125
    DEEPTB INFO    iteration:126	train_loss: 0.020063  (0.020399)	lr: 0.008824
    DEEPTB INFO    checkpoint saved as nnsk.iter126
    DEEPTB INFO    Epoch 126 summary:	train_loss: 0.020063	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep126
    DEEPTB INFO    iteration:127	train_loss: 0.019926  (0.020257)	lr: 0.008816
    DEEPTB INFO    checkpoint saved as nnsk.iter127
    DEEPTB INFO    Epoch 127 summary:	train_loss: 0.019926	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep127
    DEEPTB INFO    iteration:128	train_loss: 0.019790  (0.020117)	lr: 0.008807
    DEEPTB INFO    checkpoint saved as nnsk.iter128
    DEEPTB INFO    Epoch 128 summary:	train_loss: 0.019790	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep128
    DEEPTB INFO    iteration:129	train_loss: 0.019655  (0.019979)	lr: 0.008798
    DEEPTB INFO    checkpoint saved as nnsk.iter129
    DEEPTB INFO    Epoch 129 summary:	train_loss: 0.019655	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep129
    DEEPTB INFO    iteration:130	train_loss: 0.019524  (0.019842)	lr: 0.008789
    DEEPTB INFO    checkpoint saved as nnsk.iter130
    DEEPTB INFO    Epoch 130 summary:	train_loss: 0.019524	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep130
    DEEPTB INFO    iteration:131	train_loss: 0.019394  (0.019708)	lr: 0.00878 
    DEEPTB INFO    checkpoint saved as nnsk.iter131
    DEEPTB INFO    Epoch 131 summary:	train_loss: 0.019394	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep131
    DEEPTB INFO    iteration:132	train_loss: 0.019266  (0.019575)	lr: 0.008772
    DEEPTB INFO    checkpoint saved as nnsk.iter132
    DEEPTB INFO    Epoch 132 summary:	train_loss: 0.019266	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep132
    DEEPTB INFO    iteration:133	train_loss: 0.019140  (0.019445)	lr: 0.008763
    DEEPTB INFO    checkpoint saved as nnsk.iter133
    DEEPTB INFO    Epoch 133 summary:	train_loss: 0.019140	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep133
    DEEPTB INFO    iteration:134	train_loss: 0.019016  (0.019316)	lr: 0.008754
    DEEPTB INFO    checkpoint saved as nnsk.iter134
    DEEPTB INFO    Epoch 134 summary:	train_loss: 0.019016	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep134
    DEEPTB INFO    iteration:135	train_loss: 0.018893  (0.019189)	lr: 0.008745
    DEEPTB INFO    checkpoint saved as nnsk.iter135
    DEEPTB INFO    Epoch 135 summary:	train_loss: 0.018893	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep135
    DEEPTB INFO    iteration:136	train_loss: 0.018772  (0.019064)	lr: 0.008737
    DEEPTB INFO    checkpoint saved as nnsk.iter136
    DEEPTB INFO    Epoch 136 summary:	train_loss: 0.018772	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep136
    DEEPTB INFO    iteration:137	train_loss: 0.018653  (0.018941)	lr: 0.008728
    DEEPTB INFO    checkpoint saved as nnsk.iter137
    DEEPTB INFO    Epoch 137 summary:	train_loss: 0.018653	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep137
    DEEPTB INFO    iteration:138	train_loss: 0.018535  (0.018819)	lr: 0.008719
    DEEPTB INFO    checkpoint saved as nnsk.iter138
    DEEPTB INFO    Epoch 138 summary:	train_loss: 0.018535	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep138
    DEEPTB INFO    iteration:139	train_loss: 0.018420  (0.018699)	lr: 0.00871 
    DEEPTB INFO    checkpoint saved as nnsk.iter139
    DEEPTB INFO    Epoch 139 summary:	train_loss: 0.018420	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep139
    DEEPTB INFO    iteration:140	train_loss: 0.018305  (0.018581)	lr: 0.008702
    DEEPTB INFO    checkpoint saved as nnsk.iter140
    DEEPTB INFO    Epoch 140 summary:	train_loss: 0.018305	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep140
    DEEPTB INFO    iteration:141	train_loss: 0.018192  (0.018464)	lr: 0.008693
    DEEPTB INFO    checkpoint saved as nnsk.iter141
    DEEPTB INFO    Epoch 141 summary:	train_loss: 0.018192	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep141
    DEEPTB INFO    iteration:142	train_loss: 0.018081  (0.018349)	lr: 0.008684
    DEEPTB INFO    checkpoint saved as nnsk.iter142
    DEEPTB INFO    Epoch 142 summary:	train_loss: 0.018081	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep142
    DEEPTB INFO    iteration:143	train_loss: 0.017971  (0.018236)	lr: 0.008676
    DEEPTB INFO    checkpoint saved as nnsk.iter143
    DEEPTB INFO    Epoch 143 summary:	train_loss: 0.017971	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep143
    DEEPTB INFO    iteration:144	train_loss: 0.017863  (0.018124)	lr: 0.008667
    DEEPTB INFO    checkpoint saved as nnsk.iter144
    DEEPTB INFO    Epoch 144 summary:	train_loss: 0.017863	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep144
    DEEPTB INFO    iteration:145	train_loss: 0.017757  (0.018014)	lr: 0.008658
    DEEPTB INFO    checkpoint saved as nnsk.iter145
    DEEPTB INFO    Epoch 145 summary:	train_loss: 0.017757	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep145
    DEEPTB INFO    iteration:146	train_loss: 0.017652  (0.017905)	lr: 0.00865 
    DEEPTB INFO    checkpoint saved as nnsk.iter146
    DEEPTB INFO    Epoch 146 summary:	train_loss: 0.017652	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep146
    DEEPTB INFO    iteration:147	train_loss: 0.017548  (0.017798)	lr: 0.008641
    DEEPTB INFO    checkpoint saved as nnsk.iter147
    DEEPTB INFO    Epoch 147 summary:	train_loss: 0.017548	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep147
    DEEPTB INFO    iteration:148	train_loss: 0.017445  (0.017692)	lr: 0.008632
    DEEPTB INFO    checkpoint saved as nnsk.iter148
    DEEPTB INFO    Epoch 148 summary:	train_loss: 0.017445	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep148
    DEEPTB INFO    iteration:149	train_loss: 0.017345  (0.017588)	lr: 0.008624
    DEEPTB INFO    checkpoint saved as nnsk.iter149
    DEEPTB INFO    Epoch 149 summary:	train_loss: 0.017345	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep149
    DEEPTB INFO    iteration:150	train_loss: 0.017245  (0.017485)	lr: 0.008615
    DEEPTB INFO    checkpoint saved as nnsk.iter150
    DEEPTB INFO    Epoch 150 summary:	train_loss: 0.017245	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep150
    DEEPTB INFO    iteration:151	train_loss: 0.017147  (0.017384)	lr: 0.008606
    DEEPTB INFO    checkpoint saved as nnsk.iter151
    DEEPTB INFO    Epoch 151 summary:	train_loss: 0.017147	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep151
    DEEPTB INFO    iteration:152	train_loss: 0.017050  (0.017284)	lr: 0.008598
    DEEPTB INFO    checkpoint saved as nnsk.iter152
    DEEPTB INFO    Epoch 152 summary:	train_loss: 0.017050	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep152
    DEEPTB INFO    iteration:153	train_loss: 0.016955  (0.017185)	lr: 0.008589
    DEEPTB INFO    checkpoint saved as nnsk.iter153
    DEEPTB INFO    Epoch 153 summary:	train_loss: 0.016955	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep153
    DEEPTB INFO    iteration:154	train_loss: 0.016860  (0.017088)	lr: 0.008581
    DEEPTB INFO    checkpoint saved as nnsk.iter154
    DEEPTB INFO    Epoch 154 summary:	train_loss: 0.016860	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep154
    DEEPTB INFO    iteration:155	train_loss: 0.016767  (0.016992)	lr: 0.008572
    DEEPTB INFO    checkpoint saved as nnsk.iter155
    DEEPTB INFO    Epoch 155 summary:	train_loss: 0.016767	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep155
    DEEPTB INFO    iteration:156	train_loss: 0.016676  (0.016897)	lr: 0.008563
    DEEPTB INFO    checkpoint saved as nnsk.iter156
    DEEPTB INFO    Epoch 156 summary:	train_loss: 0.016676	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep156
    DEEPTB INFO    iteration:157	train_loss: 0.016586  (0.016804)	lr: 0.008555
    DEEPTB INFO    checkpoint saved as nnsk.iter157
    DEEPTB INFO    Epoch 157 summary:	train_loss: 0.016586	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep157
    DEEPTB INFO    iteration:158	train_loss: 0.016497  (0.016711)	lr: 0.008546
    DEEPTB INFO    checkpoint saved as nnsk.iter158
    DEEPTB INFO    Epoch 158 summary:	train_loss: 0.016497	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep158
    DEEPTB INFO    iteration:159	train_loss: 0.016409  (0.016621)	lr: 0.008538
    DEEPTB INFO    checkpoint saved as nnsk.iter159
    DEEPTB INFO    Epoch 159 summary:	train_loss: 0.016409	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep159
    DEEPTB INFO    iteration:160	train_loss: 0.016322  (0.016531)	lr: 0.008529
    DEEPTB INFO    checkpoint saved as nnsk.iter160
    DEEPTB INFO    Epoch 160 summary:	train_loss: 0.016322	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep160
    DEEPTB INFO    iteration:161	train_loss: 0.016236  (0.016443)	lr: 0.008521
    DEEPTB INFO    checkpoint saved as nnsk.iter161
    DEEPTB INFO    Epoch 161 summary:	train_loss: 0.016236	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep161
    DEEPTB INFO    iteration:162	train_loss: 0.016152  (0.016355)	lr: 0.008512
    DEEPTB INFO    checkpoint saved as nnsk.iter162
    DEEPTB INFO    Epoch 162 summary:	train_loss: 0.016152	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep162
    DEEPTB INFO    iteration:163	train_loss: 0.016069  (0.016269)	lr: 0.008504
    DEEPTB INFO    checkpoint saved as nnsk.iter163
    DEEPTB INFO    Epoch 163 summary:	train_loss: 0.016069	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep163
    DEEPTB INFO    iteration:164	train_loss: 0.015986  (0.016184)	lr: 0.008495
    DEEPTB INFO    checkpoint saved as nnsk.iter164
    DEEPTB INFO    Epoch 164 summary:	train_loss: 0.015986	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep164
    DEEPTB INFO    iteration:165	train_loss: 0.015905  (0.016101)	lr: 0.008487
    DEEPTB INFO    checkpoint saved as nnsk.iter165
    DEEPTB INFO    Epoch 165 summary:	train_loss: 0.015905	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep165
    DEEPTB INFO    iteration:166	train_loss: 0.015825  (0.016018)	lr: 0.008478
    DEEPTB INFO    checkpoint saved as nnsk.iter166
    DEEPTB INFO    Epoch 166 summary:	train_loss: 0.015825	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep166
    DEEPTB INFO    iteration:167	train_loss: 0.015747  (0.015937)	lr: 0.00847 
    DEEPTB INFO    checkpoint saved as nnsk.iter167
    DEEPTB INFO    Epoch 167 summary:	train_loss: 0.015747	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep167
    DEEPTB INFO    iteration:168	train_loss: 0.015669  (0.015856)	lr: 0.008461
    DEEPTB INFO    checkpoint saved as nnsk.iter168
    DEEPTB INFO    Epoch 168 summary:	train_loss: 0.015669	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep168
    DEEPTB INFO    iteration:169	train_loss: 0.015592  (0.015777)	lr: 0.008453
    DEEPTB INFO    checkpoint saved as nnsk.iter169
    DEEPTB INFO    Epoch 169 summary:	train_loss: 0.015592	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep169
    DEEPTB INFO    iteration:170	train_loss: 0.015516  (0.015699)	lr: 0.008444
    DEEPTB INFO    checkpoint saved as nnsk.iter170
    DEEPTB INFO    Epoch 170 summary:	train_loss: 0.015516	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep170
    DEEPTB INFO    iteration:171	train_loss: 0.015441  (0.015622)	lr: 0.008436
    DEEPTB INFO    checkpoint saved as nnsk.iter171
    DEEPTB INFO    Epoch 171 summary:	train_loss: 0.015441	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep171
    DEEPTB INFO    iteration:172	train_loss: 0.015368  (0.015545)	lr: 0.008427
    DEEPTB INFO    checkpoint saved as nnsk.iter172
    DEEPTB INFO    Epoch 172 summary:	train_loss: 0.015368	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep172
    DEEPTB INFO    iteration:173	train_loss: 0.015295  (0.015470)	lr: 0.008419
    DEEPTB INFO    checkpoint saved as nnsk.iter173
    DEEPTB INFO    Epoch 173 summary:	train_loss: 0.015295	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep173
    DEEPTB INFO    iteration:174	train_loss: 0.015223  (0.015396)	lr: 0.008411
    DEEPTB INFO    checkpoint saved as nnsk.iter174
    DEEPTB INFO    Epoch 174 summary:	train_loss: 0.015223	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep174
    DEEPTB INFO    iteration:175	train_loss: 0.015152  (0.015323)	lr: 0.008402
    DEEPTB INFO    checkpoint saved as nnsk.iter175
    DEEPTB INFO    Epoch 175 summary:	train_loss: 0.015152	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep175
    DEEPTB INFO    iteration:176	train_loss: 0.015082  (0.015251)	lr: 0.008394
    DEEPTB INFO    checkpoint saved as nnsk.iter176
    DEEPTB INFO    Epoch 176 summary:	train_loss: 0.015082	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep176
    DEEPTB INFO    iteration:177	train_loss: 0.015013  (0.015179)	lr: 0.008385
    DEEPTB INFO    checkpoint saved as nnsk.iter177
    DEEPTB INFO    Epoch 177 summary:	train_loss: 0.015013	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep177
    DEEPTB INFO    iteration:178	train_loss: 0.014945  (0.015109)	lr: 0.008377
    DEEPTB INFO    checkpoint saved as nnsk.iter178
    DEEPTB INFO    Epoch 178 summary:	train_loss: 0.014945	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep178
    DEEPTB INFO    iteration:179	train_loss: 0.014877  (0.015039)	lr: 0.008369
    DEEPTB INFO    checkpoint saved as nnsk.iter179
    DEEPTB INFO    Epoch 179 summary:	train_loss: 0.014877	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep179
    DEEPTB INFO    iteration:180	train_loss: 0.014811  (0.014971)	lr: 0.00836 
    DEEPTB INFO    checkpoint saved as nnsk.iter180
    DEEPTB INFO    Epoch 180 summary:	train_loss: 0.014811	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep180
    DEEPTB INFO    iteration:181	train_loss: 0.014745  (0.014903)	lr: 0.008352
    DEEPTB INFO    checkpoint saved as nnsk.iter181
    DEEPTB INFO    Epoch 181 summary:	train_loss: 0.014745	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep181
    DEEPTB INFO    iteration:182	train_loss: 0.014681  (0.014836)	lr: 0.008344
    DEEPTB INFO    checkpoint saved as nnsk.iter182
    DEEPTB INFO    Epoch 182 summary:	train_loss: 0.014681	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep182
    DEEPTB INFO    iteration:183	train_loss: 0.014617  (0.014770)	lr: 0.008335
    DEEPTB INFO    checkpoint saved as nnsk.iter183
    DEEPTB INFO    Epoch 183 summary:	train_loss: 0.014617	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep183
    DEEPTB INFO    iteration:184	train_loss: 0.014553  (0.014705)	lr: 0.008327
    DEEPTB INFO    checkpoint saved as nnsk.iter184
    DEEPTB INFO    Epoch 184 summary:	train_loss: 0.014553	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep184
    DEEPTB INFO    iteration:185	train_loss: 0.014491  (0.014641)	lr: 0.008319
    DEEPTB INFO    checkpoint saved as nnsk.iter185
    DEEPTB INFO    Epoch 185 summary:	train_loss: 0.014491	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep185
    DEEPTB INFO    iteration:186	train_loss: 0.014430  (0.014578)	lr: 0.00831 
    DEEPTB INFO    checkpoint saved as nnsk.iter186
    DEEPTB INFO    Epoch 186 summary:	train_loss: 0.014430	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep186
    DEEPTB INFO    iteration:187	train_loss: 0.014369  (0.014515)	lr: 0.008302
    DEEPTB INFO    checkpoint saved as nnsk.iter187
    DEEPTB INFO    Epoch 187 summary:	train_loss: 0.014369	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep187
    DEEPTB INFO    iteration:188	train_loss: 0.014309  (0.014453)	lr: 0.008294
    DEEPTB INFO    checkpoint saved as nnsk.iter188
    DEEPTB INFO    Epoch 188 summary:	train_loss: 0.014309	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep188
    DEEPTB INFO    iteration:189	train_loss: 0.014249  (0.014392)	lr: 0.008285
    DEEPTB INFO    checkpoint saved as nnsk.iter189
    DEEPTB INFO    Epoch 189 summary:	train_loss: 0.014249	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep189
    DEEPTB INFO    iteration:190	train_loss: 0.014191  (0.014332)	lr: 0.008277
    DEEPTB INFO    checkpoint saved as nnsk.iter190
    DEEPTB INFO    Epoch 190 summary:	train_loss: 0.014191	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep190
    DEEPTB INFO    iteration:191	train_loss: 0.014133  (0.014272)	lr: 0.008269
    DEEPTB INFO    checkpoint saved as nnsk.iter191
    DEEPTB INFO    Epoch 191 summary:	train_loss: 0.014133	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep191
    DEEPTB INFO    iteration:192	train_loss: 0.014077  (0.014213)	lr: 0.008261
    DEEPTB INFO    checkpoint saved as nnsk.iter192
    DEEPTB INFO    Epoch 192 summary:	train_loss: 0.014077	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep192
    DEEPTB INFO    iteration:193	train_loss: 0.014023  (0.014156)	lr: 0.008252
    DEEPTB INFO    checkpoint saved as nnsk.iter193
    DEEPTB INFO    Epoch 193 summary:	train_loss: 0.014023	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep193
    DEEPTB INFO    iteration:194	train_loss: 0.013966  (0.014099)	lr: 0.008244
    DEEPTB INFO    checkpoint saved as nnsk.iter194
    DEEPTB INFO    Epoch 194 summary:	train_loss: 0.013966	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep194
    DEEPTB INFO    iteration:195	train_loss: 0.013915  (0.014044)	lr: 0.008236
    DEEPTB INFO    checkpoint saved as nnsk.iter195
    DEEPTB INFO    Epoch 195 summary:	train_loss: 0.013915	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep195
    DEEPTB INFO    iteration:196	train_loss: 0.013864  (0.013990)	lr: 0.008228
    DEEPTB INFO    checkpoint saved as nnsk.iter196
    DEEPTB INFO    Epoch 196 summary:	train_loss: 0.013864	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep196
    DEEPTB INFO    iteration:197	train_loss: 0.013813  (0.013937)	lr: 0.008219
    DEEPTB INFO    checkpoint saved as nnsk.iter197
    DEEPTB INFO    Epoch 197 summary:	train_loss: 0.013813	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep197
    DEEPTB INFO    iteration:198	train_loss: 0.013762  (0.013885)	lr: 0.008211
    DEEPTB INFO    checkpoint saved as nnsk.iter198
    DEEPTB INFO    Epoch 198 summary:	train_loss: 0.013762	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep198
    DEEPTB INFO    iteration:199	train_loss: 0.013711  (0.013833)	lr: 0.008203
    DEEPTB INFO    checkpoint saved as nnsk.iter199
    DEEPTB INFO    Epoch 199 summary:	train_loss: 0.013711	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep199
    DEEPTB INFO    iteration:200	train_loss: 0.013660  (0.013781)	lr: 0.008195
    DEEPTB INFO    checkpoint saved as nnsk.iter200
    DEEPTB INFO    Epoch 200 summary:	train_loss: 0.013660	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep200
    DEEPTB INFO    iteration:201	train_loss: 0.013611  (0.013730)	lr: 0.008186
    DEEPTB INFO    checkpoint saved as nnsk.iter201
    DEEPTB INFO    Epoch 201 summary:	train_loss: 0.013611	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep201
    DEEPTB INFO    iteration:202	train_loss: 0.013565  (0.013680)	lr: 0.008178
    DEEPTB INFO    checkpoint saved as nnsk.iter202
    DEEPTB INFO    Epoch 202 summary:	train_loss: 0.013565	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep202
    DEEPTB INFO    iteration:203	train_loss: 0.013516  (0.013631)	lr: 0.00817 
    DEEPTB INFO    checkpoint saved as nnsk.iter203
    DEEPTB INFO    Epoch 203 summary:	train_loss: 0.013516	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep203
    DEEPTB INFO    iteration:204	train_loss: 0.013468  (0.013582)	lr: 0.008162
    DEEPTB INFO    checkpoint saved as nnsk.iter204
    DEEPTB INFO    Epoch 204 summary:	train_loss: 0.013468	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep204
    DEEPTB INFO    iteration:205	train_loss: 0.013422  (0.013534)	lr: 0.008154
    DEEPTB INFO    checkpoint saved as nnsk.iter205
    DEEPTB INFO    Epoch 205 summary:	train_loss: 0.013422	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep205
    DEEPTB INFO    iteration:206	train_loss: 0.013376  (0.013487)	lr: 0.008146
    DEEPTB INFO    checkpoint saved as nnsk.iter206
    DEEPTB INFO    Epoch 206 summary:	train_loss: 0.013376	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep206
    DEEPTB INFO    iteration:207	train_loss: 0.013332  (0.013440)	lr: 0.008137
    DEEPTB INFO    checkpoint saved as nnsk.iter207
    DEEPTB INFO    Epoch 207 summary:	train_loss: 0.013332	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep207
    DEEPTB INFO    iteration:208	train_loss: 0.013287  (0.013394)	lr: 0.008129
    DEEPTB INFO    checkpoint saved as nnsk.iter208
    DEEPTB INFO    Epoch 208 summary:	train_loss: 0.013287	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep208
    DEEPTB INFO    iteration:209	train_loss: 0.013244  (0.013349)	lr: 0.008121
    DEEPTB INFO    checkpoint saved as nnsk.iter209
    DEEPTB INFO    Epoch 209 summary:	train_loss: 0.013244	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep209
    DEEPTB INFO    iteration:210	train_loss: 0.013201  (0.013305)	lr: 0.008113
    DEEPTB INFO    checkpoint saved as nnsk.iter210
    DEEPTB INFO    Epoch 210 summary:	train_loss: 0.013201	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep210
    DEEPTB INFO    iteration:211	train_loss: 0.013158  (0.013261)	lr: 0.008105
    DEEPTB INFO    checkpoint saved as nnsk.iter211
    DEEPTB INFO    Epoch 211 summary:	train_loss: 0.013158	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep211
    DEEPTB INFO    iteration:212	train_loss: 0.013116  (0.013217)	lr: 0.008097
    DEEPTB INFO    checkpoint saved as nnsk.iter212
    DEEPTB INFO    Epoch 212 summary:	train_loss: 0.013116	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep212
    DEEPTB INFO    iteration:213	train_loss: 0.013074  (0.013174)	lr: 0.008089
    DEEPTB INFO    checkpoint saved as nnsk.iter213
    DEEPTB INFO    Epoch 213 summary:	train_loss: 0.013074	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep213
    DEEPTB INFO    iteration:214	train_loss: 0.013033  (0.013132)	lr: 0.008081
    DEEPTB INFO    checkpoint saved as nnsk.iter214
    DEEPTB INFO    Epoch 214 summary:	train_loss: 0.013033	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep214
    DEEPTB INFO    iteration:215	train_loss: 0.012993  (0.013090)	lr: 0.008073
    DEEPTB INFO    checkpoint saved as nnsk.iter215
    DEEPTB INFO    Epoch 215 summary:	train_loss: 0.012993	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep215
    DEEPTB INFO    iteration:216	train_loss: 0.012954  (0.013049)	lr: 0.008065
    DEEPTB INFO    checkpoint saved as nnsk.iter216
    DEEPTB INFO    Epoch 216 summary:	train_loss: 0.012954	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep216
    DEEPTB INFO    iteration:217	train_loss: 0.012914  (0.013009)	lr: 0.008056
    DEEPTB INFO    checkpoint saved as nnsk.iter217
    DEEPTB INFO    Epoch 217 summary:	train_loss: 0.012914	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep217
    DEEPTB INFO    iteration:218	train_loss: 0.012874  (0.012968)	lr: 0.008048
    DEEPTB INFO    checkpoint saved as nnsk.iter218
    DEEPTB INFO    Epoch 218 summary:	train_loss: 0.012874	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep218
    DEEPTB INFO    iteration:219	train_loss: 0.012835  (0.012928)	lr: 0.00804 
    DEEPTB INFO    checkpoint saved as nnsk.iter219
    DEEPTB INFO    Epoch 219 summary:	train_loss: 0.012835	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep219
    DEEPTB INFO    iteration:220	train_loss: 0.012797  (0.012889)	lr: 0.008032
    DEEPTB INFO    checkpoint saved as nnsk.iter220
    DEEPTB INFO    Epoch 220 summary:	train_loss: 0.012797	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep220
    DEEPTB INFO    iteration:221	train_loss: 0.012759  (0.012850)	lr: 0.008024
    DEEPTB INFO    checkpoint saved as nnsk.iter221
    DEEPTB INFO    Epoch 221 summary:	train_loss: 0.012759	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep221
    DEEPTB INFO    iteration:222	train_loss: 0.012722  (0.012811)	lr: 0.008016
    DEEPTB INFO    checkpoint saved as nnsk.iter222
    DEEPTB INFO    Epoch 222 summary:	train_loss: 0.012722	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep222
    DEEPTB INFO    iteration:223	train_loss: 0.012684  (0.012773)	lr: 0.008008
    DEEPTB INFO    checkpoint saved as nnsk.iter223
    DEEPTB INFO    Epoch 223 summary:	train_loss: 0.012684	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep223
    DEEPTB INFO    iteration:224	train_loss: 0.012647  (0.012735)	lr: 0.008   
    DEEPTB INFO    checkpoint saved as nnsk.iter224
    DEEPTB INFO    Epoch 224 summary:	train_loss: 0.012647	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep224
    DEEPTB INFO    iteration:225	train_loss: 0.012610  (0.012698)	lr: 0.007992
    DEEPTB INFO    checkpoint saved as nnsk.iter225
    DEEPTB INFO    Epoch 225 summary:	train_loss: 0.012610	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep225
    DEEPTB INFO    iteration:226	train_loss: 0.012575  (0.012661)	lr: 0.007984
    DEEPTB INFO    checkpoint saved as nnsk.iter226
    DEEPTB INFO    Epoch 226 summary:	train_loss: 0.012575	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep226
    DEEPTB INFO    iteration:227	train_loss: 0.012538  (0.012624)	lr: 0.007976
    DEEPTB INFO    checkpoint saved as nnsk.iter227
    DEEPTB INFO    Epoch 227 summary:	train_loss: 0.012538	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep227
    DEEPTB INFO    iteration:228	train_loss: 0.012503  (0.012588)	lr: 0.007968
    DEEPTB INFO    checkpoint saved as nnsk.iter228
    DEEPTB INFO    Epoch 228 summary:	train_loss: 0.012503	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep228
    DEEPTB INFO    iteration:229	train_loss: 0.012469  (0.012552)	lr: 0.00796 
    DEEPTB INFO    checkpoint saved as nnsk.iter229
    DEEPTB INFO    Epoch 229 summary:	train_loss: 0.012469	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep229
    DEEPTB INFO    iteration:230	train_loss: 0.012434  (0.012517)	lr: 0.007952
    DEEPTB INFO    checkpoint saved as nnsk.iter230
    DEEPTB INFO    Epoch 230 summary:	train_loss: 0.012434	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep230
    DEEPTB INFO    iteration:231	train_loss: 0.012399  (0.012481)	lr: 0.007944
    DEEPTB INFO    checkpoint saved as nnsk.iter231
    DEEPTB INFO    Epoch 231 summary:	train_loss: 0.012399	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep231
    DEEPTB INFO    iteration:232	train_loss: 0.012364  (0.012446)	lr: 0.007936
    DEEPTB INFO    checkpoint saved as nnsk.iter232
    DEEPTB INFO    Epoch 232 summary:	train_loss: 0.012364	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep232
    DEEPTB INFO    iteration:233	train_loss: 0.012329  (0.012411)	lr: 0.007929
    DEEPTB INFO    checkpoint saved as nnsk.iter233
    DEEPTB INFO    Epoch 233 summary:	train_loss: 0.012329	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep233
    DEEPTB INFO    iteration:234	train_loss: 0.012297  (0.012377)	lr: 0.007921
    DEEPTB INFO    checkpoint saved as nnsk.iter234
    DEEPTB INFO    Epoch 234 summary:	train_loss: 0.012297	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep234
    DEEPTB INFO    iteration:235	train_loss: 0.012263  (0.012343)	lr: 0.007913
    DEEPTB INFO    checkpoint saved as nnsk.iter235
    DEEPTB INFO    Epoch 235 summary:	train_loss: 0.012263	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep235
    DEEPTB INFO    iteration:236	train_loss: 0.012230  (0.012309)	lr: 0.007905
    DEEPTB INFO    checkpoint saved as nnsk.iter236
    DEEPTB INFO    Epoch 236 summary:	train_loss: 0.012230	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep236
    DEEPTB INFO    iteration:237	train_loss: 0.012196  (0.012275)	lr: 0.007897
    DEEPTB INFO    checkpoint saved as nnsk.iter237
    DEEPTB INFO    Epoch 237 summary:	train_loss: 0.012196	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep237
    DEEPTB INFO    iteration:238	train_loss: 0.012166  (0.012242)	lr: 0.007889
    DEEPTB INFO    checkpoint saved as nnsk.iter238
    DEEPTB INFO    Epoch 238 summary:	train_loss: 0.012166	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep238
    DEEPTB INFO    iteration:239	train_loss: 0.012133  (0.012210)	lr: 0.007881
    DEEPTB INFO    checkpoint saved as nnsk.iter239
    DEEPTB INFO    Epoch 239 summary:	train_loss: 0.012133	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep239
    DEEPTB INFO    iteration:240	train_loss: 0.012100  (0.012177)	lr: 0.007873
    DEEPTB INFO    checkpoint saved as nnsk.iter240
    DEEPTB INFO    Epoch 240 summary:	train_loss: 0.012100	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep240
    DEEPTB INFO    iteration:241	train_loss: 0.012068  (0.012144)	lr: 0.007865
    DEEPTB INFO    checkpoint saved as nnsk.iter241
    DEEPTB INFO    Epoch 241 summary:	train_loss: 0.012068	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep241
    DEEPTB INFO    iteration:242	train_loss: 0.012037  (0.012112)	lr: 0.007857
    DEEPTB INFO    checkpoint saved as nnsk.iter242
    DEEPTB INFO    Epoch 242 summary:	train_loss: 0.012037	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep242
    DEEPTB INFO    iteration:243	train_loss: 0.012005  (0.012080)	lr: 0.00785 
    DEEPTB INFO    checkpoint saved as nnsk.iter243
    DEEPTB INFO    Epoch 243 summary:	train_loss: 0.012005	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep243
    DEEPTB INFO    iteration:244	train_loss: 0.011973  (0.012048)	lr: 0.007842
    DEEPTB INFO    checkpoint saved as nnsk.iter244
    DEEPTB INFO    Epoch 244 summary:	train_loss: 0.011973	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep244
    DEEPTB INFO    iteration:245	train_loss: 0.011943  (0.012016)	lr: 0.007834
    DEEPTB INFO    checkpoint saved as nnsk.iter245
    DEEPTB INFO    Epoch 245 summary:	train_loss: 0.011943	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep245
    DEEPTB INFO    iteration:246	train_loss: 0.011911  (0.011985)	lr: 0.007826
    DEEPTB INFO    checkpoint saved as nnsk.iter246
    DEEPTB INFO    Epoch 246 summary:	train_loss: 0.011911	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep246
    DEEPTB INFO    iteration:247	train_loss: 0.011881  (0.011954)	lr: 0.007818
    DEEPTB INFO    checkpoint saved as nnsk.iter247
    DEEPTB INFO    Epoch 247 summary:	train_loss: 0.011881	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep247
    DEEPTB INFO    iteration:248	train_loss: 0.011852  (0.011923)	lr: 0.00781 
    DEEPTB INFO    checkpoint saved as nnsk.iter248
    DEEPTB INFO    Epoch 248 summary:	train_loss: 0.011852	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep248
    DEEPTB INFO    iteration:249	train_loss: 0.011821  (0.011893)	lr: 0.007803
    DEEPTB INFO    checkpoint saved as nnsk.iter249
    DEEPTB INFO    Epoch 249 summary:	train_loss: 0.011821	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep249
    DEEPTB INFO    iteration:250	train_loss: 0.011791  (0.011862)	lr: 0.007795
    DEEPTB INFO    checkpoint saved as nnsk.iter250
    DEEPTB INFO    Epoch 250 summary:	train_loss: 0.011791	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep250
    DEEPTB INFO    iteration:251	train_loss: 0.011761  (0.011832)	lr: 0.007787
    DEEPTB INFO    checkpoint saved as nnsk.iter251
    DEEPTB INFO    Epoch 251 summary:	train_loss: 0.011761	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep251
    DEEPTB INFO    iteration:252	train_loss: 0.011731  (0.011801)	lr: 0.007779
    DEEPTB INFO    checkpoint saved as nnsk.iter252
    DEEPTB INFO    Epoch 252 summary:	train_loss: 0.011731	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep252
    DEEPTB INFO    iteration:253	train_loss: 0.011702  (0.011771)	lr: 0.007771
    DEEPTB INFO    checkpoint saved as nnsk.iter253
    DEEPTB INFO    Epoch 253 summary:	train_loss: 0.011702	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep253
    DEEPTB INFO    iteration:254	train_loss: 0.011672  (0.011742)	lr: 0.007764
    DEEPTB INFO    checkpoint saved as nnsk.iter254
    DEEPTB INFO    Epoch 254 summary:	train_loss: 0.011672	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep254
    DEEPTB INFO    iteration:255	train_loss: 0.011643  (0.011712)	lr: 0.007756
    DEEPTB INFO    checkpoint saved as nnsk.iter255
    DEEPTB INFO    Epoch 255 summary:	train_loss: 0.011643	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep255
    DEEPTB INFO    iteration:256	train_loss: 0.011614  (0.011683)	lr: 0.007748
    DEEPTB INFO    checkpoint saved as nnsk.iter256
    DEEPTB INFO    Epoch 256 summary:	train_loss: 0.011614	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep256
    DEEPTB INFO    iteration:257	train_loss: 0.011586  (0.011654)	lr: 0.00774 
    DEEPTB INFO    checkpoint saved as nnsk.iter257
    DEEPTB INFO    Epoch 257 summary:	train_loss: 0.011586	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep257
    DEEPTB INFO    iteration:258	train_loss: 0.011557  (0.011625)	lr: 0.007733
    DEEPTB INFO    checkpoint saved as nnsk.iter258
    DEEPTB INFO    Epoch 258 summary:	train_loss: 0.011557	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep258
    DEEPTB INFO    iteration:259	train_loss: 0.011528  (0.011596)	lr: 0.007725
    DEEPTB INFO    checkpoint saved as nnsk.iter259
    DEEPTB INFO    Epoch 259 summary:	train_loss: 0.011528	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep259
    DEEPTB INFO    iteration:260	train_loss: 0.011499  (0.011567)	lr: 0.007717
    DEEPTB INFO    checkpoint saved as nnsk.iter260
    DEEPTB INFO    Epoch 260 summary:	train_loss: 0.011499	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep260
    DEEPTB INFO    iteration:261	train_loss: 0.011470  (0.011538)	lr: 0.00771 
    DEEPTB INFO    checkpoint saved as nnsk.iter261
    DEEPTB INFO    Epoch 261 summary:	train_loss: 0.011470	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep261
    DEEPTB INFO    iteration:262	train_loss: 0.011444  (0.011510)	lr: 0.007702
    DEEPTB INFO    checkpoint saved as nnsk.iter262
    DEEPTB INFO    Epoch 262 summary:	train_loss: 0.011444	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep262
    DEEPTB INFO    iteration:263	train_loss: 0.011416  (0.011482)	lr: 0.007694
    DEEPTB INFO    checkpoint saved as nnsk.iter263
    DEEPTB INFO    Epoch 263 summary:	train_loss: 0.011416	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep263
    DEEPTB INFO    iteration:264	train_loss: 0.011387  (0.011453)	lr: 0.007686
    DEEPTB INFO    checkpoint saved as nnsk.iter264
    DEEPTB INFO    Epoch 264 summary:	train_loss: 0.011387	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep264
    DEEPTB INFO    iteration:265	train_loss: 0.011360  (0.011425)	lr: 0.007679
    DEEPTB INFO    checkpoint saved as nnsk.iter265
    DEEPTB INFO    Epoch 265 summary:	train_loss: 0.011360	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep265
    DEEPTB INFO    iteration:266	train_loss: 0.011333  (0.011397)	lr: 0.007671
    DEEPTB INFO    checkpoint saved as nnsk.iter266
    DEEPTB INFO    Epoch 266 summary:	train_loss: 0.011333	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep266
    DEEPTB INFO    iteration:267	train_loss: 0.011305  (0.011370)	lr: 0.007663
    DEEPTB INFO    checkpoint saved as nnsk.iter267
    DEEPTB INFO    Epoch 267 summary:	train_loss: 0.011305	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep267
    DEEPTB INFO    iteration:268	train_loss: 0.011278  (0.011342)	lr: 0.007656
    DEEPTB INFO    checkpoint saved as nnsk.iter268
    DEEPTB INFO    Epoch 268 summary:	train_loss: 0.011278	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep268
    DEEPTB INFO    iteration:269	train_loss: 0.011250  (0.011315)	lr: 0.007648
    DEEPTB INFO    checkpoint saved as nnsk.iter269
    DEEPTB INFO    Epoch 269 summary:	train_loss: 0.011250	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep269
    DEEPTB INFO    iteration:270	train_loss: 0.011223  (0.011287)	lr: 0.00764 
    DEEPTB INFO    checkpoint saved as nnsk.iter270
    DEEPTB INFO    Epoch 270 summary:	train_loss: 0.011223	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep270
    DEEPTB INFO    iteration:271	train_loss: 0.011196  (0.011260)	lr: 0.007633
    DEEPTB INFO    checkpoint saved as nnsk.iter271
    DEEPTB INFO    Epoch 271 summary:	train_loss: 0.011196	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep271
    DEEPTB INFO    iteration:272	train_loss: 0.011169  (0.011232)	lr: 0.007625
    DEEPTB INFO    checkpoint saved as nnsk.iter272
    DEEPTB INFO    Epoch 272 summary:	train_loss: 0.011169	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep272
    DEEPTB INFO    iteration:273	train_loss: 0.011142  (0.011205)	lr: 0.007618
    DEEPTB INFO    checkpoint saved as nnsk.iter273
    DEEPTB INFO    Epoch 273 summary:	train_loss: 0.011142	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep273
    DEEPTB INFO    iteration:274	train_loss: 0.011115  (0.011178)	lr: 0.00761 
    DEEPTB INFO    checkpoint saved as nnsk.iter274
    DEEPTB INFO    Epoch 274 summary:	train_loss: 0.011115	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep274
    DEEPTB INFO    iteration:275	train_loss: 0.011090  (0.011152)	lr: 0.007602
    DEEPTB INFO    checkpoint saved as nnsk.iter275
    DEEPTB INFO    Epoch 275 summary:	train_loss: 0.011090	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep275
    DEEPTB INFO    iteration:276	train_loss: 0.011063  (0.011125)	lr: 0.007595
    DEEPTB INFO    checkpoint saved as nnsk.iter276
    DEEPTB INFO    Epoch 276 summary:	train_loss: 0.011063	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep276
    DEEPTB INFO    iteration:277	train_loss: 0.011037  (0.011099)	lr: 0.007587
    DEEPTB INFO    checkpoint saved as nnsk.iter277
    DEEPTB INFO    Epoch 277 summary:	train_loss: 0.011037	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep277
    DEEPTB INFO    iteration:278	train_loss: 0.011011  (0.011072)	lr: 0.007579
    DEEPTB INFO    checkpoint saved as nnsk.iter278
    DEEPTB INFO    Epoch 278 summary:	train_loss: 0.011011	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep278
    DEEPTB INFO    iteration:279	train_loss: 0.010985  (0.011046)	lr: 0.007572
    DEEPTB INFO    checkpoint saved as nnsk.iter279
    DEEPTB INFO    Epoch 279 summary:	train_loss: 0.010985	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep279
    DEEPTB INFO    iteration:280	train_loss: 0.010959  (0.011020)	lr: 0.007564
    DEEPTB INFO    checkpoint saved as nnsk.iter280
    DEEPTB INFO    Epoch 280 summary:	train_loss: 0.010959	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep280
    DEEPTB INFO    iteration:281	train_loss: 0.010932  (0.010994)	lr: 0.007557
    DEEPTB INFO    checkpoint saved as nnsk.iter281
    DEEPTB INFO    Epoch 281 summary:	train_loss: 0.010932	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep281
    DEEPTB INFO    iteration:282	train_loss: 0.010908  (0.010968)	lr: 0.007549
    DEEPTB INFO    checkpoint saved as nnsk.iter282
    DEEPTB INFO    Epoch 282 summary:	train_loss: 0.010908	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep282
    DEEPTB INFO    iteration:283	train_loss: 0.010881  (0.010942)	lr: 0.007542
    DEEPTB INFO    checkpoint saved as nnsk.iter283
    DEEPTB INFO    Epoch 283 summary:	train_loss: 0.010881	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep283
    DEEPTB INFO    iteration:284	train_loss: 0.010856  (0.010916)	lr: 0.007534
    DEEPTB INFO    checkpoint saved as nnsk.iter284
    DEEPTB INFO    Epoch 284 summary:	train_loss: 0.010856	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep284
    DEEPTB INFO    iteration:285	train_loss: 0.010830  (0.010890)	lr: 0.007527
    DEEPTB INFO    checkpoint saved as nnsk.iter285
    DEEPTB INFO    Epoch 285 summary:	train_loss: 0.010830	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep285
    DEEPTB INFO    iteration:286	train_loss: 0.010805  (0.010865)	lr: 0.007519
    DEEPTB INFO    checkpoint saved as nnsk.iter286
    DEEPTB INFO    Epoch 286 summary:	train_loss: 0.010805	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep286
    DEEPTB INFO    iteration:287	train_loss: 0.010780  (0.010839)	lr: 0.007512
    DEEPTB INFO    checkpoint saved as nnsk.iter287
    DEEPTB INFO    Epoch 287 summary:	train_loss: 0.010780	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep287
    DEEPTB INFO    iteration:288	train_loss: 0.010755  (0.010814)	lr: 0.007504
    DEEPTB INFO    checkpoint saved as nnsk.iter288
    DEEPTB INFO    Epoch 288 summary:	train_loss: 0.010755	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep288
    DEEPTB INFO    iteration:289	train_loss: 0.010730  (0.010789)	lr: 0.007497
    DEEPTB INFO    checkpoint saved as nnsk.iter289
    DEEPTB INFO    Epoch 289 summary:	train_loss: 0.010730	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep289
    DEEPTB INFO    iteration:290	train_loss: 0.010705  (0.010764)	lr: 0.007489
    DEEPTB INFO    checkpoint saved as nnsk.iter290
    DEEPTB INFO    Epoch 290 summary:	train_loss: 0.010705	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep290
    DEEPTB INFO    iteration:291	train_loss: 0.010680  (0.010738)	lr: 0.007482
    DEEPTB INFO    checkpoint saved as nnsk.iter291
    DEEPTB INFO    Epoch 291 summary:	train_loss: 0.010680	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep291
    DEEPTB INFO    iteration:292	train_loss: 0.010655  (0.010713)	lr: 0.007474
    DEEPTB INFO    checkpoint saved as nnsk.iter292
    DEEPTB INFO    Epoch 292 summary:	train_loss: 0.010655	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep292
    DEEPTB INFO    iteration:293	train_loss: 0.010630  (0.010688)	lr: 0.007467
    DEEPTB INFO    checkpoint saved as nnsk.iter293
    DEEPTB INFO    Epoch 293 summary:	train_loss: 0.010630	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep293
    DEEPTB INFO    iteration:294	train_loss: 0.010606  (0.010664)	lr: 0.007459
    DEEPTB INFO    checkpoint saved as nnsk.iter294
    DEEPTB INFO    Epoch 294 summary:	train_loss: 0.010606	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep294
    DEEPTB INFO    iteration:295	train_loss: 0.010581  (0.010639)	lr: 0.007452
    DEEPTB INFO    checkpoint saved as nnsk.iter295
    DEEPTB INFO    Epoch 295 summary:	train_loss: 0.010581	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep295
    DEEPTB INFO    iteration:296	train_loss: 0.010557  (0.010614)	lr: 0.007444
    DEEPTB INFO    checkpoint saved as nnsk.iter296
    DEEPTB INFO    Epoch 296 summary:	train_loss: 0.010557	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep296
    DEEPTB INFO    iteration:297	train_loss: 0.010532  (0.010590)	lr: 0.007437
    DEEPTB INFO    checkpoint saved as nnsk.iter297
    DEEPTB INFO    Epoch 297 summary:	train_loss: 0.010532	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep297
    DEEPTB INFO    iteration:298	train_loss: 0.010508  (0.010565)	lr: 0.007429
    DEEPTB INFO    checkpoint saved as nnsk.iter298
    DEEPTB INFO    Epoch 298 summary:	train_loss: 0.010508	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep298
    DEEPTB INFO    iteration:299	train_loss: 0.010484  (0.010541)	lr: 0.007422
    DEEPTB INFO    checkpoint saved as nnsk.iter299
    DEEPTB INFO    Epoch 299 summary:	train_loss: 0.010484	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep299
    DEEPTB INFO    iteration:300	train_loss: 0.010459  (0.010516)	lr: 0.007414
    DEEPTB INFO    checkpoint saved as nnsk.iter300
    DEEPTB INFO    Epoch 300 summary:	train_loss: 0.010459	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep300
    DEEPTB INFO    iteration:301	train_loss: 0.010435  (0.010492)	lr: 0.007407
    DEEPTB INFO    checkpoint saved as nnsk.iter301
    DEEPTB INFO    Epoch 301 summary:	train_loss: 0.010435	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep301
    DEEPTB INFO    iteration:302	train_loss: 0.010412  (0.010468)	lr: 0.0074  
    DEEPTB INFO    checkpoint saved as nnsk.iter302
    DEEPTB INFO    Epoch 302 summary:	train_loss: 0.010412	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep302
    DEEPTB INFO    iteration:303	train_loss: 0.010388  (0.010444)	lr: 0.007392
    DEEPTB INFO    checkpoint saved as nnsk.iter303
    DEEPTB INFO    Epoch 303 summary:	train_loss: 0.010388	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep303
    DEEPTB INFO    iteration:304	train_loss: 0.010364  (0.010420)	lr: 0.007385
    DEEPTB INFO    checkpoint saved as nnsk.iter304
    DEEPTB INFO    Epoch 304 summary:	train_loss: 0.010364	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep304
    DEEPTB INFO    iteration:305	train_loss: 0.010340  (0.010396)	lr: 0.007377
    DEEPTB INFO    checkpoint saved as nnsk.iter305
    DEEPTB INFO    Epoch 305 summary:	train_loss: 0.010340	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep305
    DEEPTB INFO    iteration:306	train_loss: 0.010316  (0.010372)	lr: 0.00737 
    DEEPTB INFO    checkpoint saved as nnsk.iter306
    DEEPTB INFO    Epoch 306 summary:	train_loss: 0.010316	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep306
    DEEPTB INFO    iteration:307	train_loss: 0.010292  (0.010348)	lr: 0.007363
    DEEPTB INFO    checkpoint saved as nnsk.iter307
    DEEPTB INFO    Epoch 307 summary:	train_loss: 0.010292	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep307
    DEEPTB INFO    iteration:308	train_loss: 0.010270  (0.010325)	lr: 0.007355
    DEEPTB INFO    checkpoint saved as nnsk.iter308
    DEEPTB INFO    Epoch 308 summary:	train_loss: 0.010270	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep308
    DEEPTB INFO    iteration:309	train_loss: 0.010245  (0.010301)	lr: 0.007348
    DEEPTB INFO    checkpoint saved as nnsk.iter309
    DEEPTB INFO    Epoch 309 summary:	train_loss: 0.010245	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep309
    DEEPTB INFO    iteration:310	train_loss: 0.010222  (0.010277)	lr: 0.007341
    DEEPTB INFO    checkpoint saved as nnsk.iter310
    DEEPTB INFO    Epoch 310 summary:	train_loss: 0.010222	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep310
    DEEPTB INFO    iteration:311	train_loss: 0.010199  (0.010253)	lr: 0.007333
    DEEPTB INFO    checkpoint saved as nnsk.iter311
    DEEPTB INFO    Epoch 311 summary:	train_loss: 0.010199	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep311
    DEEPTB INFO    iteration:312	train_loss: 0.010175  (0.010230)	lr: 0.007326
    DEEPTB INFO    checkpoint saved as nnsk.iter312
    DEEPTB INFO    Epoch 312 summary:	train_loss: 0.010175	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep312
    DEEPTB INFO    iteration:313	train_loss: 0.010152  (0.010207)	lr: 0.007319
    DEEPTB INFO    checkpoint saved as nnsk.iter313
    DEEPTB INFO    Epoch 313 summary:	train_loss: 0.010152	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep313
    DEEPTB INFO    iteration:314	train_loss: 0.010128  (0.010183)	lr: 0.007311
    DEEPTB INFO    checkpoint saved as nnsk.iter314
    DEEPTB INFO    Epoch 314 summary:	train_loss: 0.010128	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep314
    DEEPTB INFO    iteration:315	train_loss: 0.010106  (0.010160)	lr: 0.007304
    DEEPTB INFO    checkpoint saved as nnsk.iter315
    DEEPTB INFO    Epoch 315 summary:	train_loss: 0.010106	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep315
    DEEPTB INFO    iteration:316	train_loss: 0.010083  (0.010137)	lr: 0.007297
    DEEPTB INFO    checkpoint saved as nnsk.iter316
    DEEPTB INFO    Epoch 316 summary:	train_loss: 0.010083	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep316
    DEEPTB INFO    iteration:317	train_loss: 0.010060  (0.010114)	lr: 0.007289
    DEEPTB INFO    checkpoint saved as nnsk.iter317
    DEEPTB INFO    Epoch 317 summary:	train_loss: 0.010060	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep317
    DEEPTB INFO    iteration:318	train_loss: 0.010037  (0.010091)	lr: 0.007282
    DEEPTB INFO    checkpoint saved as nnsk.iter318
    DEEPTB INFO    Epoch 318 summary:	train_loss: 0.010037	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep318
    DEEPTB INFO    iteration:319	train_loss: 0.010015  (0.010068)	lr: 0.007275
    DEEPTB INFO    checkpoint saved as nnsk.iter319
    DEEPTB INFO    Epoch 319 summary:	train_loss: 0.010015	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep319
    DEEPTB INFO    iteration:320	train_loss: 0.009992  (0.010045)	lr: 0.007268
    DEEPTB INFO    checkpoint saved as nnsk.iter320
    DEEPTB INFO    Epoch 320 summary:	train_loss: 0.009992	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep320
    DEEPTB INFO    iteration:321	train_loss: 0.009969  (0.010022)	lr: 0.00726 
    DEEPTB INFO    checkpoint saved as nnsk.iter321
    DEEPTB INFO    Epoch 321 summary:	train_loss: 0.009969	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep321
    DEEPTB INFO    iteration:322	train_loss: 0.009946  (0.010000)	lr: 0.007253
    DEEPTB INFO    checkpoint saved as nnsk.iter322
    DEEPTB INFO    Epoch 322 summary:	train_loss: 0.009946	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep322
    DEEPTB INFO    iteration:323	train_loss: 0.009924  (0.009977)	lr: 0.007246
    DEEPTB INFO    checkpoint saved as nnsk.iter323
    DEEPTB INFO    Epoch 323 summary:	train_loss: 0.009924	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep323
    DEEPTB INFO    iteration:324	train_loss: 0.009901  (0.009954)	lr: 0.007239
    DEEPTB INFO    checkpoint saved as nnsk.iter324
    DEEPTB INFO    Epoch 324 summary:	train_loss: 0.009901	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep324
    DEEPTB INFO    iteration:325	train_loss: 0.009878  (0.009931)	lr: 0.007231
    DEEPTB INFO    checkpoint saved as nnsk.iter325
    DEEPTB INFO    Epoch 325 summary:	train_loss: 0.009878	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep325
    DEEPTB INFO    iteration:326	train_loss: 0.009856  (0.009909)	lr: 0.007224
    DEEPTB INFO    checkpoint saved as nnsk.iter326
    DEEPTB INFO    Epoch 326 summary:	train_loss: 0.009856	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep326
    DEEPTB INFO    iteration:327	train_loss: 0.009834  (0.009886)	lr: 0.007217
    DEEPTB INFO    checkpoint saved as nnsk.iter327
    DEEPTB INFO    Epoch 327 summary:	train_loss: 0.009834	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep327
    DEEPTB INFO    iteration:328	train_loss: 0.009811  (0.009864)	lr: 0.00721 
    DEEPTB INFO    checkpoint saved as nnsk.iter328
    DEEPTB INFO    Epoch 328 summary:	train_loss: 0.009811	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep328
    DEEPTB INFO    iteration:329	train_loss: 0.009789  (0.009841)	lr: 0.007202
    DEEPTB INFO    checkpoint saved as nnsk.iter329
    DEEPTB INFO    Epoch 329 summary:	train_loss: 0.009789	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep329
    DEEPTB INFO    iteration:330	train_loss: 0.009767  (0.009819)	lr: 0.007195
    DEEPTB INFO    checkpoint saved as nnsk.iter330
    DEEPTB INFO    Epoch 330 summary:	train_loss: 0.009767	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep330
    DEEPTB INFO    iteration:331	train_loss: 0.009744  (0.009797)	lr: 0.007188
    DEEPTB INFO    checkpoint saved as nnsk.iter331
    DEEPTB INFO    Epoch 331 summary:	train_loss: 0.009744	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep331
    DEEPTB INFO    iteration:332	train_loss: 0.009722  (0.009774)	lr: 0.007181
    DEEPTB INFO    checkpoint saved as nnsk.iter332
    DEEPTB INFO    Epoch 332 summary:	train_loss: 0.009722	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep332
    DEEPTB INFO    iteration:333	train_loss: 0.009700  (0.009752)	lr: 0.007174
    DEEPTB INFO    checkpoint saved as nnsk.iter333
    DEEPTB INFO    Epoch 333 summary:	train_loss: 0.009700	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep333
    DEEPTB INFO    iteration:334	train_loss: 0.009678  (0.009730)	lr: 0.007167
    DEEPTB INFO    checkpoint saved as nnsk.iter334
    DEEPTB INFO    Epoch 334 summary:	train_loss: 0.009678	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep334
    DEEPTB INFO    iteration:335	train_loss: 0.009656  (0.009708)	lr: 0.007159
    DEEPTB INFO    checkpoint saved as nnsk.iter335
    DEEPTB INFO    Epoch 335 summary:	train_loss: 0.009656	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep335
    DEEPTB INFO    iteration:336	train_loss: 0.009634  (0.009686)	lr: 0.007152
    DEEPTB INFO    checkpoint saved as nnsk.iter336
    DEEPTB INFO    Epoch 336 summary:	train_loss: 0.009634	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep336
    DEEPTB INFO    iteration:337	train_loss: 0.009612  (0.009664)	lr: 0.007145
    DEEPTB INFO    checkpoint saved as nnsk.iter337
    DEEPTB INFO    Epoch 337 summary:	train_loss: 0.009612	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep337
    DEEPTB INFO    iteration:338	train_loss: 0.009590  (0.009642)	lr: 0.007138
    DEEPTB INFO    checkpoint saved as nnsk.iter338
    DEEPTB INFO    Epoch 338 summary:	train_loss: 0.009590	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep338
    DEEPTB INFO    iteration:339	train_loss: 0.009569  (0.009620)	lr: 0.007131
    DEEPTB INFO    checkpoint saved as nnsk.iter339
    DEEPTB INFO    Epoch 339 summary:	train_loss: 0.009569	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep339
    DEEPTB INFO    iteration:340	train_loss: 0.009547  (0.009598)	lr: 0.007124
    DEEPTB INFO    checkpoint saved as nnsk.iter340
    DEEPTB INFO    Epoch 340 summary:	train_loss: 0.009547	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep340
    DEEPTB INFO    iteration:341	train_loss: 0.009525  (0.009576)	lr: 0.007116
    DEEPTB INFO    checkpoint saved as nnsk.iter341
    DEEPTB INFO    Epoch 341 summary:	train_loss: 0.009525	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep341
    DEEPTB INFO    iteration:342	train_loss: 0.009503  (0.009554)	lr: 0.007109
    DEEPTB INFO    checkpoint saved as nnsk.iter342
    DEEPTB INFO    Epoch 342 summary:	train_loss: 0.009503	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep342
    DEEPTB INFO    iteration:343	train_loss: 0.009481  (0.009532)	lr: 0.007102
    DEEPTB INFO    checkpoint saved as nnsk.iter343
    DEEPTB INFO    Epoch 343 summary:	train_loss: 0.009481	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep343
    DEEPTB INFO    iteration:344	train_loss: 0.009460  (0.009511)	lr: 0.007095
    DEEPTB INFO    checkpoint saved as nnsk.iter344
    DEEPTB INFO    Epoch 344 summary:	train_loss: 0.009460	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep344
    DEEPTB INFO    iteration:345	train_loss: 0.009438  (0.009489)	lr: 0.007088
    DEEPTB INFO    checkpoint saved as nnsk.iter345
    DEEPTB INFO    Epoch 345 summary:	train_loss: 0.009438	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep345
    DEEPTB INFO    iteration:346	train_loss: 0.009416  (0.009467)	lr: 0.007081
    DEEPTB INFO    checkpoint saved as nnsk.iter346
    DEEPTB INFO    Epoch 346 summary:	train_loss: 0.009416	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep346
    DEEPTB INFO    iteration:347	train_loss: 0.009395  (0.009445)	lr: 0.007074
    DEEPTB INFO    checkpoint saved as nnsk.iter347
    DEEPTB INFO    Epoch 347 summary:	train_loss: 0.009395	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep347
    DEEPTB INFO    iteration:348	train_loss: 0.009374  (0.009424)	lr: 0.007067
    DEEPTB INFO    checkpoint saved as nnsk.iter348
    DEEPTB INFO    Epoch 348 summary:	train_loss: 0.009374	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep348
    DEEPTB INFO    iteration:349	train_loss: 0.009352  (0.009402)	lr: 0.00706 
    DEEPTB INFO    checkpoint saved as nnsk.iter349
    DEEPTB INFO    Epoch 349 summary:	train_loss: 0.009352	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep349
    DEEPTB INFO    iteration:350	train_loss: 0.009331  (0.009381)	lr: 0.007053
    DEEPTB INFO    checkpoint saved as nnsk.iter350
    DEEPTB INFO    Epoch 350 summary:	train_loss: 0.009331	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep350
    DEEPTB INFO    iteration:351	train_loss: 0.009309  (0.009359)	lr: 0.007046
    DEEPTB INFO    checkpoint saved as nnsk.iter351
    DEEPTB INFO    Epoch 351 summary:	train_loss: 0.009309	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep351
    DEEPTB INFO    iteration:352	train_loss: 0.009288  (0.009338)	lr: 0.007039
    DEEPTB INFO    checkpoint saved as nnsk.iter352
    DEEPTB INFO    Epoch 352 summary:	train_loss: 0.009288	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep352
    DEEPTB INFO    iteration:353	train_loss: 0.009266  (0.009316)	lr: 0.007032
    DEEPTB INFO    checkpoint saved as nnsk.iter353
    DEEPTB INFO    Epoch 353 summary:	train_loss: 0.009266	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep353
    DEEPTB INFO    iteration:354	train_loss: 0.009245  (0.009295)	lr: 0.007025
    DEEPTB INFO    checkpoint saved as nnsk.iter354
    DEEPTB INFO    Epoch 354 summary:	train_loss: 0.009245	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep354
    DEEPTB INFO    iteration:355	train_loss: 0.009224  (0.009274)	lr: 0.007018
    DEEPTB INFO    checkpoint saved as nnsk.iter355
    DEEPTB INFO    Epoch 355 summary:	train_loss: 0.009224	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep355
    DEEPTB INFO    iteration:356	train_loss: 0.009202  (0.009252)	lr: 0.00701 
    DEEPTB INFO    checkpoint saved as nnsk.iter356
    DEEPTB INFO    Epoch 356 summary:	train_loss: 0.009202	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep356
    DEEPTB INFO    iteration:357	train_loss: 0.009182  (0.009231)	lr: 0.007003
    DEEPTB INFO    checkpoint saved as nnsk.iter357
    DEEPTB INFO    Epoch 357 summary:	train_loss: 0.009182	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep357
    DEEPTB INFO    iteration:358	train_loss: 0.009160  (0.009210)	lr: 0.006996
    DEEPTB INFO    checkpoint saved as nnsk.iter358
    DEEPTB INFO    Epoch 358 summary:	train_loss: 0.009160	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep358
    DEEPTB INFO    iteration:359	train_loss: 0.009139  (0.009189)	lr: 0.006989
    DEEPTB INFO    checkpoint saved as nnsk.iter359
    DEEPTB INFO    Epoch 359 summary:	train_loss: 0.009139	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep359
    DEEPTB INFO    iteration:360	train_loss: 0.009118  (0.009167)	lr: 0.006982
    DEEPTB INFO    checkpoint saved as nnsk.iter360
    DEEPTB INFO    Epoch 360 summary:	train_loss: 0.009118	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep360
    DEEPTB INFO    iteration:361	train_loss: 0.009097  (0.009146)	lr: 0.006976
    DEEPTB INFO    checkpoint saved as nnsk.iter361
    DEEPTB INFO    Epoch 361 summary:	train_loss: 0.009097	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep361
    DEEPTB INFO    iteration:362	train_loss: 0.009076  (0.009125)	lr: 0.006969
    DEEPTB INFO    checkpoint saved as nnsk.iter362
    DEEPTB INFO    Epoch 362 summary:	train_loss: 0.009076	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep362
    DEEPTB INFO    iteration:363	train_loss: 0.009055  (0.009104)	lr: 0.006962
    DEEPTB INFO    checkpoint saved as nnsk.iter363
    DEEPTB INFO    Epoch 363 summary:	train_loss: 0.009055	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep363
    DEEPTB INFO    iteration:364	train_loss: 0.009034  (0.009083)	lr: 0.006955
    DEEPTB INFO    checkpoint saved as nnsk.iter364
    DEEPTB INFO    Epoch 364 summary:	train_loss: 0.009034	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep364
    DEEPTB INFO    iteration:365	train_loss: 0.009013  (0.009062)	lr: 0.006948
    DEEPTB INFO    checkpoint saved as nnsk.iter365
    DEEPTB INFO    Epoch 365 summary:	train_loss: 0.009013	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep365
    DEEPTB INFO    iteration:366	train_loss: 0.008991  (0.009041)	lr: 0.006941
    DEEPTB INFO    checkpoint saved as nnsk.iter366
    DEEPTB INFO    Epoch 366 summary:	train_loss: 0.008991	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep366
    DEEPTB INFO    iteration:367	train_loss: 0.008970  (0.009020)	lr: 0.006934
    DEEPTB INFO    checkpoint saved as nnsk.iter367
    DEEPTB INFO    Epoch 367 summary:	train_loss: 0.008970	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep367
    DEEPTB INFO    iteration:368	train_loss: 0.008949  (0.008999)	lr: 0.006927
    DEEPTB INFO    checkpoint saved as nnsk.iter368
    DEEPTB INFO    Epoch 368 summary:	train_loss: 0.008949	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep368
    DEEPTB INFO    iteration:369	train_loss: 0.008928  (0.008978)	lr: 0.00692 
    DEEPTB INFO    checkpoint saved as nnsk.iter369
    DEEPTB INFO    Epoch 369 summary:	train_loss: 0.008928	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep369
    DEEPTB INFO    iteration:370	train_loss: 0.008907  (0.008956)	lr: 0.006913
    DEEPTB INFO    checkpoint saved as nnsk.iter370
    DEEPTB INFO    Epoch 370 summary:	train_loss: 0.008907	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep370
    DEEPTB INFO    iteration:371	train_loss: 0.008886  (0.008935)	lr: 0.006906
    DEEPTB INFO    checkpoint saved as nnsk.iter371
    DEEPTB INFO    Epoch 371 summary:	train_loss: 0.008886	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep371
    DEEPTB INFO    iteration:372	train_loss: 0.008865  (0.008914)	lr: 0.006899
    DEEPTB INFO    checkpoint saved as nnsk.iter372
    DEEPTB INFO    Epoch 372 summary:	train_loss: 0.008865	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep372
    DEEPTB INFO    iteration:373	train_loss: 0.008845  (0.008893)	lr: 0.006892
    DEEPTB INFO    checkpoint saved as nnsk.iter373
    DEEPTB INFO    Epoch 373 summary:	train_loss: 0.008845	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep373
    DEEPTB INFO    iteration:374	train_loss: 0.008824  (0.008873)	lr: 0.006885
    DEEPTB INFO    checkpoint saved as nnsk.iter374
    DEEPTB INFO    Epoch 374 summary:	train_loss: 0.008824	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep374
    DEEPTB INFO    iteration:375	train_loss: 0.008803  (0.008852)	lr: 0.006878
    DEEPTB INFO    checkpoint saved as nnsk.iter375
    DEEPTB INFO    Epoch 375 summary:	train_loss: 0.008803	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep375
    DEEPTB INFO    iteration:376	train_loss: 0.008782  (0.008831)	lr: 0.006872
    DEEPTB INFO    checkpoint saved as nnsk.iter376
    DEEPTB INFO    Epoch 376 summary:	train_loss: 0.008782	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep376
    DEEPTB INFO    iteration:377	train_loss: 0.008761  (0.008810)	lr: 0.006865
    DEEPTB INFO    checkpoint saved as nnsk.iter377
    DEEPTB INFO    Epoch 377 summary:	train_loss: 0.008761	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep377
    DEEPTB INFO    iteration:378	train_loss: 0.008741  (0.008789)	lr: 0.006858
    DEEPTB INFO    checkpoint saved as nnsk.iter378
    DEEPTB INFO    Epoch 378 summary:	train_loss: 0.008741	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep378
    DEEPTB INFO    iteration:379	train_loss: 0.008720  (0.008768)	lr: 0.006851
    DEEPTB INFO    checkpoint saved as nnsk.iter379
    DEEPTB INFO    Epoch 379 summary:	train_loss: 0.008720	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep379
    DEEPTB INFO    iteration:380	train_loss: 0.008699  (0.008748)	lr: 0.006844
    DEEPTB INFO    checkpoint saved as nnsk.iter380
    DEEPTB INFO    Epoch 380 summary:	train_loss: 0.008699	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep380
    DEEPTB INFO    iteration:381	train_loss: 0.008678  (0.008727)	lr: 0.006837
    DEEPTB INFO    checkpoint saved as nnsk.iter381
    DEEPTB INFO    Epoch 381 summary:	train_loss: 0.008678	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep381
    DEEPTB INFO    iteration:382	train_loss: 0.008658  (0.008706)	lr: 0.00683 
    DEEPTB INFO    checkpoint saved as nnsk.iter382
    DEEPTB INFO    Epoch 382 summary:	train_loss: 0.008658	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep382
    DEEPTB INFO    iteration:383	train_loss: 0.008637  (0.008685)	lr: 0.006824
    DEEPTB INFO    checkpoint saved as nnsk.iter383
    DEEPTB INFO    Epoch 383 summary:	train_loss: 0.008637	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep383
    DEEPTB INFO    iteration:384	train_loss: 0.008617  (0.008665)	lr: 0.006817
    DEEPTB INFO    checkpoint saved as nnsk.iter384
    DEEPTB INFO    Epoch 384 summary:	train_loss: 0.008617	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep384
    DEEPTB INFO    iteration:385	train_loss: 0.008596  (0.008644)	lr: 0.00681 
    DEEPTB INFO    checkpoint saved as nnsk.iter385
    DEEPTB INFO    Epoch 385 summary:	train_loss: 0.008596	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep385
    DEEPTB INFO    iteration:386	train_loss: 0.008575  (0.008623)	lr: 0.006803
    DEEPTB INFO    checkpoint saved as nnsk.iter386
    DEEPTB INFO    Epoch 386 summary:	train_loss: 0.008575	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep386
    DEEPTB INFO    iteration:387	train_loss: 0.008555  (0.008603)	lr: 0.006796
    DEEPTB INFO    checkpoint saved as nnsk.iter387
    DEEPTB INFO    Epoch 387 summary:	train_loss: 0.008555	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep387
    DEEPTB INFO    iteration:388	train_loss: 0.008534  (0.008582)	lr: 0.00679 
    DEEPTB INFO    checkpoint saved as nnsk.iter388
    DEEPTB INFO    Epoch 388 summary:	train_loss: 0.008534	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep388
    DEEPTB INFO    iteration:389	train_loss: 0.008514  (0.008562)	lr: 0.006783
    DEEPTB INFO    checkpoint saved as nnsk.iter389
    DEEPTB INFO    Epoch 389 summary:	train_loss: 0.008514	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep389
    DEEPTB INFO    iteration:390	train_loss: 0.008493  (0.008541)	lr: 0.006776
    DEEPTB INFO    checkpoint saved as nnsk.iter390
    DEEPTB INFO    Epoch 390 summary:	train_loss: 0.008493	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep390
    DEEPTB INFO    iteration:391	train_loss: 0.008473  (0.008521)	lr: 0.006769
    DEEPTB INFO    checkpoint saved as nnsk.iter391
    DEEPTB INFO    Epoch 391 summary:	train_loss: 0.008473	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep391
    DEEPTB INFO    iteration:392	train_loss: 0.008452  (0.008500)	lr: 0.006762
    DEEPTB INFO    checkpoint saved as nnsk.iter392
    DEEPTB INFO    Epoch 392 summary:	train_loss: 0.008452	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep392
    DEEPTB INFO    iteration:393	train_loss: 0.008432  (0.008480)	lr: 0.006756
    DEEPTB INFO    checkpoint saved as nnsk.iter393
    DEEPTB INFO    Epoch 393 summary:	train_loss: 0.008432	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep393
    DEEPTB INFO    iteration:394	train_loss: 0.008412  (0.008459)	lr: 0.006749
    DEEPTB INFO    checkpoint saved as nnsk.iter394
    DEEPTB INFO    Epoch 394 summary:	train_loss: 0.008412	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep394
    DEEPTB INFO    iteration:395	train_loss: 0.008391  (0.008439)	lr: 0.006742
    DEEPTB INFO    checkpoint saved as nnsk.iter395
    DEEPTB INFO    Epoch 395 summary:	train_loss: 0.008391	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep395
    DEEPTB INFO    iteration:396	train_loss: 0.008372  (0.008419)	lr: 0.006735
    DEEPTB INFO    checkpoint saved as nnsk.iter396
    DEEPTB INFO    Epoch 396 summary:	train_loss: 0.008372	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep396
    DEEPTB INFO    iteration:397	train_loss: 0.008352  (0.008399)	lr: 0.006729
    DEEPTB INFO    checkpoint saved as nnsk.iter397
    DEEPTB INFO    Epoch 397 summary:	train_loss: 0.008352	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep397
    DEEPTB INFO    iteration:398	train_loss: 0.008332  (0.008379)	lr: 0.006722
    DEEPTB INFO    checkpoint saved as nnsk.iter398
    DEEPTB INFO    Epoch 398 summary:	train_loss: 0.008332	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep398
    DEEPTB INFO    iteration:399	train_loss: 0.008313  (0.008359)	lr: 0.006715
    DEEPTB INFO    checkpoint saved as nnsk.iter399
    DEEPTB INFO    Epoch 399 summary:	train_loss: 0.008313	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep399
    DEEPTB INFO    iteration:400	train_loss: 0.008293  (0.008339)	lr: 0.006709
    DEEPTB INFO    checkpoint saved as nnsk.iter400
    DEEPTB INFO    Epoch 400 summary:	train_loss: 0.008293	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep400
    DEEPTB INFO    iteration:401	train_loss: 0.008273  (0.008319)	lr: 0.006702
    DEEPTB INFO    checkpoint saved as nnsk.iter401
    DEEPTB INFO    Epoch 401 summary:	train_loss: 0.008273	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep401
    DEEPTB INFO    iteration:402	train_loss: 0.008253  (0.008299)	lr: 0.006695
    DEEPTB INFO    checkpoint saved as nnsk.iter402
    DEEPTB INFO    Epoch 402 summary:	train_loss: 0.008253	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep402
    DEEPTB INFO    iteration:403	train_loss: 0.008233  (0.008279)	lr: 0.006688
    DEEPTB INFO    checkpoint saved as nnsk.iter403
    DEEPTB INFO    Epoch 403 summary:	train_loss: 0.008233	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep403
    DEEPTB INFO    iteration:404	train_loss: 0.008213  (0.008260)	lr: 0.006682
    DEEPTB INFO    checkpoint saved as nnsk.iter404
    DEEPTB INFO    Epoch 404 summary:	train_loss: 0.008213	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep404
    DEEPTB INFO    iteration:405	train_loss: 0.008193  (0.008240)	lr: 0.006675
    DEEPTB INFO    checkpoint saved as nnsk.iter405
    DEEPTB INFO    Epoch 405 summary:	train_loss: 0.008193	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep405
    DEEPTB INFO    iteration:406	train_loss: 0.008173  (0.008220)	lr: 0.006668
    DEEPTB INFO    checkpoint saved as nnsk.iter406
    DEEPTB INFO    Epoch 406 summary:	train_loss: 0.008173	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep406
    DEEPTB INFO    iteration:407	train_loss: 0.008153  (0.008200)	lr: 0.006662
    DEEPTB INFO    checkpoint saved as nnsk.iter407
    DEEPTB INFO    Epoch 407 summary:	train_loss: 0.008153	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep407
    DEEPTB INFO    iteration:408	train_loss: 0.008133  (0.008180)	lr: 0.006655
    DEEPTB INFO    checkpoint saved as nnsk.iter408
    DEEPTB INFO    Epoch 408 summary:	train_loss: 0.008133	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep408
    DEEPTB INFO    iteration:409	train_loss: 0.008114  (0.008160)	lr: 0.006648
    DEEPTB INFO    checkpoint saved as nnsk.iter409
    DEEPTB INFO    Epoch 409 summary:	train_loss: 0.008114	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep409
    DEEPTB INFO    iteration:410	train_loss: 0.008094  (0.008140)	lr: 0.006642
    DEEPTB INFO    checkpoint saved as nnsk.iter410
    DEEPTB INFO    Epoch 410 summary:	train_loss: 0.008094	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep410
    DEEPTB INFO    iteration:411	train_loss: 0.008074  (0.008120)	lr: 0.006635
    DEEPTB INFO    checkpoint saved as nnsk.iter411
    DEEPTB INFO    Epoch 411 summary:	train_loss: 0.008074	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep411
    DEEPTB INFO    iteration:412	train_loss: 0.008055  (0.008101)	lr: 0.006629
    DEEPTB INFO    checkpoint saved as nnsk.iter412
    DEEPTB INFO    Epoch 412 summary:	train_loss: 0.008055	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep412
    DEEPTB INFO    iteration:413	train_loss: 0.008035  (0.008081)	lr: 0.006622
    DEEPTB INFO    checkpoint saved as nnsk.iter413
    DEEPTB INFO    Epoch 413 summary:	train_loss: 0.008035	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep413
    DEEPTB INFO    iteration:414	train_loss: 0.008015  (0.008061)	lr: 0.006615
    DEEPTB INFO    checkpoint saved as nnsk.iter414
    DEEPTB INFO    Epoch 414 summary:	train_loss: 0.008015	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep414
    DEEPTB INFO    iteration:415	train_loss: 0.007996  (0.008042)	lr: 0.006609
    DEEPTB INFO    checkpoint saved as nnsk.iter415
    DEEPTB INFO    Epoch 415 summary:	train_loss: 0.007996	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep415
    DEEPTB INFO    iteration:416	train_loss: 0.007976  (0.008022)	lr: 0.006602
    DEEPTB INFO    checkpoint saved as nnsk.iter416
    DEEPTB INFO    Epoch 416 summary:	train_loss: 0.007976	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep416
    DEEPTB INFO    iteration:417	train_loss: 0.007956  (0.008002)	lr: 0.006595
    DEEPTB INFO    checkpoint saved as nnsk.iter417
    DEEPTB INFO    Epoch 417 summary:	train_loss: 0.007956	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep417
    DEEPTB INFO    iteration:418	train_loss: 0.007937  (0.007982)	lr: 0.006589
    DEEPTB INFO    checkpoint saved as nnsk.iter418
    DEEPTB INFO    Epoch 418 summary:	train_loss: 0.007937	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep418
    DEEPTB INFO    iteration:419	train_loss: 0.007917  (0.007963)	lr: 0.006582
    DEEPTB INFO    checkpoint saved as nnsk.iter419
    DEEPTB INFO    Epoch 419 summary:	train_loss: 0.007917	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep419
    DEEPTB INFO    iteration:420	train_loss: 0.007898  (0.007943)	lr: 0.006576
    DEEPTB INFO    checkpoint saved as nnsk.iter420
    DEEPTB INFO    Epoch 420 summary:	train_loss: 0.007898	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep420
    DEEPTB INFO    iteration:421	train_loss: 0.007878  (0.007924)	lr: 0.006569
    DEEPTB INFO    checkpoint saved as nnsk.iter421
    DEEPTB INFO    Epoch 421 summary:	train_loss: 0.007878	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep421
    DEEPTB INFO    iteration:422	train_loss: 0.007859  (0.007904)	lr: 0.006563
    DEEPTB INFO    checkpoint saved as nnsk.iter422
    DEEPTB INFO    Epoch 422 summary:	train_loss: 0.007859	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep422
    DEEPTB INFO    iteration:423	train_loss: 0.007839  (0.007885)	lr: 0.006556
    DEEPTB INFO    checkpoint saved as nnsk.iter423
    DEEPTB INFO    Epoch 423 summary:	train_loss: 0.007839	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep423
    DEEPTB INFO    iteration:424	train_loss: 0.007819  (0.007865)	lr: 0.006549
    DEEPTB INFO    checkpoint saved as nnsk.iter424
    DEEPTB INFO    Epoch 424 summary:	train_loss: 0.007819	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep424
    DEEPTB INFO    iteration:425	train_loss: 0.007800  (0.007846)	lr: 0.006543
    DEEPTB INFO    checkpoint saved as nnsk.iter425
    DEEPTB INFO    Epoch 425 summary:	train_loss: 0.007800	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep425
    DEEPTB INFO    iteration:426	train_loss: 0.007781  (0.007826)	lr: 0.006536
    DEEPTB INFO    checkpoint saved as nnsk.iter426
    DEEPTB INFO    Epoch 426 summary:	train_loss: 0.007781	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep426
    DEEPTB INFO    iteration:427	train_loss: 0.007762  (0.007807)	lr: 0.00653 
    DEEPTB INFO    checkpoint saved as nnsk.iter427
    DEEPTB INFO    Epoch 427 summary:	train_loss: 0.007762	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep427
    DEEPTB INFO    iteration:428	train_loss: 0.007742  (0.007787)	lr: 0.006523
    DEEPTB INFO    checkpoint saved as nnsk.iter428
    DEEPTB INFO    Epoch 428 summary:	train_loss: 0.007742	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep428
    DEEPTB INFO    iteration:429	train_loss: 0.007723  (0.007768)	lr: 0.006517
    DEEPTB INFO    checkpoint saved as nnsk.iter429
    DEEPTB INFO    Epoch 429 summary:	train_loss: 0.007723	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep429
    DEEPTB INFO    iteration:430	train_loss: 0.007703  (0.007749)	lr: 0.00651 
    DEEPTB INFO    checkpoint saved as nnsk.iter430
    DEEPTB INFO    Epoch 430 summary:	train_loss: 0.007703	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep430
    DEEPTB INFO    iteration:431	train_loss: 0.007684  (0.007729)	lr: 0.006504
    DEEPTB INFO    checkpoint saved as nnsk.iter431
    DEEPTB INFO    Epoch 431 summary:	train_loss: 0.007684	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep431
    DEEPTB INFO    iteration:432	train_loss: 0.007665  (0.007710)	lr: 0.006497
    DEEPTB INFO    checkpoint saved as nnsk.iter432
    DEEPTB INFO    Epoch 432 summary:	train_loss: 0.007665	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep432
    DEEPTB INFO    iteration:433	train_loss: 0.007646  (0.007691)	lr: 0.006491
    DEEPTB INFO    checkpoint saved as nnsk.iter433
    DEEPTB INFO    Epoch 433 summary:	train_loss: 0.007646	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep433
    DEEPTB INFO    iteration:434	train_loss: 0.007626  (0.007671)	lr: 0.006484
    DEEPTB INFO    checkpoint saved as nnsk.iter434
    DEEPTB INFO    Epoch 434 summary:	train_loss: 0.007626	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep434
    DEEPTB INFO    iteration:435	train_loss: 0.007607  (0.007652)	lr: 0.006478
    DEEPTB INFO    checkpoint saved as nnsk.iter435
    DEEPTB INFO    Epoch 435 summary:	train_loss: 0.007607	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep435
    DEEPTB INFO    iteration:436	train_loss: 0.007588  (0.007633)	lr: 0.006471
    DEEPTB INFO    checkpoint saved as nnsk.iter436
    DEEPTB INFO    Epoch 436 summary:	train_loss: 0.007588	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep436
    DEEPTB INFO    iteration:437	train_loss: 0.007569  (0.007614)	lr: 0.006465
    DEEPTB INFO    checkpoint saved as nnsk.iter437
    DEEPTB INFO    Epoch 437 summary:	train_loss: 0.007569	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep437
    DEEPTB INFO    iteration:438	train_loss: 0.007550  (0.007594)	lr: 0.006458
    DEEPTB INFO    checkpoint saved as nnsk.iter438
    DEEPTB INFO    Epoch 438 summary:	train_loss: 0.007550	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep438
    DEEPTB INFO    iteration:439	train_loss: 0.007531  (0.007575)	lr: 0.006452
    DEEPTB INFO    checkpoint saved as nnsk.iter439
    DEEPTB INFO    Epoch 439 summary:	train_loss: 0.007531	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep439
    DEEPTB INFO    iteration:440	train_loss: 0.007512  (0.007556)	lr: 0.006445
    DEEPTB INFO    checkpoint saved as nnsk.iter440
    DEEPTB INFO    Epoch 440 summary:	train_loss: 0.007512	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep440
    DEEPTB INFO    iteration:441	train_loss: 0.007493  (0.007537)	lr: 0.006439
    DEEPTB INFO    checkpoint saved as nnsk.iter441
    DEEPTB INFO    Epoch 441 summary:	train_loss: 0.007493	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep441
    DEEPTB INFO    iteration:442	train_loss: 0.007474  (0.007518)	lr: 0.006433
    DEEPTB INFO    checkpoint saved as nnsk.iter442
    DEEPTB INFO    Epoch 442 summary:	train_loss: 0.007474	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep442
    DEEPTB INFO    iteration:443	train_loss: 0.007455  (0.007499)	lr: 0.006426
    DEEPTB INFO    checkpoint saved as nnsk.iter443
    DEEPTB INFO    Epoch 443 summary:	train_loss: 0.007455	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep443
    DEEPTB INFO    iteration:444	train_loss: 0.007436  (0.007480)	lr: 0.00642 
    DEEPTB INFO    checkpoint saved as nnsk.iter444
    DEEPTB INFO    Epoch 444 summary:	train_loss: 0.007436	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep444
    DEEPTB INFO    iteration:445	train_loss: 0.007417  (0.007461)	lr: 0.006413
    DEEPTB INFO    checkpoint saved as nnsk.iter445
    DEEPTB INFO    Epoch 445 summary:	train_loss: 0.007417	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep445
    DEEPTB INFO    iteration:446	train_loss: 0.007398  (0.007443)	lr: 0.006407
    DEEPTB INFO    checkpoint saved as nnsk.iter446
    DEEPTB INFO    Epoch 446 summary:	train_loss: 0.007398	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep446
    DEEPTB INFO    iteration:447	train_loss: 0.007380  (0.007424)	lr: 0.0064  
    DEEPTB INFO    checkpoint saved as nnsk.iter447
    DEEPTB INFO    Epoch 447 summary:	train_loss: 0.007380	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep447
    DEEPTB INFO    iteration:448	train_loss: 0.007361  (0.007405)	lr: 0.006394
    DEEPTB INFO    checkpoint saved as nnsk.iter448
    DEEPTB INFO    Epoch 448 summary:	train_loss: 0.007361	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep448
    DEEPTB INFO    iteration:449	train_loss: 0.007342  (0.007386)	lr: 0.006388
    DEEPTB INFO    checkpoint saved as nnsk.iter449
    DEEPTB INFO    Epoch 449 summary:	train_loss: 0.007342	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep449
    DEEPTB INFO    iteration:450	train_loss: 0.007324  (0.007368)	lr: 0.006381
    DEEPTB INFO    checkpoint saved as nnsk.iter450
    DEEPTB INFO    Epoch 450 summary:	train_loss: 0.007324	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep450
    DEEPTB INFO    iteration:451	train_loss: 0.007305  (0.007349)	lr: 0.006375
    DEEPTB INFO    checkpoint saved as nnsk.iter451
    DEEPTB INFO    Epoch 451 summary:	train_loss: 0.007305	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep451
    DEEPTB INFO    iteration:452	train_loss: 0.007287  (0.007330)	lr: 0.006368
    DEEPTB INFO    checkpoint saved as nnsk.iter452
    DEEPTB INFO    Epoch 452 summary:	train_loss: 0.007287	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep452
    DEEPTB INFO    iteration:453	train_loss: 0.007268  (0.007312)	lr: 0.006362
    DEEPTB INFO    checkpoint saved as nnsk.iter453
    DEEPTB INFO    Epoch 453 summary:	train_loss: 0.007268	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep453
    DEEPTB INFO    iteration:454	train_loss: 0.007250  (0.007293)	lr: 0.006356
    DEEPTB INFO    checkpoint saved as nnsk.iter454
    DEEPTB INFO    Epoch 454 summary:	train_loss: 0.007250	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep454
    DEEPTB INFO    iteration:455	train_loss: 0.007232  (0.007275)	lr: 0.006349
    DEEPTB INFO    checkpoint saved as nnsk.iter455
    DEEPTB INFO    Epoch 455 summary:	train_loss: 0.007232	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep455
    DEEPTB INFO    iteration:456	train_loss: 0.007213  (0.007256)	lr: 0.006343
    DEEPTB INFO    checkpoint saved as nnsk.iter456
    DEEPTB INFO    Epoch 456 summary:	train_loss: 0.007213	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep456
    DEEPTB INFO    iteration:457	train_loss: 0.007195  (0.007238)	lr: 0.006337
    DEEPTB INFO    checkpoint saved as nnsk.iter457
    DEEPTB INFO    Epoch 457 summary:	train_loss: 0.007195	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep457
    DEEPTB INFO    iteration:458	train_loss: 0.007177  (0.007220)	lr: 0.00633 
    DEEPTB INFO    checkpoint saved as nnsk.iter458
    DEEPTB INFO    Epoch 458 summary:	train_loss: 0.007177	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep458
    DEEPTB INFO    iteration:459	train_loss: 0.007159  (0.007202)	lr: 0.006324
    DEEPTB INFO    checkpoint saved as nnsk.iter459
    DEEPTB INFO    Epoch 459 summary:	train_loss: 0.007159	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep459
    DEEPTB INFO    iteration:460	train_loss: 0.007141  (0.007183)	lr: 0.006318
    DEEPTB INFO    checkpoint saved as nnsk.iter460
    DEEPTB INFO    Epoch 460 summary:	train_loss: 0.007141	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep460
    DEEPTB INFO    iteration:461	train_loss: 0.007123  (0.007165)	lr: 0.006311
    DEEPTB INFO    checkpoint saved as nnsk.iter461
    DEEPTB INFO    Epoch 461 summary:	train_loss: 0.007123	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep461
    DEEPTB INFO    iteration:462	train_loss: 0.007105  (0.007147)	lr: 0.006305
    DEEPTB INFO    checkpoint saved as nnsk.iter462
    DEEPTB INFO    Epoch 462 summary:	train_loss: 0.007105	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep462
    DEEPTB INFO    iteration:463	train_loss: 0.007087  (0.007129)	lr: 0.006299
    DEEPTB INFO    checkpoint saved as nnsk.iter463
    DEEPTB INFO    Epoch 463 summary:	train_loss: 0.007087	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep463
    DEEPTB INFO    iteration:464	train_loss: 0.007070  (0.007111)	lr: 0.006292
    DEEPTB INFO    checkpoint saved as nnsk.iter464
    DEEPTB INFO    Epoch 464 summary:	train_loss: 0.007070	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep464
    DEEPTB INFO    iteration:465	train_loss: 0.007052  (0.007094)	lr: 0.006286
    DEEPTB INFO    checkpoint saved as nnsk.iter465
    DEEPTB INFO    Epoch 465 summary:	train_loss: 0.007052	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep465
    DEEPTB INFO    iteration:466	train_loss: 0.007034  (0.007076)	lr: 0.00628 
    DEEPTB INFO    checkpoint saved as nnsk.iter466
    DEEPTB INFO    Epoch 466 summary:	train_loss: 0.007034	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep466
    DEEPTB INFO    iteration:467	train_loss: 0.007017  (0.007058)	lr: 0.006274
    DEEPTB INFO    checkpoint saved as nnsk.iter467
    DEEPTB INFO    Epoch 467 summary:	train_loss: 0.007017	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep467
    DEEPTB INFO    iteration:468	train_loss: 0.006999  (0.007040)	lr: 0.006267
    DEEPTB INFO    checkpoint saved as nnsk.iter468
    DEEPTB INFO    Epoch 468 summary:	train_loss: 0.006999	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep468
    DEEPTB INFO    iteration:469	train_loss: 0.006982  (0.007023)	lr: 0.006261
    DEEPTB INFO    checkpoint saved as nnsk.iter469
    DEEPTB INFO    Epoch 469 summary:	train_loss: 0.006982	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep469
    DEEPTB INFO    iteration:470	train_loss: 0.006964  (0.007005)	lr: 0.006255
    DEEPTB INFO    checkpoint saved as nnsk.iter470
    DEEPTB INFO    Epoch 470 summary:	train_loss: 0.006964	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep470
    DEEPTB INFO    iteration:471	train_loss: 0.006947  (0.006988)	lr: 0.006249
    DEEPTB INFO    checkpoint saved as nnsk.iter471
    DEEPTB INFO    Epoch 471 summary:	train_loss: 0.006947	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep471
    DEEPTB INFO    iteration:472	train_loss: 0.006930  (0.006970)	lr: 0.006242
    DEEPTB INFO    checkpoint saved as nnsk.iter472
    DEEPTB INFO    Epoch 472 summary:	train_loss: 0.006930	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep472
    DEEPTB INFO    iteration:473	train_loss: 0.006913  (0.006953)	lr: 0.006236
    DEEPTB INFO    checkpoint saved as nnsk.iter473
    DEEPTB INFO    Epoch 473 summary:	train_loss: 0.006913	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep473
    DEEPTB INFO    iteration:474	train_loss: 0.006896  (0.006936)	lr: 0.00623 
    DEEPTB INFO    checkpoint saved as nnsk.iter474
    DEEPTB INFO    Epoch 474 summary:	train_loss: 0.006896	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep474
    DEEPTB INFO    iteration:475	train_loss: 0.006879  (0.006919)	lr: 0.006224
    DEEPTB INFO    checkpoint saved as nnsk.iter475
    DEEPTB INFO    Epoch 475 summary:	train_loss: 0.006879	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep475
    DEEPTB INFO    iteration:476	train_loss: 0.006862  (0.006902)	lr: 0.006217
    DEEPTB INFO    checkpoint saved as nnsk.iter476
    DEEPTB INFO    Epoch 476 summary:	train_loss: 0.006862	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep476
    DEEPTB INFO    iteration:477	train_loss: 0.006845  (0.006885)	lr: 0.006211
    DEEPTB INFO    checkpoint saved as nnsk.iter477
    DEEPTB INFO    Epoch 477 summary:	train_loss: 0.006845	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep477
    DEEPTB INFO    iteration:478	train_loss: 0.006828  (0.006868)	lr: 0.006205
    DEEPTB INFO    checkpoint saved as nnsk.iter478
    DEEPTB INFO    Epoch 478 summary:	train_loss: 0.006828	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep478
    DEEPTB INFO    iteration:479	train_loss: 0.006812  (0.006851)	lr: 0.006199
    DEEPTB INFO    checkpoint saved as nnsk.iter479
    DEEPTB INFO    Epoch 479 summary:	train_loss: 0.006812	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep479
    DEEPTB INFO    iteration:480	train_loss: 0.006795  (0.006834)	lr: 0.006193
    DEEPTB INFO    checkpoint saved as nnsk.iter480
    DEEPTB INFO    Epoch 480 summary:	train_loss: 0.006795	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep480
    DEEPTB INFO    iteration:481	train_loss: 0.006778  (0.006817)	lr: 0.006186
    DEEPTB INFO    checkpoint saved as nnsk.iter481
    DEEPTB INFO    Epoch 481 summary:	train_loss: 0.006778	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep481
    DEEPTB INFO    iteration:482	train_loss: 0.006762  (0.006801)	lr: 0.00618 
    DEEPTB INFO    checkpoint saved as nnsk.iter482
    DEEPTB INFO    Epoch 482 summary:	train_loss: 0.006762	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep482
    DEEPTB INFO    iteration:483	train_loss: 0.006746  (0.006784)	lr: 0.006174
    DEEPTB INFO    checkpoint saved as nnsk.iter483
    DEEPTB INFO    Epoch 483 summary:	train_loss: 0.006746	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep483
    DEEPTB INFO    iteration:484	train_loss: 0.006729  (0.006768)	lr: 0.006168
    DEEPTB INFO    checkpoint saved as nnsk.iter484
    DEEPTB INFO    Epoch 484 summary:	train_loss: 0.006729	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep484
    DEEPTB INFO    iteration:485	train_loss: 0.006713  (0.006751)	lr: 0.006162
    DEEPTB INFO    checkpoint saved as nnsk.iter485
    DEEPTB INFO    Epoch 485 summary:	train_loss: 0.006713	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep485
    DEEPTB INFO    iteration:486	train_loss: 0.006697  (0.006735)	lr: 0.006155
    DEEPTB INFO    checkpoint saved as nnsk.iter486
    DEEPTB INFO    Epoch 486 summary:	train_loss: 0.006697	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep486
    DEEPTB INFO    iteration:487	train_loss: 0.006681  (0.006719)	lr: 0.006149
    DEEPTB INFO    checkpoint saved as nnsk.iter487
    DEEPTB INFO    Epoch 487 summary:	train_loss: 0.006681	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep487
    DEEPTB INFO    iteration:488	train_loss: 0.006665  (0.006703)	lr: 0.006143
    DEEPTB INFO    checkpoint saved as nnsk.iter488
    DEEPTB INFO    Epoch 488 summary:	train_loss: 0.006665	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep488
    DEEPTB INFO    iteration:489	train_loss: 0.006649  (0.006687)	lr: 0.006137
    DEEPTB INFO    checkpoint saved as nnsk.iter489
    DEEPTB INFO    Epoch 489 summary:	train_loss: 0.006649	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep489
    DEEPTB INFO    iteration:490	train_loss: 0.006633  (0.006671)	lr: 0.006131
    DEEPTB INFO    checkpoint saved as nnsk.iter490
    DEEPTB INFO    Epoch 490 summary:	train_loss: 0.006633	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep490
    DEEPTB INFO    iteration:491	train_loss: 0.006618  (0.006655)	lr: 0.006125
    DEEPTB INFO    checkpoint saved as nnsk.iter491
    DEEPTB INFO    Epoch 491 summary:	train_loss: 0.006618	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep491
    DEEPTB INFO    iteration:492	train_loss: 0.006602  (0.006639)	lr: 0.006119
    DEEPTB INFO    checkpoint saved as nnsk.iter492
    DEEPTB INFO    Epoch 492 summary:	train_loss: 0.006602	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep492
    DEEPTB INFO    iteration:493	train_loss: 0.006587  (0.006624)	lr: 0.006113
    DEEPTB INFO    checkpoint saved as nnsk.iter493
    DEEPTB INFO    Epoch 493 summary:	train_loss: 0.006587	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep493
    DEEPTB INFO    iteration:494	train_loss: 0.006571  (0.006608)	lr: 0.006106
    DEEPTB INFO    checkpoint saved as nnsk.iter494
    DEEPTB INFO    Epoch 494 summary:	train_loss: 0.006571	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep494
    DEEPTB INFO    iteration:495	train_loss: 0.006556  (0.006592)	lr: 0.0061  
    DEEPTB INFO    checkpoint saved as nnsk.iter495
    DEEPTB INFO    Epoch 495 summary:	train_loss: 0.006556	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep495
    DEEPTB INFO    iteration:496	train_loss: 0.006541  (0.006577)	lr: 0.006094
    DEEPTB INFO    checkpoint saved as nnsk.iter496
    DEEPTB INFO    Epoch 496 summary:	train_loss: 0.006541	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep496
    DEEPTB INFO    iteration:497	train_loss: 0.006526  (0.006561)	lr: 0.006088
    DEEPTB INFO    checkpoint saved as nnsk.iter497
    DEEPTB INFO    Epoch 497 summary:	train_loss: 0.006526	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep497
    DEEPTB INFO    iteration:498	train_loss: 0.006510  (0.006546)	lr: 0.006082
    DEEPTB INFO    checkpoint saved as nnsk.iter498
    DEEPTB INFO    Epoch 498 summary:	train_loss: 0.006510	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep498
    DEEPTB INFO    iteration:499	train_loss: 0.006495  (0.006531)	lr: 0.006076
    DEEPTB INFO    checkpoint saved as nnsk.iter499
    DEEPTB INFO    Epoch 499 summary:	train_loss: 0.006495	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep499
    DEEPTB INFO    iteration:500	train_loss: 0.006480  (0.006516)	lr: 0.00607 
    DEEPTB INFO    checkpoint saved as nnsk.iter500
    DEEPTB INFO    Epoch 500 summary:	train_loss: 0.006480	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep500
    DEEPTB INFO    finished training
    DEEPTB INFO    wall time: 95.351 s



```python
# !dptb run band.json -i ./nnsk1/checkpoint/nnsk.best.pth -o band1 -stu ../data/silicon.vasp
!dptb run band.json -i ./ref_ckpt/nnsk_tr1.pth  -o band1 -stu ../data/silicon.vasp

# display the band plot:
image_path = f'./band1/results/band.png'
img = mpimg.imread(image_path)
plt.figure(figsize=(10, 8))
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
     
     
    DEEPTB INFO    The ['overlap_param'] are frozen!
    DEEPTB INFO    The ['overlap_param'] are frozen!
    /opt/mamba/lib/python3.10/site-packages/torch/nested/__init__.py:107: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. (Triggered internally at ../aten/src/ATen/NestedTensorImpl.cpp:178.)
      return torch._nested_tensor_from_tensor_list(ts, dtype, None, device, None)
    DEEPTB INFO    KPOINTS  klist: 302 kpoints
    DEEPTB INFO    The eigenvalues are already in data. will use them.
    DEEPTB INFO    Calculating Fermi energy in the case of spin-degeneracy.
    DEEPTB INFO    Fermi energy converged after 20 iterations.
    DEEPTB INFO    q_cal: 8.000000000060934, total_electrons: 8.0, diff q: 6.093436866194679e-11
    DEEPTB INFO    Estimated E_fermi: -3.8649364919354783 based on the valence electrons setting nel_atom : {'Si': 4} .
    DEEPTB INFO    Using input Fermi energy: -4.7220 eV (estimated: -3.8649 eV)
    Figure(640x560)
    DEEPTB INFO    band calculation successfully completed.



    
![png](tu3_files/tu3_15_1.png)
    


#### **2.3 Training on MD data (bond length dependence)**
In **DeePTB**, the SK integral based on physical images is parameterized by various bond length-related functions. For example, in the above forms of `powerlaw` and `poly4pow`, the bond integral is an explicit function of bond length. This provides good transferability for the `NNSK` model, allowing it to fully simulate the changes in electronic structure caused by structural distortion.

To further improve the transferability of such models, we strongly recommend training bond length-dependent parameters. This type of training can be easily obtained from MD trajectory datasets. Additionally, it is important to ensure that the training dataset is diverse and representative of the various bond lengths encountered in practical scenarios.

We provide datasets for 10 MD frames at 25K, 100K, and 300K. Users can easily obtain bond length-dependent `NNSK` models by modifying the `data_options/train/prefix` in the input configuration to `kpathmd25/kpathmd100/kpathmd300`, and using the `-i` option to initialize the checkpoint for training.

During the training on MD trajectory datasets, in addition to providing the `train` dataset of MD trajectory data, it is also recommended to provide the single structure dataset used in the previous training as a `reference` dataset. This can help stabilize the training process for the MD trajectory. When using the `reference` dataset, it is necessary to specify the `ref_batch_size` in `train_option` and the corresponding loss calculation method for the `reference` dataset in `train_loss`. The rest of the input content remains unchanged. For specific input details, please refer to `input_2.json`.

After training the model, users can use the same band plotting API as mentioned earlier to visualize the band structure. The plotting parameters are located in `./run/band_2.json`.

We now adjust the training input parameters. For details, please refer to `input_2.json`. We change the prefix of the dataset to `kpathmd100.0`. We can also set other parameters such as learning rate and number of iterations. For specific parameter settings, please refer to `input_2.json`.

main changes in  data_options：
```json
{
    "data_options": {
        "train": {
            "root": "./data/",
            "prefix": "kpathmd100",
            "get_eigenvalues": true,
            "get_Hamiltonian": false
        },
        "reference": {
            "root": "./data/",
            "prefix": "kpath_spk",
            "type": "DefaultDataset",
            "get_eigenvalues": true,
            "get_Hamiltonian": false
        }
    }
}
```
and loss_options

```json
{
        "loss_options": {
            "train": {
                "method": "eigvals",
                "diff_on": false,
                "eout_weight": 0.001,
                "diff_weight": 0.01
            },
            "reference": {
                "method": "eigvals",
                "diff_on": false,
                "eout_weight": 0.001,
                "diff_weight": 0.01
            }
        },
}
```


Here, just as a demonstration, we also reduced the number of training iterations.


```python
# v100, 4m4s 
# !dptb train input_2.json -i nnsk1/checkpoint/nnsk.best.pth -o nnskmd100
!dptb train input_2.json -i ./ref_ckpt/nnsk_tr1.pth  -o nnskmd100
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
     
     
    DEEPTB INFO    ------------------------------------------------------------------
    DEEPTB INFO         Cutoff options:                                            
    DEEPTB INFO                                                                    
    DEEPTB INFO         r_max            : {'Si-Si': 6.24}                         
    DEEPTB INFO         er_max           : None                                    
    DEEPTB INFO         oer_max          : None                                    
    DEEPTB INFO    ------------------------------------------------------------------
    /opt/mamba/lib/python3.10/site-packages/torch/nested/__init__.py:107: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. (Triggered internally at ../aten/src/ATen/NestedTensorImpl.cpp:178.)
      return torch._nested_tensor_from_tensor_list(ts, dtype, None, device, None)
    DEEPTB WARNING The cutoffs in data and model are not checked. be careful!
    DEEPTB WARNING The cutoffs in data and model are not checked. be careful!
    DEEPTB INFO    The ['overlap_param'] are frozen!
    DEEPTB INFO    The ['overlap_param'] are frozen!
    DEEPTB INFO    iteration:1	train_loss: 0.127401  (0.038220)	lr: 0.01
    DEEPTB INFO    checkpoint saved as nnsk.iter1
    DEEPTB INFO    iteration:2	train_loss: 0.469675  (0.167657)	lr: 0.01
    DEEPTB INFO    checkpoint saved as nnsk.iter2
    DEEPTB INFO    iteration:3	train_loss: 0.126339  (0.155261)	lr: 0.01
    DEEPTB INFO    checkpoint saved as nnsk.iter3
    DEEPTB INFO    iteration:4	train_loss: 0.213772  (0.172815)	lr: 0.01
    DEEPTB INFO    checkpoint saved as nnsk.iter4
    DEEPTB INFO    iteration:5	train_loss: 0.307565  (0.213240)	lr: 0.01
    DEEPTB INFO    checkpoint saved as nnsk.iter5
    DEEPTB INFO    iteration:6	train_loss: 0.205331  (0.210867)	lr: 0.01
    DEEPTB INFO    checkpoint saved as nnsk.iter6
    DEEPTB INFO    iteration:7	train_loss: 0.102731  (0.178426)	lr: 0.01
    DEEPTB INFO    checkpoint saved as nnsk.iter7
    DEEPTB INFO    iteration:8	train_loss: 0.112137  (0.158539)	lr: 0.01
    DEEPTB INFO    checkpoint saved as nnsk.iter8
    DEEPTB INFO    iteration:9	train_loss: 0.176452  (0.163913)	lr: 0.01
    DEEPTB INFO    checkpoint saved as nnsk.iter9
    DEEPTB INFO    iteration:10	train_loss: 0.185835  (0.170490)	lr: 0.01
    DEEPTB INFO    checkpoint saved as nnsk.iter10
    DEEPTB INFO    Epoch 1 summary:	train_loss: 0.202724	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep1
    DEEPTB INFO    iteration:11	train_loss: 0.134073  (0.159565)	lr: 0.00999
    DEEPTB INFO    checkpoint saved as nnsk.iter11
    DEEPTB INFO    iteration:12	train_loss: 0.085555  (0.137362)	lr: 0.00999
    DEEPTB INFO    checkpoint saved as nnsk.iter12
    DEEPTB INFO    iteration:13	train_loss: 0.087610  (0.122436)	lr: 0.00999
    DEEPTB INFO    checkpoint saved as nnsk.iter13
    DEEPTB INFO    iteration:14	train_loss: 0.119257  (0.121482)	lr: 0.00999
    DEEPTB INFO    checkpoint saved as nnsk.iter14
    DEEPTB INFO    iteration:15	train_loss: 0.131779  (0.124571)	lr: 0.00999
    DEEPTB INFO    checkpoint saved as nnsk.iter15
    DEEPTB INFO    iteration:16	train_loss: 0.107228  (0.119369)	lr: 0.00999
    DEEPTB INFO    checkpoint saved as nnsk.iter16
    DEEPTB INFO    iteration:17	train_loss: 0.075378  (0.106171)	lr: 0.00999
    DEEPTB INFO    checkpoint saved as nnsk.iter17
    DEEPTB INFO    iteration:18	train_loss: 0.066472  (0.094262)	lr: 0.00999
    DEEPTB INFO    checkpoint saved as nnsk.iter18
    DEEPTB INFO    iteration:19	train_loss: 0.080439  (0.090115)	lr: 0.00999
    DEEPTB INFO    checkpoint saved as nnsk.iter19
    DEEPTB INFO    iteration:20	train_loss: 0.093409  (0.091103)	lr: 0.00999
    DEEPTB INFO    checkpoint saved as nnsk.iter20
    DEEPTB INFO    Epoch 2 summary:	train_loss: 0.098120	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep2
    DEEPTB INFO    iteration:21	train_loss: 0.086104  (0.089603)	lr: 0.00998
    DEEPTB INFO    checkpoint saved as nnsk.iter21
    DEEPTB INFO    iteration:22	train_loss: 0.066814  (0.082767)	lr: 0.00998
    DEEPTB INFO    checkpoint saved as nnsk.iter22
    DEEPTB INFO    iteration:23	train_loss: 0.055933  (0.074717)	lr: 0.00998
    DEEPTB INFO    checkpoint saved as nnsk.iter23
    DEEPTB INFO    iteration:24	train_loss: 0.060440  (0.070434)	lr: 0.00998
    DEEPTB INFO    checkpoint saved as nnsk.iter24
    DEEPTB INFO    iteration:25	train_loss: 0.069970  (0.070295)	lr: 0.00998
    DEEPTB INFO    checkpoint saved as nnsk.iter25
    DEEPTB INFO    iteration:26	train_loss: 0.068573  (0.069778)	lr: 0.00998
    DEEPTB INFO    checkpoint saved as nnsk.iter26
    DEEPTB INFO    iteration:27	train_loss: 0.057544  (0.066108)	lr: 0.00998
    DEEPTB INFO    checkpoint saved as nnsk.iter27
    DEEPTB INFO    iteration:28	train_loss: 0.048378  (0.060789)	lr: 0.00998
    DEEPTB INFO    checkpoint saved as nnsk.iter28
    DEEPTB INFO    iteration:29	train_loss: 0.048718  (0.057168)	lr: 0.00998
    DEEPTB INFO    checkpoint saved as nnsk.iter29
    DEEPTB INFO    iteration:30	train_loss: 0.054448  (0.056352)	lr: 0.00998
    DEEPTB INFO    checkpoint saved as nnsk.iter30
    DEEPTB INFO    Epoch 3 summary:	train_loss: 0.061692	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep3
    DEEPTB INFO    iteration:31	train_loss: 0.055510  (0.056099)	lr: 0.00997
    DEEPTB INFO    checkpoint saved as nnsk.iter31
    DEEPTB INFO    iteration:32	train_loss: 0.048821  (0.053916)	lr: 0.00997
    DEEPTB INFO    checkpoint saved as nnsk.iter32
    DEEPTB INFO    iteration:33	train_loss: 0.042549  (0.050506)	lr: 0.00997
    DEEPTB INFO    checkpoint saved as nnsk.iter33
    DEEPTB INFO    iteration:34	train_loss: 0.041689  (0.047861)	lr: 0.00997
    DEEPTB INFO    checkpoint saved as nnsk.iter34
    DEEPTB INFO    iteration:35	train_loss: 0.045471  (0.047144)	lr: 0.00997
    DEEPTB INFO    checkpoint saved as nnsk.iter35
    DEEPTB INFO    iteration:36	train_loss: 0.046039  (0.046812)	lr: 0.00997
    DEEPTB INFO    checkpoint saved as nnsk.iter36
    DEEPTB INFO    iteration:37	train_loss: 0.042159  (0.045416)	lr: 0.00997
    DEEPTB INFO    checkpoint saved as nnsk.iter37
    DEEPTB INFO    iteration:38	train_loss: 0.037371  (0.043003)	lr: 0.00997
    DEEPTB INFO    checkpoint saved as nnsk.iter38
    DEEPTB INFO    iteration:39	train_loss: 0.037382  (0.041317)	lr: 0.00997
    DEEPTB INFO    checkpoint saved as nnsk.iter39
    DEEPTB INFO    iteration:40	train_loss: 0.039402  (0.040742)	lr: 0.00997
    DEEPTB INFO    checkpoint saved as nnsk.iter40
    DEEPTB INFO    Epoch 4 summary:	train_loss: 0.043639	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep4
    DEEPTB INFO    iteration:41	train_loss: 0.039810  (0.040462)	lr: 0.00996
    DEEPTB INFO    checkpoint saved as nnsk.iter41
    DEEPTB INFO    iteration:42	train_loss: 0.036363  (0.039233)	lr: 0.00996
    DEEPTB INFO    checkpoint saved as nnsk.iter42
    DEEPTB INFO    iteration:43	train_loss: 0.033929  (0.037642)	lr: 0.00996
    DEEPTB INFO    checkpoint saved as nnsk.iter43
    DEEPTB INFO    iteration:44	train_loss: 0.033970  (0.036540)	lr: 0.00996
    DEEPTB INFO    checkpoint saved as nnsk.iter44
    DEEPTB INFO    iteration:45	train_loss: 0.035501  (0.036228)	lr: 0.00996
    DEEPTB INFO    checkpoint saved as nnsk.iter45
    DEEPTB INFO    iteration:46	train_loss: 0.034741  (0.035782)	lr: 0.00996
    DEEPTB INFO    checkpoint saved as nnsk.iter46
    DEEPTB INFO    iteration:47	train_loss: 0.032563  (0.034816)	lr: 0.00996
    DEEPTB INFO    checkpoint saved as nnsk.iter47
    DEEPTB INFO    iteration:48	train_loss: 0.031506  (0.033823)	lr: 0.00996
    DEEPTB INFO    checkpoint saved as nnsk.iter48
    DEEPTB INFO    iteration:49	train_loss: 0.031988  (0.033273)	lr: 0.00996
    DEEPTB INFO    checkpoint saved as nnsk.iter49
    DEEPTB INFO    iteration:50	train_loss: 0.032713  (0.033105)	lr: 0.00996
    DEEPTB INFO    checkpoint saved as nnsk.iter50
    DEEPTB INFO    Epoch 5 summary:	train_loss: 0.034308	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep5
    DEEPTB INFO    iteration:51	train_loss: 0.031454  (0.032610)	lr: 0.00995
    DEEPTB INFO    checkpoint saved as nnsk.iter51
    DEEPTB INFO    iteration:52	train_loss: 0.030248  (0.031901)	lr: 0.00995
    DEEPTB INFO    checkpoint saved as nnsk.iter52
    DEEPTB INFO    iteration:53	train_loss: 0.029831  (0.031280)	lr: 0.00995
    DEEPTB INFO    checkpoint saved as nnsk.iter53
    DEEPTB INFO    iteration:54	train_loss: 0.029725  (0.030813)	lr: 0.00995
    DEEPTB INFO    checkpoint saved as nnsk.iter54
    DEEPTB INFO    iteration:55	train_loss: 0.029843  (0.030522)	lr: 0.00995
    DEEPTB INFO    checkpoint saved as nnsk.iter55
    DEEPTB INFO    iteration:56	train_loss: 0.029073  (0.030088)	lr: 0.00995
    DEEPTB INFO    checkpoint saved as nnsk.iter56
    DEEPTB INFO    iteration:57	train_loss: 0.028133  (0.029501)	lr: 0.00995
    DEEPTB INFO    checkpoint saved as nnsk.iter57
    DEEPTB INFO    iteration:58	train_loss: 0.028236  (0.029122)	lr: 0.00995
    DEEPTB INFO    checkpoint saved as nnsk.iter58
    DEEPTB INFO    iteration:59	train_loss: 0.028036  (0.028796)	lr: 0.00995
    DEEPTB INFO    checkpoint saved as nnsk.iter59
    DEEPTB INFO    iteration:60	train_loss: 0.027892  (0.028525)	lr: 0.00995
    DEEPTB INFO    checkpoint saved as nnsk.iter60
    DEEPTB INFO    Epoch 6 summary:	train_loss: 0.029247	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep6
    DEEPTB INFO    iteration:61	train_loss: 0.027020  (0.028073)	lr: 0.00994
    DEEPTB INFO    checkpoint saved as nnsk.iter61
    DEEPTB INFO    iteration:62	train_loss: 0.026528  (0.027610)	lr: 0.00994
    DEEPTB INFO    checkpoint saved as nnsk.iter62
    DEEPTB INFO    iteration:63	train_loss: 0.026571  (0.027298)	lr: 0.00994
    DEEPTB INFO    checkpoint saved as nnsk.iter63
    DEEPTB INFO    iteration:64	train_loss: 0.026873  (0.027171)	lr: 0.00994
    DEEPTB INFO    checkpoint saved as nnsk.iter64
    DEEPTB INFO    iteration:65	train_loss: 0.026329  (0.026918)	lr: 0.00994
    DEEPTB INFO    checkpoint saved as nnsk.iter65
    DEEPTB INFO    iteration:66	train_loss: 0.025495  (0.026491)	lr: 0.00994
    DEEPTB INFO    checkpoint saved as nnsk.iter66
    DEEPTB INFO    iteration:67	train_loss: 0.025399  (0.026163)	lr: 0.00994
    DEEPTB INFO    checkpoint saved as nnsk.iter67
    DEEPTB INFO    iteration:68	train_loss: 0.025621  (0.026001)	lr: 0.00994
    DEEPTB INFO    checkpoint saved as nnsk.iter68
    DEEPTB INFO    iteration:69	train_loss: 0.025198  (0.025760)	lr: 0.00994
    DEEPTB INFO    checkpoint saved as nnsk.iter69
    DEEPTB INFO    iteration:70	train_loss: 0.024792  (0.025469)	lr: 0.00994
    DEEPTB INFO    checkpoint saved as nnsk.iter70
    DEEPTB INFO    Epoch 7 summary:	train_loss: 0.025983	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep7
    DEEPTB INFO    iteration:71	train_loss: 0.024568  (0.025199)	lr: 0.00993
    DEEPTB INFO    checkpoint saved as nnsk.iter71
    DEEPTB INFO    iteration:72	train_loss: 0.024628  (0.025028)	lr: 0.00993
    DEEPTB INFO    checkpoint saved as nnsk.iter72
    DEEPTB INFO    iteration:73	train_loss: 0.024313  (0.024813)	lr: 0.00993
    DEEPTB INFO    checkpoint saved as nnsk.iter73
    DEEPTB INFO    iteration:74	train_loss: 0.023718  (0.024485)	lr: 0.00993
    DEEPTB INFO    checkpoint saved as nnsk.iter74
    DEEPTB INFO    iteration:75	train_loss: 0.023986  (0.024335)	lr: 0.00993
    DEEPTB INFO    checkpoint saved as nnsk.iter75
    DEEPTB INFO    iteration:76	train_loss: 0.023502  (0.024085)	lr: 0.00993
    DEEPTB INFO    checkpoint saved as nnsk.iter76
    DEEPTB INFO    iteration:77	train_loss: 0.023578  (0.023933)	lr: 0.00993
    DEEPTB INFO    checkpoint saved as nnsk.iter77
    DEEPTB INFO    iteration:78	train_loss: 0.023428  (0.023782)	lr: 0.00993
    DEEPTB INFO    checkpoint saved as nnsk.iter78
    DEEPTB INFO    iteration:79	train_loss: 0.022878  (0.023510)	lr: 0.00993
    DEEPTB INFO    checkpoint saved as nnsk.iter79
    DEEPTB INFO    iteration:80	train_loss: 0.022938  (0.023339)	lr: 0.00993
    DEEPTB INFO    checkpoint saved as nnsk.iter80
    DEEPTB INFO    Epoch 8 summary:	train_loss: 0.023754	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep8
    DEEPTB INFO    iteration:81	train_loss: 0.022786  (0.023173)	lr: 0.00992
    DEEPTB INFO    checkpoint saved as nnsk.iter81
    DEEPTB INFO    iteration:82	train_loss: 0.022516  (0.022976)	lr: 0.00992
    DEEPTB INFO    checkpoint saved as nnsk.iter82
    DEEPTB INFO    iteration:83	train_loss: 0.022418  (0.022809)	lr: 0.00992
    DEEPTB INFO    checkpoint saved as nnsk.iter83
    DEEPTB INFO    iteration:84	train_loss: 0.022018  (0.022571)	lr: 0.00992
    DEEPTB INFO    checkpoint saved as nnsk.iter84
    DEEPTB INFO    iteration:85	train_loss: 0.022115  (0.022435)	lr: 0.00992
    DEEPTB INFO    checkpoint saved as nnsk.iter85
    DEEPTB INFO    iteration:86	train_loss: 0.022045  (0.022318)	lr: 0.00992
    DEEPTB INFO    checkpoint saved as nnsk.iter86
    DEEPTB INFO    iteration:87	train_loss: 0.021489  (0.022069)	lr: 0.00992
    DEEPTB INFO    checkpoint saved as nnsk.iter87
    DEEPTB INFO    iteration:88	train_loss: 0.021422  (0.021875)	lr: 0.00992
    DEEPTB INFO    checkpoint saved as nnsk.iter88
    DEEPTB INFO    iteration:89	train_loss: 0.021699  (0.021822)	lr: 0.00992
    DEEPTB INFO    checkpoint saved as nnsk.iter89
    DEEPTB INFO    iteration:90	train_loss: 0.021134  (0.021616)	lr: 0.00992
    DEEPTB INFO    checkpoint saved as nnsk.iter90
    DEEPTB INFO    Epoch 9 summary:	train_loss: 0.021964	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep9
    DEEPTB INFO    iteration:91	train_loss: 0.021141  (0.021473)	lr: 0.00991
    DEEPTB INFO    checkpoint saved as nnsk.iter91
    DEEPTB INFO    iteration:92	train_loss: 0.021076  (0.021354)	lr: 0.00991
    DEEPTB INFO    checkpoint saved as nnsk.iter92
    DEEPTB INFO    iteration:93	train_loss: 0.020705  (0.021159)	lr: 0.00991
    DEEPTB INFO    checkpoint saved as nnsk.iter93
    DEEPTB INFO    iteration:94	train_loss: 0.020469  (0.020952)	lr: 0.00991
    DEEPTB INFO    checkpoint saved as nnsk.iter94
    DEEPTB INFO    iteration:95	train_loss: 0.020598  (0.020846)	lr: 0.00991
    DEEPTB INFO    checkpoint saved as nnsk.iter95
    DEEPTB INFO    iteration:96	train_loss: 0.020288  (0.020678)	lr: 0.00991
    DEEPTB INFO    checkpoint saved as nnsk.iter96
    DEEPTB INFO    iteration:97	train_loss: 0.020088  (0.020501)	lr: 0.00991
    DEEPTB INFO    checkpoint saved as nnsk.iter97
    DEEPTB INFO    iteration:98	train_loss: 0.020145  (0.020394)	lr: 0.00991
    DEEPTB INFO    checkpoint saved as nnsk.iter98
    DEEPTB INFO    iteration:99	train_loss: 0.020194  (0.020334)	lr: 0.00991
    DEEPTB INFO    checkpoint saved as nnsk.iter99
    DEEPTB INFO    iteration:100	train_loss: 0.020118  (0.020269)	lr: 0.00991
    DEEPTB INFO    checkpoint saved as nnsk.iter100
    DEEPTB INFO    Epoch 10 summary:	train_loss: 0.020482	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep10
    DEEPTB INFO    iteration:101	train_loss: 0.019777  (0.020122)	lr: 0.0099 
    DEEPTB INFO    checkpoint saved as nnsk.iter101
    DEEPTB INFO    iteration:102	train_loss: 0.019691  (0.019992)	lr: 0.0099 
    DEEPTB INFO    checkpoint saved as nnsk.iter102
    DEEPTB INFO    iteration:103	train_loss: 0.019316  (0.019789)	lr: 0.0099 
    DEEPTB INFO    checkpoint saved as nnsk.iter103
    DEEPTB INFO    iteration:104	train_loss: 0.019296  (0.019641)	lr: 0.0099 
    DEEPTB INFO    checkpoint saved as nnsk.iter104
    DEEPTB INFO    iteration:105	train_loss: 0.019279  (0.019533)	lr: 0.0099 
    DEEPTB INFO    checkpoint saved as nnsk.iter105
    DEEPTB INFO    iteration:106	train_loss: 0.018943  (0.019356)	lr: 0.0099 
    DEEPTB INFO    checkpoint saved as nnsk.iter106
    DEEPTB INFO    iteration:107	train_loss: 0.019106  (0.019281)	lr: 0.0099 
    DEEPTB INFO    checkpoint saved as nnsk.iter107
    DEEPTB INFO    iteration:108	train_loss: 0.019122  (0.019233)	lr: 0.0099 
    DEEPTB INFO    checkpoint saved as nnsk.iter108
    DEEPTB INFO    iteration:109	train_loss: 0.018965  (0.019153)	lr: 0.0099 
    DEEPTB INFO    checkpoint saved as nnsk.iter109
    DEEPTB INFO    iteration:110	train_loss: 0.018585  (0.018983)	lr: 0.0099 
    DEEPTB INFO    checkpoint saved as nnsk.iter110
    DEEPTB INFO    Epoch 11 summary:	train_loss: 0.019208	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep11
    DEEPTB INFO    iteration:111	train_loss: 0.018363  (0.018797)	lr: 0.009891
    DEEPTB INFO    checkpoint saved as nnsk.iter111
    DEEPTB INFO    iteration:112	train_loss: 0.018346  (0.018662)	lr: 0.009891
    DEEPTB INFO    checkpoint saved as nnsk.iter112
    DEEPTB INFO    iteration:113	train_loss: 0.018600  (0.018643)	lr: 0.009891
    DEEPTB INFO    checkpoint saved as nnsk.iter113
    DEEPTB INFO    iteration:114	train_loss: 0.018115  (0.018485)	lr: 0.009891
    DEEPTB INFO    checkpoint saved as nnsk.iter114
    DEEPTB INFO    iteration:115	train_loss: 0.018279  (0.018423)	lr: 0.009891
    DEEPTB INFO    checkpoint saved as nnsk.iter115
    DEEPTB INFO    iteration:116	train_loss: 0.018119  (0.018332)	lr: 0.009891
    DEEPTB INFO    checkpoint saved as nnsk.iter116
    DEEPTB INFO    iteration:117	train_loss: 0.017914  (0.018207)	lr: 0.009891
    DEEPTB INFO    checkpoint saved as nnsk.iter117
    DEEPTB INFO    iteration:118	train_loss: 0.017834  (0.018095)	lr: 0.009891
    DEEPTB INFO    checkpoint saved as nnsk.iter118
    DEEPTB INFO    iteration:119	train_loss: 0.017536  (0.017927)	lr: 0.009891
    DEEPTB INFO    checkpoint saved as nnsk.iter119
    DEEPTB INFO    iteration:120	train_loss: 0.017606  (0.017831)	lr: 0.009891
    DEEPTB INFO    checkpoint saved as nnsk.iter120
    DEEPTB INFO    Epoch 12 summary:	train_loss: 0.018071	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep12
    DEEPTB INFO    iteration:121	train_loss: 0.017366  (0.017691)	lr: 0.009881
    DEEPTB INFO    checkpoint saved as nnsk.iter121
    DEEPTB INFO    iteration:122	train_loss: 0.017212  (0.017548)	lr: 0.009881
    DEEPTB INFO    checkpoint saved as nnsk.iter122
    DEEPTB INFO    iteration:123	train_loss: 0.017500  (0.017533)	lr: 0.009881
    DEEPTB INFO    checkpoint saved as nnsk.iter123
    DEEPTB INFO    iteration:124	train_loss: 0.017040  (0.017385)	lr: 0.009881
    DEEPTB INFO    checkpoint saved as nnsk.iter124
    DEEPTB INFO    iteration:125	train_loss: 0.017184  (0.017325)	lr: 0.009881
    DEEPTB INFO    checkpoint saved as nnsk.iter125
    DEEPTB INFO    iteration:126	train_loss: 0.017040  (0.017239)	lr: 0.009881
    DEEPTB INFO    checkpoint saved as nnsk.iter126
    DEEPTB INFO    iteration:127	train_loss: 0.016936  (0.017148)	lr: 0.009881
    DEEPTB INFO    checkpoint saved as nnsk.iter127
    DEEPTB INFO    iteration:128	train_loss: 0.016819  (0.017050)	lr: 0.009881
    DEEPTB INFO    checkpoint saved as nnsk.iter128
    DEEPTB INFO    iteration:129	train_loss: 0.016868  (0.016995)	lr: 0.009881
    DEEPTB INFO    checkpoint saved as nnsk.iter129
    DEEPTB INFO    iteration:130	train_loss: 0.016563  (0.016866)	lr: 0.009881
    DEEPTB INFO    checkpoint saved as nnsk.iter130
    DEEPTB INFO    Epoch 13 summary:	train_loss: 0.017053	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep13
    DEEPTB INFO    iteration:131	train_loss: 0.016733  (0.016826)	lr: 0.009871
    DEEPTB INFO    checkpoint saved as nnsk.iter131
    DEEPTB INFO    iteration:132	train_loss: 0.016458  (0.016716)	lr: 0.009871
    DEEPTB INFO    checkpoint saved as nnsk.iter132
    DEEPTB INFO    iteration:133	train_loss: 0.016335  (0.016601)	lr: 0.009871
    DEEPTB INFO    checkpoint saved as nnsk.iter133
    DEEPTB INFO    iteration:134	train_loss: 0.016103  (0.016452)	lr: 0.009871
    DEEPTB INFO    checkpoint saved as nnsk.iter134
    DEEPTB INFO    iteration:135	train_loss: 0.016062  (0.016335)	lr: 0.009871
    DEEPTB INFO    checkpoint saved as nnsk.iter135
    DEEPTB INFO    iteration:136	train_loss: 0.016124  (0.016272)	lr: 0.009871
    DEEPTB INFO    checkpoint saved as nnsk.iter136
    DEEPTB INFO    iteration:137	train_loss: 0.016150  (0.016235)	lr: 0.009871
    DEEPTB INFO    checkpoint saved as nnsk.iter137
    DEEPTB INFO    iteration:138	train_loss: 0.015749  (0.016089)	lr: 0.009871
    DEEPTB INFO    checkpoint saved as nnsk.iter138
    DEEPTB INFO    iteration:139	train_loss: 0.015935  (0.016043)	lr: 0.009871
    DEEPTB INFO    checkpoint saved as nnsk.iter139
    DEEPTB INFO    iteration:140	train_loss: 0.015659  (0.015928)	lr: 0.009871
    DEEPTB INFO    checkpoint saved as nnsk.iter140
    DEEPTB INFO    Epoch 14 summary:	train_loss: 0.016131	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep14
    DEEPTB INFO    iteration:141	train_loss: 0.015769  (0.015880)	lr: 0.009861
    DEEPTB INFO    checkpoint saved as nnsk.iter141
    DEEPTB INFO    iteration:142	train_loss: 0.015492  (0.015764)	lr: 0.009861
    DEEPTB INFO    checkpoint saved as nnsk.iter142
    DEEPTB INFO    iteration:143	train_loss: 0.015724  (0.015752)	lr: 0.009861
    DEEPTB INFO    checkpoint saved as nnsk.iter143
    DEEPTB INFO    iteration:144	train_loss: 0.015271  (0.015607)	lr: 0.009861
    DEEPTB INFO    checkpoint saved as nnsk.iter144
    DEEPTB INFO    iteration:145	train_loss: 0.015417  (0.015550)	lr: 0.009861
    DEEPTB INFO    checkpoint saved as nnsk.iter145
    DEEPTB INFO    iteration:146	train_loss: 0.015255  (0.015462)	lr: 0.009861
    DEEPTB INFO    checkpoint saved as nnsk.iter146
    DEEPTB INFO    iteration:147	train_loss: 0.015107  (0.015355)	lr: 0.009861
    DEEPTB INFO    checkpoint saved as nnsk.iter147
    DEEPTB INFO    iteration:148	train_loss: 0.014995  (0.015247)	lr: 0.009861
    DEEPTB INFO    checkpoint saved as nnsk.iter148
    DEEPTB INFO    iteration:149	train_loss: 0.015104  (0.015204)	lr: 0.009861
    DEEPTB INFO    checkpoint saved as nnsk.iter149
    DEEPTB INFO    iteration:150	train_loss: 0.015126  (0.015181)	lr: 0.009861
    DEEPTB INFO    checkpoint saved as nnsk.iter150
    DEEPTB INFO    Epoch 15 summary:	train_loss: 0.015326	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep15
    DEEPTB INFO    iteration:151	train_loss: 0.014925  (0.015104)	lr: 0.009851
    DEEPTB INFO    checkpoint saved as nnsk.iter151
    DEEPTB INFO    iteration:152	train_loss: 0.014906  (0.015045)	lr: 0.009851
    DEEPTB INFO    checkpoint saved as nnsk.iter152
    DEEPTB INFO    iteration:153	train_loss: 0.014857  (0.014988)	lr: 0.009851
    DEEPTB INFO    checkpoint saved as nnsk.iter153
    DEEPTB INFO    iteration:154	train_loss: 0.014671  (0.014893)	lr: 0.009851
    DEEPTB INFO    checkpoint saved as nnsk.iter154
    DEEPTB INFO    iteration:155	train_loss: 0.014466  (0.014765)	lr: 0.009851
    DEEPTB INFO    checkpoint saved as nnsk.iter155
    DEEPTB INFO    iteration:156	train_loss: 0.014458  (0.014673)	lr: 0.009851
    DEEPTB INFO    checkpoint saved as nnsk.iter156
    DEEPTB INFO    iteration:157	train_loss: 0.014319  (0.014567)	lr: 0.009851
    DEEPTB INFO    checkpoint saved as nnsk.iter157
    DEEPTB INFO    iteration:158	train_loss: 0.014544  (0.014560)	lr: 0.009851
    DEEPTB INFO    checkpoint saved as nnsk.iter158
    DEEPTB INFO    iteration:159	train_loss: 0.014221  (0.014458)	lr: 0.009851
    DEEPTB INFO    checkpoint saved as nnsk.iter159
    DEEPTB INFO    iteration:160	train_loss: 0.014452  (0.014456)	lr: 0.009851
    DEEPTB INFO    checkpoint saved as nnsk.iter160
    DEEPTB INFO    Epoch 16 summary:	train_loss: 0.014582	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep16
    DEEPTB INFO    iteration:161	train_loss: 0.014077  (0.014342)	lr: 0.009841
    DEEPTB INFO    checkpoint saved as nnsk.iter161
    DEEPTB INFO    iteration:162	train_loss: 0.014329  (0.014338)	lr: 0.009841
    DEEPTB INFO    checkpoint saved as nnsk.iter162
    DEEPTB INFO    iteration:163	train_loss: 0.013905  (0.014209)	lr: 0.009841
    DEEPTB INFO    checkpoint saved as nnsk.iter163
    DEEPTB INFO    iteration:164	train_loss: 0.014052  (0.014161)	lr: 0.009841
    DEEPTB INFO    checkpoint saved as nnsk.iter164
    DEEPTB INFO    iteration:165	train_loss: 0.013798  (0.014053)	lr: 0.009841
    DEEPTB INFO    checkpoint saved as nnsk.iter165
    DEEPTB INFO    iteration:166	train_loss: 0.014027  (0.014045)	lr: 0.009841
    DEEPTB INFO    checkpoint saved as nnsk.iter166
    DEEPTB INFO    iteration:167	train_loss: 0.013922  (0.014008)	lr: 0.009841
    DEEPTB INFO    checkpoint saved as nnsk.iter167
    DEEPTB INFO    iteration:168	train_loss: 0.013689  (0.013912)	lr: 0.009841
    DEEPTB INFO    checkpoint saved as nnsk.iter168
    DEEPTB INFO    iteration:169	train_loss: 0.013673  (0.013840)	lr: 0.009841
    DEEPTB INFO    checkpoint saved as nnsk.iter169
    DEEPTB INFO    iteration:170	train_loss: 0.013650  (0.013783)	lr: 0.009841
    DEEPTB INFO    checkpoint saved as nnsk.iter170
    DEEPTB INFO    Epoch 17 summary:	train_loss: 0.013912	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep17
    DEEPTB INFO    iteration:171	train_loss: 0.013618  (0.013734)	lr: 0.009831
    DEEPTB INFO    checkpoint saved as nnsk.iter171
    DEEPTB INFO    iteration:172	train_loss: 0.013495  (0.013662)	lr: 0.009831
    DEEPTB INFO    checkpoint saved as nnsk.iter172
    DEEPTB INFO    iteration:173	train_loss: 0.013565  (0.013633)	lr: 0.009831
    DEEPTB INFO    checkpoint saved as nnsk.iter173
    DEEPTB INFO    iteration:174	train_loss: 0.013417  (0.013568)	lr: 0.009831
    DEEPTB INFO    checkpoint saved as nnsk.iter174
    DEEPTB INFO    iteration:175	train_loss: 0.013539  (0.013559)	lr: 0.009831
    DEEPTB INFO    checkpoint saved as nnsk.iter175
    DEEPTB INFO    iteration:176	train_loss: 0.013167  (0.013441)	lr: 0.009831
    DEEPTB INFO    checkpoint saved as nnsk.iter176
    DEEPTB INFO    iteration:177	train_loss: 0.013182  (0.013364)	lr: 0.009831
    DEEPTB INFO    checkpoint saved as nnsk.iter177
    DEEPTB INFO    iteration:178	train_loss: 0.013022  (0.013261)	lr: 0.009831
    DEEPTB INFO    checkpoint saved as nnsk.iter178
    DEEPTB INFO    iteration:179	train_loss: 0.013273  (0.013265)	lr: 0.009831
    DEEPTB INFO    checkpoint saved as nnsk.iter179
    DEEPTB INFO    iteration:180	train_loss: 0.012936  (0.013166)	lr: 0.009831
    DEEPTB INFO    checkpoint saved as nnsk.iter180
    DEEPTB INFO    Epoch 18 summary:	train_loss: 0.013321	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep18
    DEEPTB INFO    iteration:181	train_loss: 0.012882  (0.013081)	lr: 0.009822
    DEEPTB INFO    checkpoint saved as nnsk.iter181
    DEEPTB INFO    iteration:182	train_loss: 0.013144  (0.013100)	lr: 0.009822
    DEEPTB INFO    checkpoint saved as nnsk.iter182
    DEEPTB INFO    iteration:183	train_loss: 0.012797  (0.013009)	lr: 0.009822
    DEEPTB INFO    checkpoint saved as nnsk.iter183
    DEEPTB INFO    iteration:184	train_loss: 0.012919  (0.012982)	lr: 0.009822
    DEEPTB INFO    checkpoint saved as nnsk.iter184
    DEEPTB INFO    iteration:185	train_loss: 0.012917  (0.012962)	lr: 0.009822
    DEEPTB INFO    checkpoint saved as nnsk.iter185
    DEEPTB INFO    iteration:186	train_loss: 0.012702  (0.012884)	lr: 0.009822
    DEEPTB INFO    checkpoint saved as nnsk.iter186
    DEEPTB INFO    iteration:187	train_loss: 0.012891  (0.012886)	lr: 0.009822
    DEEPTB INFO    checkpoint saved as nnsk.iter187
    DEEPTB INFO    iteration:188	train_loss: 0.012676  (0.012823)	lr: 0.009822
    DEEPTB INFO    checkpoint saved as nnsk.iter188
    DEEPTB INFO    iteration:189	train_loss: 0.012471  (0.012717)	lr: 0.009822
    DEEPTB INFO    checkpoint saved as nnsk.iter189
    DEEPTB INFO    iteration:190	train_loss: 0.012544  (0.012665)	lr: 0.009822
    DEEPTB INFO    checkpoint saved as nnsk.iter190
    DEEPTB INFO    Epoch 19 summary:	train_loss: 0.012794	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep19
    DEEPTB INFO    iteration:191	train_loss: 0.012649  (0.012660)	lr: 0.009812
    DEEPTB INFO    checkpoint saved as nnsk.iter191
    DEEPTB INFO    iteration:192	train_loss: 0.012577  (0.012636)	lr: 0.009812
    DEEPTB INFO    checkpoint saved as nnsk.iter192
    DEEPTB INFO    iteration:193	train_loss: 0.012307  (0.012537)	lr: 0.009812
    DEEPTB INFO    checkpoint saved as nnsk.iter193
    DEEPTB INFO    iteration:194	train_loss: 0.012312  (0.012469)	lr: 0.009812
    DEEPTB INFO    checkpoint saved as nnsk.iter194
    DEEPTB INFO    iteration:195	train_loss: 0.012305  (0.012420)	lr: 0.009812
    DEEPTB INFO    checkpoint saved as nnsk.iter195
    DEEPTB INFO    iteration:196	train_loss: 0.012306  (0.012386)	lr: 0.009812
    DEEPTB INFO    checkpoint saved as nnsk.iter196
    DEEPTB INFO    iteration:197	train_loss: 0.012117  (0.012305)	lr: 0.009812
    DEEPTB INFO    checkpoint saved as nnsk.iter197
    DEEPTB INFO    iteration:198	train_loss: 0.012237  (0.012285)	lr: 0.009812
    DEEPTB INFO    checkpoint saved as nnsk.iter198
    DEEPTB INFO    iteration:199	train_loss: 0.012349  (0.012304)	lr: 0.009812
    DEEPTB INFO    checkpoint saved as nnsk.iter199
    DEEPTB INFO    iteration:200	train_loss: 0.011955  (0.012199)	lr: 0.009812
    DEEPTB INFO    checkpoint saved as nnsk.iter200
    DEEPTB INFO    Epoch 20 summary:	train_loss: 0.012311	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep20
    DEEPTB INFO    finished training
    DEEPTB INFO    wall time: 240.534 s



```python
# !dptb run band.json -i ./nnskmd100/checkpoint/nnsk.best.pth -o band2 -stu ../data/silicon.vasp
!dptb run band.json -i ./ref_ckpt/nnsk.md100.pth -o band2 -stu ../data/silicon.vasp

# display the band plot:
image_path = f'./band2/results/band.png'
img = mpimg.imread(image_path)
plt.figure(figsize=(10, 8))
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
     
     
    DEEPTB INFO    The ['overlap_param'] are frozen!
    DEEPTB INFO    The ['overlap_param'] are frozen!
    /opt/mamba/lib/python3.10/site-packages/torch/nested/__init__.py:107: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. (Triggered internally at ../aten/src/ATen/NestedTensorImpl.cpp:178.)
      return torch._nested_tensor_from_tensor_list(ts, dtype, None, device, None)
    DEEPTB INFO    KPOINTS  klist: 302 kpoints
    DEEPTB INFO    The eigenvalues are already in data. will use them.
    DEEPTB INFO    Calculating Fermi energy in the case of spin-degeneracy.
    DEEPTB WARNING Fermi level bisection did not converge under tolerance 1e-10 after 55 iterations.
    DEEPTB INFO    q_cal: 8.000000000595051, total_electrons: 8.0, diff q: 5.950511194896535e-10
    DEEPTB INFO    Estimated E_fermi: -4.091686964035034 based on the valence electrons setting nel_atom : {'Si': 4} .
    DEEPTB INFO    Using input Fermi energy: -4.7220 eV (estimated: -4.0917 eV)
    Figure(640x560)
    DEEPTB INFO    band calculation successfully completed.



    
![png](tu3_files/tu3_19_1.png)
    



```python
# !dptb run band_2.json -i ./nnskmd100/checkpoint/nnsk.best.pth -stu ./data/kpathmd100.0/struct.vasp -o  band3
!dptb run band_2.json -i ./ref_ckpt/nnsk.md100.pth -stu ../data/kpathmd100.0/struct.vasp -o  band3

# display the band plot:
image_path = f'./band3/results/band.png'
img = mpimg.imread(image_path)
plt.figure(figsize=(10, 8))
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
     
     
    DEEPTB INFO    The ['overlap_param'] are frozen!
    DEEPTB INFO    The ['overlap_param'] are frozen!
    /opt/mamba/lib/python3.10/site-packages/torch/nested/__init__.py:107: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. (Triggered internally at ../aten/src/ATen/NestedTensorImpl.cpp:178.)
      return torch._nested_tensor_from_tensor_list(ts, dtype, None, device, None)
    DEEPTB INFO    KPOINTS  klist: 354 kpoints
    DEEPTB INFO    The eigenvalues are already in data. will use them.
    DEEPTB INFO    Calculating Fermi energy in the case of spin-degeneracy.
    DEEPTB INFO    Fermi energy converged after 26 iterations.
    DEEPTB INFO    q_cal: 32.000000000191456, total_electrons: 32.0, diff q: 1.91455740150559e-10
    DEEPTB INFO    Estimated E_fermi: -4.106476799620408 based on the valence electrons setting nel_atom : {'Si': 4} .
    DEEPTB INFO    Using Fermi energy: -4.1065 eV (matches estimated value)
    Figure(640x560)
    DEEPTB INFO    band calculation successfully completed.



    
![png](tu3_files/tu3_20_1.png)
    


#### **2.4 Training Environment Correction**

The **DeePTB-SK** module provides powerful environment-dependent modeling with symmetry-preserving neural networks. Based on the previously constructed `nnsk` model, we can further enhance the TB model's descriptive ability by adding an environment-dependent component to overcome the accuracy limitations imposed by the two-center approximation. The model that incorporates environment dependence into the `nnsk` model is referred to as the `mix` model, and its expression is as follows:
$$
\begin{equation}
h^{\text{env}}_{ll^\prime{\zeta}} =  h_{ll^\prime{\zeta}}(r_{ij}) \times \left[1+\Phi_{ll^\prime\zeta}^{o_i,o_j}\left(r_{ij},\mathcal{D}^{ij}\right)\right]	
\end{equation}
$$
where $\mathcal{D}^{ij}$ is the environment descriptor defined by the `embedding` keyword, and $\Phi_{ll^\prime\zeta}^{o_i,o_j}$ is the neural network that provides the environment correction prediction value.

To define the `mix` correction model, you need to provide the following keywords in the `model_options` section of the training input file:
- `embedding`: The `method` here specifies the form of the atomic environment used in the `dptb` model. In this example, we use the `se2` form of descriptor similar to that used in **DeePMD**. 
- `prediction`: The `method` specifies the prediction method of the model, which is set to `sktb` here. The `neurons` keyword specifies the size of the prediction network.
- `nnsk`: This section is consistent with the content in the `nnsk` model. The `freeze` option should be set to `true`, indicating that the trained SK parameters of the `nnsk` model are fixed, and only the neural network parameters of the environment-dependent part are trained. This fixing is crucial; otherwise, the initialization of the `mix` model may completely destroy the parameters of the `nnsk` model, leading to non-convergence during training.
  
For example:
```json
    "model_options": {
        "embedding":{
            "method": "se2",
            "rs": 2.5,
            "rc": 5.0,
            "radial_net": {
                "neurons": [10,20,30]
            }
        },
        "prediction":{
            "method": "sktb",
            "neurons": [16,16,16]
        },
        "nnsk": {
            ...
            "freeze": true
            ...
        }
    }
```
The complete input content can be found in `input_3.json`. The environment-dependent `mix` model training requires reading the `nnsk` model. We can run:


```python
# !dptb train input_3.json -i ./nnskmd100/checkpoint/nnsk.ep20.pth -o ./mix
!dptb train input_3.json -i ./ref_ckpt/nnsk.md100.pth -o ./mix
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
     
     
    DEEPTB WARNING The model options embedding is not defined in checkpoint, set to {'method': 'se2', 'rs': 2.5, 'rc': 5.0, 'radial_net': {'neurons': [10, 20, 30], 'activation': 'tanh', 'if_batch_normalized': False}, 'n_axis': None}.
    DEEPTB WARNING The model options prediction is not defined in checkpoint, set to {'method': 'sktb', 'neurons': [16, 16, 16], 'activation': 'tanh', 'if_batch_normalized': False}.
    DEEPTB WARNING The model option freeze is set to True, but in checkpoint it is ['overlap'], make sure it it correct!
    DEEPTB INFO    ------------------------------------------------------------------
    DEEPTB INFO         Cutoff options:                                            
    DEEPTB INFO                                                                    
    DEEPTB INFO         r_max            : {'Si-Si': 6.24}                         
    DEEPTB INFO         er_max           : 5.0                                     
    DEEPTB INFO         oer_max          : None                                    
    DEEPTB INFO    ------------------------------------------------------------------
    /opt/mamba/lib/python3.10/site-packages/torch/nested/__init__.py:107: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. (Triggered internally at ../aten/src/ATen/NestedTensorImpl.cpp:178.)
      return torch._nested_tensor_from_tensor_list(ts, dtype, None, device, None)
    DEEPTB WARNING The cutoffs in data and model are not checked. be careful!
    DEEPTB WARNING The cutoffs in data and model are not checked. be careful!
    DEEPTB WARNING The push option is not supported in the mixed model. The push option is only supported in the nnsk model.
    DEEPTB INFO    The ['hopping_param', 'overlap_param', 'onsite_param'] are frozen!
    DEEPTB INFO    iteration:1	train_loss: 0.012923  (0.003877)	lr: 0.001
    DEEPTB INFO    checkpoint saved as mix.iter1
    DEEPTB INFO    iteration:2	train_loss: 0.016042  (0.007526)	lr: 0.001
    DEEPTB INFO    checkpoint saved as mix.iter2
    DEEPTB INFO    iteration:3	train_loss: 0.012602  (0.009049)	lr: 0.001
    DEEPTB INFO    checkpoint saved as mix.iter3
    DEEPTB INFO    iteration:4	train_loss: 0.012336  (0.010035)	lr: 0.001
    DEEPTB INFO    checkpoint saved as mix.iter4
    DEEPTB INFO    iteration:5	train_loss: 0.013278  (0.011008)	lr: 0.001
    DEEPTB INFO    checkpoint saved as mix.iter5
    DEEPTB INFO    iteration:6	train_loss: 0.012853  (0.011561)	lr: 0.001
    DEEPTB INFO    checkpoint saved as mix.iter6
    DEEPTB INFO    iteration:7	train_loss: 0.012130  (0.011732)	lr: 0.001
    DEEPTB INFO    checkpoint saved as mix.iter7
    DEEPTB INFO    iteration:8	train_loss: 0.011827  (0.011760)	lr: 0.001
    DEEPTB INFO    checkpoint saved as mix.iter8
    DEEPTB INFO    iteration:9	train_loss: 0.012503  (0.011983)	lr: 0.001
    DEEPTB INFO    checkpoint saved as mix.iter9
    DEEPTB INFO    iteration:10	train_loss: 0.012240  (0.012060)	lr: 0.001
    DEEPTB INFO    checkpoint saved as mix.iter10
    DEEPTB INFO    Epoch 1 summary:	train_loss: 0.012873	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as mix.ep1
    DEEPTB INFO    iteration:11	train_loss: 0.011943  (0.012025)	lr: 0.000999
    DEEPTB INFO    checkpoint saved as mix.iter11
    DEEPTB INFO    iteration:12	train_loss: 0.011564  (0.011887)	lr: 0.000999
    DEEPTB INFO    checkpoint saved as mix.iter12
    DEEPTB INFO    iteration:13	train_loss: 0.011930  (0.011900)	lr: 0.000999
    DEEPTB INFO    checkpoint saved as mix.iter13
    DEEPTB INFO    iteration:14	train_loss: 0.012044  (0.011943)	lr: 0.000999
    DEEPTB INFO    checkpoint saved as mix.iter14
    DEEPTB INFO    iteration:15	train_loss: 0.011880  (0.011924)	lr: 0.000999
    DEEPTB INFO    checkpoint saved as mix.iter15
    DEEPTB INFO    iteration:16	train_loss: 0.011474  (0.011789)	lr: 0.000999
    DEEPTB INFO    checkpoint saved as mix.iter16
    DEEPTB INFO    iteration:17	train_loss: 0.011509  (0.011705)	lr: 0.000999
    DEEPTB INFO    checkpoint saved as mix.iter17
    DEEPTB INFO    iteration:18	train_loss: 0.011747  (0.011718)	lr: 0.000999
    DEEPTB INFO    checkpoint saved as mix.iter18
    DEEPTB INFO    iteration:19	train_loss: 0.011595  (0.011681)	lr: 0.000999
    DEEPTB INFO    checkpoint saved as mix.iter19
    DEEPTB INFO    iteration:20	train_loss: 0.011615  (0.011661)	lr: 0.000999
    DEEPTB INFO    checkpoint saved as mix.iter20
    DEEPTB INFO    Epoch 2 summary:	train_loss: 0.011730	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as mix.ep2
    DEEPTB INFO    iteration:21	train_loss: 0.011626  (0.011651)	lr: 0.000998
    DEEPTB INFO    checkpoint saved as mix.iter21
    DEEPTB INFO    iteration:22	train_loss: 0.011402  (0.011576)	lr: 0.000998
    DEEPTB INFO    checkpoint saved as mix.iter22
    DEEPTB INFO    iteration:23	train_loss: 0.011555  (0.011570)	lr: 0.000998
    DEEPTB INFO    checkpoint saved as mix.iter23
    DEEPTB INFO    iteration:24	train_loss: 0.011458  (0.011536)	lr: 0.000998
    DEEPTB INFO    checkpoint saved as mix.iter24
    DEEPTB INFO    iteration:25	train_loss: 0.011385  (0.011491)	lr: 0.000998
    DEEPTB INFO    checkpoint saved as mix.iter25
    DEEPTB INFO    iteration:26	train_loss: 0.011412  (0.011467)	lr: 0.000998
    DEEPTB INFO    checkpoint saved as mix.iter26
    DEEPTB INFO    iteration:27	train_loss: 0.011503  (0.011478)	lr: 0.000998
    DEEPTB INFO    checkpoint saved as mix.iter27
    DEEPTB INFO    iteration:28	train_loss: 0.011466  (0.011474)	lr: 0.000998
    DEEPTB INFO    checkpoint saved as mix.iter28
    DEEPTB INFO    iteration:29	train_loss: 0.011669  (0.011533)	lr: 0.000998
    DEEPTB INFO    checkpoint saved as mix.iter29
    DEEPTB INFO    iteration:30	train_loss: 0.011285  (0.011459)	lr: 0.000998
    DEEPTB INFO    checkpoint saved as mix.iter30
    DEEPTB INFO    Epoch 3 summary:	train_loss: 0.011476	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as mix.ep3
    DEEPTB INFO    iteration:31	train_loss: 0.011497  (0.011470)	lr: 0.000997
    DEEPTB INFO    checkpoint saved as mix.iter31
    DEEPTB INFO    iteration:32	train_loss: 0.011230  (0.011398)	lr: 0.000997
    DEEPTB INFO    checkpoint saved as mix.iter32
    DEEPTB INFO    iteration:33	train_loss: 0.011447  (0.011413)	lr: 0.000997
    DEEPTB INFO    checkpoint saved as mix.iter33
    DEEPTB INFO    iteration:34	train_loss: 0.011389  (0.011406)	lr: 0.000997
    DEEPTB INFO    checkpoint saved as mix.iter34
    DEEPTB INFO    iteration:35	train_loss: 0.011500  (0.011434)	lr: 0.000997
    DEEPTB INFO    checkpoint saved as mix.iter35
    DEEPTB INFO    iteration:36	train_loss: 0.011169  (0.011354)	lr: 0.000997
    DEEPTB INFO    checkpoint saved as mix.iter36
    DEEPTB INFO    iteration:37	train_loss: 0.011279  (0.011332)	lr: 0.000997
    DEEPTB INFO    checkpoint saved as mix.iter37
    DEEPTB INFO    iteration:38	train_loss: 0.011217  (0.011297)	lr: 0.000997
    DEEPTB INFO    checkpoint saved as mix.iter38
    DEEPTB INFO    iteration:39	train_loss: 0.011352  (0.011314)	lr: 0.000997
    DEEPTB INFO    checkpoint saved as mix.iter39
    DEEPTB INFO    iteration:40	train_loss: 0.011480  (0.011364)	lr: 0.000997
    DEEPTB INFO    checkpoint saved as mix.iter40
    DEEPTB INFO    Epoch 4 summary:	train_loss: 0.011356	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as mix.ep4
    DEEPTB INFO    iteration:41	train_loss: 0.011409  (0.011377)	lr: 0.000996
    DEEPTB INFO    checkpoint saved as mix.iter41
    DEEPTB INFO    iteration:42	train_loss: 0.011185  (0.011320)	lr: 0.000996
    DEEPTB INFO    checkpoint saved as mix.iter42
    DEEPTB INFO    iteration:43	train_loss: 0.011341  (0.011326)	lr: 0.000996
    DEEPTB INFO    checkpoint saved as mix.iter43
    DEEPTB INFO    iteration:44	train_loss: 0.011174  (0.011280)	lr: 0.000996
    DEEPTB INFO    checkpoint saved as mix.iter44
    DEEPTB INFO    iteration:45	train_loss: 0.011342  (0.011299)	lr: 0.000996
    DEEPTB INFO    checkpoint saved as mix.iter45
    DEEPTB INFO    iteration:46	train_loss: 0.011222  (0.011276)	lr: 0.000996
    DEEPTB INFO    checkpoint saved as mix.iter46
    DEEPTB INFO    iteration:47	train_loss: 0.011125  (0.011230)	lr: 0.000996
    DEEPTB INFO    checkpoint saved as mix.iter47
    DEEPTB INFO    iteration:48	train_loss: 0.011265  (0.011241)	lr: 0.000996
    DEEPTB INFO    checkpoint saved as mix.iter48
    DEEPTB INFO    iteration:49	train_loss: 0.011438  (0.011300)	lr: 0.000996
    DEEPTB INFO    checkpoint saved as mix.iter49
    DEEPTB INFO    iteration:50	train_loss: 0.011371  (0.011321)	lr: 0.000996
    DEEPTB INFO    checkpoint saved as mix.iter50
    DEEPTB INFO    Epoch 5 summary:	train_loss: 0.011287	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as mix.ep5
    DEEPTB INFO    iteration:51	train_loss: 0.011114  (0.011259)	lr: 0.000995
    DEEPTB INFO    checkpoint saved as mix.iter51
    DEEPTB INFO    iteration:52	train_loss: 0.011362  (0.011290)	lr: 0.000995
    DEEPTB INFO    checkpoint saved as mix.iter52
    DEEPTB INFO    iteration:53	train_loss: 0.011376  (0.011316)	lr: 0.000995
    DEEPTB INFO    checkpoint saved as mix.iter53
    DEEPTB INFO    iteration:54	train_loss: 0.011208  (0.011284)	lr: 0.000995
    DEEPTB INFO    checkpoint saved as mix.iter54
    DEEPTB INFO    iteration:55	train_loss: 0.011114  (0.011233)	lr: 0.000995
    DEEPTB INFO    checkpoint saved as mix.iter55
    DEEPTB INFO    iteration:56	train_loss: 0.011220  (0.011229)	lr: 0.000995
    DEEPTB INFO    checkpoint saved as mix.iter56
    DEEPTB INFO    iteration:57	train_loss: 0.011264  (0.011239)	lr: 0.000995
    DEEPTB INFO    checkpoint saved as mix.iter57
    DEEPTB INFO    iteration:58	train_loss: 0.011359  (0.011275)	lr: 0.000995
    DEEPTB INFO    checkpoint saved as mix.iter58
    DEEPTB INFO    iteration:59	train_loss: 0.011103  (0.011224)	lr: 0.000995
    DEEPTB INFO    checkpoint saved as mix.iter59
    DEEPTB INFO    iteration:60	train_loss: 0.011354  (0.011263)	lr: 0.000995
    DEEPTB INFO    checkpoint saved as mix.iter60
    DEEPTB INFO    Epoch 6 summary:	train_loss: 0.011248	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as mix.ep6
    DEEPTB INFO    iteration:61	train_loss: 0.011180  (0.011238)	lr: 0.000994
    DEEPTB INFO    checkpoint saved as mix.iter61
    DEEPTB INFO    iteration:62	train_loss: 0.011182  (0.011221)	lr: 0.000994
    DEEPTB INFO    checkpoint saved as mix.iter62
    DEEPTB INFO    iteration:63	train_loss: 0.011087  (0.011181)	lr: 0.000994
    DEEPTB INFO    checkpoint saved as mix.iter63
    DEEPTB INFO    iteration:64	train_loss: 0.011344  (0.011230)	lr: 0.000994
    DEEPTB INFO    checkpoint saved as mix.iter64
    DEEPTB INFO    iteration:65	train_loss: 0.011371  (0.011272)	lr: 0.000994
    DEEPTB INFO    checkpoint saved as mix.iter65
    DEEPTB INFO    iteration:66	train_loss: 0.011072  (0.011212)	lr: 0.000994
    DEEPTB INFO    checkpoint saved as mix.iter66
    DEEPTB INFO    iteration:67	train_loss: 0.011041  (0.011161)	lr: 0.000994
    DEEPTB INFO    checkpoint saved as mix.iter67
    DEEPTB INFO    iteration:68	train_loss: 0.011237  (0.011184)	lr: 0.000994
    DEEPTB INFO    checkpoint saved as mix.iter68
    DEEPTB INFO    iteration:69	train_loss: 0.011252  (0.011204)	lr: 0.000994
    DEEPTB INFO    checkpoint saved as mix.iter69
    DEEPTB INFO    iteration:70	train_loss: 0.011297  (0.011232)	lr: 0.000994
    DEEPTB INFO    checkpoint saved as mix.iter70
    DEEPTB INFO    Epoch 7 summary:	train_loss: 0.011206	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as mix.ep7
    DEEPTB INFO    iteration:71	train_loss: 0.011078  (0.011186)	lr: 0.000993
    DEEPTB INFO    checkpoint saved as mix.iter71
    DEEPTB INFO    iteration:72	train_loss: 0.011302  (0.011221)	lr: 0.000993
    DEEPTB INFO    checkpoint saved as mix.iter72
    DEEPTB INFO    iteration:73	train_loss: 0.011138  (0.011196)	lr: 0.000993
    DEEPTB INFO    checkpoint saved as mix.iter73
    DEEPTB INFO    iteration:74	train_loss: 0.011052  (0.011153)	lr: 0.000993
    DEEPTB INFO    checkpoint saved as mix.iter74
    DEEPTB INFO    iteration:75	train_loss: 0.011288  (0.011193)	lr: 0.000993
    DEEPTB INFO    checkpoint saved as mix.iter75
    DEEPTB INFO    iteration:76	train_loss: 0.011017  (0.011141)	lr: 0.000993
    DEEPTB INFO    checkpoint saved as mix.iter76
    DEEPTB INFO    iteration:77	train_loss: 0.011215  (0.011163)	lr: 0.000993
    DEEPTB INFO    checkpoint saved as mix.iter77
    DEEPTB INFO    iteration:78	train_loss: 0.011316  (0.011209)	lr: 0.000993
    DEEPTB INFO    checkpoint saved as mix.iter78
    DEEPTB INFO    iteration:79	train_loss: 0.011098  (0.011176)	lr: 0.000993
    DEEPTB INFO    checkpoint saved as mix.iter79
    DEEPTB INFO    iteration:80	train_loss: 0.011164  (0.011172)	lr: 0.000993
    DEEPTB INFO    checkpoint saved as mix.iter80
    DEEPTB INFO    Epoch 8 summary:	train_loss: 0.011167	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as mix.ep8
    DEEPTB INFO    iteration:81	train_loss: 0.011098  (0.011150)	lr: 0.000992
    DEEPTB INFO    checkpoint saved as mix.iter81
    DEEPTB INFO    iteration:82	train_loss: 0.011161  (0.011153)	lr: 0.000992
    DEEPTB INFO    checkpoint saved as mix.iter82
    DEEPTB INFO    iteration:83	train_loss: 0.011253  (0.011183)	lr: 0.000992
    DEEPTB INFO    checkpoint saved as mix.iter83
    DEEPTB INFO    iteration:84	train_loss: 0.011005  (0.011130)	lr: 0.000992
    DEEPTB INFO    checkpoint saved as mix.iter84
    DEEPTB INFO    iteration:85	train_loss: 0.011172  (0.011142)	lr: 0.000992
    DEEPTB INFO    checkpoint saved as mix.iter85
    DEEPTB INFO    iteration:86	train_loss: 0.011253  (0.011176)	lr: 0.000992
    DEEPTB INFO    checkpoint saved as mix.iter86
    DEEPTB INFO    iteration:87	train_loss: 0.010945  (0.011107)	lr: 0.000992
    DEEPTB INFO    checkpoint saved as mix.iter87
    DEEPTB INFO    iteration:88	train_loss: 0.011071  (0.011096)	lr: 0.000992
    DEEPTB INFO    checkpoint saved as mix.iter88
    DEEPTB INFO    iteration:89	train_loss: 0.011236  (0.011138)	lr: 0.000992
    DEEPTB INFO    checkpoint saved as mix.iter89
    DEEPTB INFO    iteration:90	train_loss: 0.010972  (0.011088)	lr: 0.000992
    DEEPTB INFO    checkpoint saved as mix.iter90
    DEEPTB INFO    Epoch 9 summary:	train_loss: 0.011117	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as mix.ep9
    DEEPTB INFO    iteration:91	train_loss: 0.011105  (0.011093)	lr: 0.000991
    DEEPTB INFO    checkpoint saved as mix.iter91
    DEEPTB INFO    iteration:92	train_loss: 0.011192  (0.011123)	lr: 0.000991
    DEEPTB INFO    checkpoint saved as mix.iter92
    DEEPTB INFO    iteration:93	train_loss: 0.010961  (0.011074)	lr: 0.000991
    DEEPTB INFO    checkpoint saved as mix.iter93
    DEEPTB INFO    iteration:94	train_loss: 0.010902  (0.011022)	lr: 0.000991
    DEEPTB INFO    checkpoint saved as mix.iter94
    DEEPTB INFO    iteration:95	train_loss: 0.011117  (0.011051)	lr: 0.000991
    DEEPTB INFO    checkpoint saved as mix.iter95
    DEEPTB INFO    iteration:96	train_loss: 0.011003  (0.011036)	lr: 0.000991
    DEEPTB INFO    checkpoint saved as mix.iter96
    DEEPTB INFO    iteration:97	train_loss: 0.010937  (0.011006)	lr: 0.000991
    DEEPTB INFO    checkpoint saved as mix.iter97
    DEEPTB INFO    iteration:98	train_loss: 0.011007  (0.011007)	lr: 0.000991
    DEEPTB INFO    checkpoint saved as mix.iter98
    DEEPTB INFO    iteration:99	train_loss: 0.011148  (0.011049)	lr: 0.000991
    DEEPTB INFO    checkpoint saved as mix.iter99
    DEEPTB INFO    iteration:100	train_loss: 0.011168  (0.011085)	lr: 0.000991
    DEEPTB INFO    checkpoint saved as mix.iter100
    DEEPTB INFO    Epoch 10 summary:	train_loss: 0.011054	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as mix.ep10
    DEEPTB INFO    iteration:101	train_loss: 0.010975  (0.011052)	lr: 0.00099 
    DEEPTB INFO    checkpoint saved as mix.iter101
    DEEPTB INFO    iteration:102	train_loss: 0.011060  (0.011054)	lr: 0.00099 
    DEEPTB INFO    checkpoint saved as mix.iter102
    DEEPTB INFO    iteration:103	train_loss: 0.010903  (0.011009)	lr: 0.00099 
    DEEPTB INFO    checkpoint saved as mix.iter103
    DEEPTB INFO    iteration:104	train_loss: 0.010864  (0.010966)	lr: 0.00099 
    DEEPTB INFO    checkpoint saved as mix.iter104
    DEEPTB INFO    iteration:105	train_loss: 0.011036  (0.010987)	lr: 0.00099 
    DEEPTB INFO    checkpoint saved as mix.iter105
    DEEPTB INFO    iteration:106	train_loss: 0.010800  (0.010931)	lr: 0.00099 
    DEEPTB INFO    checkpoint saved as mix.iter106
    DEEPTB INFO    iteration:107	train_loss: 0.011094  (0.010980)	lr: 0.00099 
    DEEPTB INFO    checkpoint saved as mix.iter107
    DEEPTB INFO    iteration:108	train_loss: 0.011089  (0.011012)	lr: 0.00099 
    DEEPTB INFO    checkpoint saved as mix.iter108
    DEEPTB INFO    iteration:109	train_loss: 0.011145  (0.011052)	lr: 0.00099 
    DEEPTB INFO    checkpoint saved as mix.iter109
    DEEPTB INFO    iteration:110	train_loss: 0.010884  (0.011002)	lr: 0.00099 
    DEEPTB INFO    checkpoint saved as mix.iter110
    DEEPTB INFO    Epoch 11 summary:	train_loss: 0.010985	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as mix.ep11
    DEEPTB INFO    iteration:111	train_loss: 0.010920  (0.010977)	lr: 0.0009891
    DEEPTB INFO    checkpoint saved as mix.iter111
    DEEPTB INFO    iteration:112	train_loss: 0.010797  (0.010923)	lr: 0.0009891
    DEEPTB INFO    checkpoint saved as mix.iter112
    DEEPTB INFO    iteration:113	train_loss: 0.011136  (0.010987)	lr: 0.0009891
    DEEPTB INFO    checkpoint saved as mix.iter113
    DEEPTB INFO    iteration:114	train_loss: 0.010923  (0.010968)	lr: 0.0009891
    DEEPTB INFO    checkpoint saved as mix.iter114
    DEEPTB INFO    iteration:115	train_loss: 0.011238  (0.011049)	lr: 0.0009891
    DEEPTB INFO    checkpoint saved as mix.iter115
    DEEPTB INFO    iteration:116	train_loss: 0.011008  (0.011037)	lr: 0.0009891
    DEEPTB INFO    checkpoint saved as mix.iter116
    DEEPTB INFO    iteration:117	train_loss: 0.011013  (0.011029)	lr: 0.0009891
    DEEPTB INFO    checkpoint saved as mix.iter117
    DEEPTB INFO    iteration:118	train_loss: 0.010946  (0.011004)	lr: 0.0009891
    DEEPTB INFO    checkpoint saved as mix.iter118
    DEEPTB INFO    iteration:119	train_loss: 0.010842  (0.010956)	lr: 0.0009891
    DEEPTB INFO    checkpoint saved as mix.iter119
    DEEPTB INFO    iteration:120	train_loss: 0.010863  (0.010928)	lr: 0.0009891
    DEEPTB INFO    checkpoint saved as mix.iter120
    DEEPTB INFO    Epoch 12 summary:	train_loss: 0.010969	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as mix.ep12
    DEEPTB INFO    iteration:121	train_loss: 0.010825  (0.010897)	lr: 0.0009881
    DEEPTB INFO    checkpoint saved as mix.iter121
    DEEPTB INFO    iteration:122	train_loss: 0.010726  (0.010846)	lr: 0.0009881
    DEEPTB INFO    checkpoint saved as mix.iter122
    DEEPTB INFO    iteration:123	train_loss: 0.011102  (0.010923)	lr: 0.0009881
    DEEPTB INFO    checkpoint saved as mix.iter123
    DEEPTB INFO    iteration:124	train_loss: 0.010709  (0.010858)	lr: 0.0009881
    DEEPTB INFO    checkpoint saved as mix.iter124
    DEEPTB INFO    iteration:125	train_loss: 0.010991  (0.010898)	lr: 0.0009881
    DEEPTB INFO    checkpoint saved as mix.iter125
    DEEPTB INFO    iteration:126	train_loss: 0.010861  (0.010887)	lr: 0.0009881
    DEEPTB INFO    checkpoint saved as mix.iter126
    DEEPTB INFO    iteration:127	train_loss: 0.010845  (0.010874)	lr: 0.0009881
    DEEPTB INFO    checkpoint saved as mix.iter127
    DEEPTB INFO    iteration:128	train_loss: 0.010805  (0.010854)	lr: 0.0009881
    DEEPTB INFO    checkpoint saved as mix.iter128
    DEEPTB INFO    iteration:129	train_loss: 0.010917  (0.010873)	lr: 0.0009881
    DEEPTB INFO    checkpoint saved as mix.iter129
    DEEPTB INFO    iteration:130	train_loss: 0.010734  (0.010831)	lr: 0.0009881
    DEEPTB INFO    checkpoint saved as mix.iter130
    DEEPTB INFO    Epoch 13 summary:	train_loss: 0.010851	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as mix.ep13
    DEEPTB INFO    iteration:131	train_loss: 0.011046  (0.010895)	lr: 0.0009871
    DEEPTB INFO    checkpoint saved as mix.iter131
    DEEPTB INFO    iteration:132	train_loss: 0.010795  (0.010865)	lr: 0.0009871
    DEEPTB INFO    checkpoint saved as mix.iter132
    DEEPTB INFO    iteration:133	train_loss: 0.010740  (0.010828)	lr: 0.0009871
    DEEPTB INFO    checkpoint saved as mix.iter133
    DEEPTB INFO    iteration:134	train_loss: 0.010633  (0.010769)	lr: 0.0009871
    DEEPTB INFO    checkpoint saved as mix.iter134
    DEEPTB INFO    iteration:135	train_loss: 0.010657  (0.010735)	lr: 0.0009871
    DEEPTB INFO    checkpoint saved as mix.iter135
    DEEPTB INFO    iteration:136	train_loss: 0.010808  (0.010757)	lr: 0.0009871
    DEEPTB INFO    checkpoint saved as mix.iter136
    DEEPTB INFO    iteration:137	train_loss: 0.010827  (0.010778)	lr: 0.0009871
    DEEPTB INFO    checkpoint saved as mix.iter137
    DEEPTB INFO    iteration:138	train_loss: 0.010544  (0.010708)	lr: 0.0009871
    DEEPTB INFO    checkpoint saved as mix.iter138
    DEEPTB INFO    iteration:139	train_loss: 0.010805  (0.010737)	lr: 0.0009871
    DEEPTB INFO    checkpoint saved as mix.iter139
    DEEPTB INFO    iteration:140	train_loss: 0.010635  (0.010706)	lr: 0.0009871
    DEEPTB INFO    checkpoint saved as mix.iter140
    DEEPTB INFO    Epoch 14 summary:	train_loss: 0.010749	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as mix.ep14
    DEEPTB INFO    iteration:141	train_loss: 0.010789  (0.010731)	lr: 0.0009861
    DEEPTB INFO    checkpoint saved as mix.iter141
    DEEPTB INFO    iteration:142	train_loss: 0.010560  (0.010680)	lr: 0.0009861
    DEEPTB INFO    checkpoint saved as mix.iter142
    DEEPTB INFO    iteration:143	train_loss: 0.010803  (0.010717)	lr: 0.0009861
    DEEPTB INFO    checkpoint saved as mix.iter143
    DEEPTB INFO    iteration:144	train_loss: 0.010519  (0.010657)	lr: 0.0009861
    DEEPTB INFO    checkpoint saved as mix.iter144
    DEEPTB INFO    iteration:145	train_loss: 0.010736  (0.010681)	lr: 0.0009861
    DEEPTB INFO    checkpoint saved as mix.iter145
    DEEPTB INFO    iteration:146	train_loss: 0.010596  (0.010655)	lr: 0.0009861
    DEEPTB INFO    checkpoint saved as mix.iter146
    DEEPTB INFO    iteration:147	train_loss: 0.010711  (0.010672)	lr: 0.0009861
    DEEPTB INFO    checkpoint saved as mix.iter147
    DEEPTB INFO    iteration:148	train_loss: 0.010727  (0.010689)	lr: 0.0009861
    DEEPTB INFO    checkpoint saved as mix.iter148
    DEEPTB INFO    iteration:149	train_loss: 0.010689  (0.010689)	lr: 0.0009861
    DEEPTB INFO    checkpoint saved as mix.iter149
    DEEPTB INFO    iteration:150	train_loss: 0.010757  (0.010709)	lr: 0.0009861
    DEEPTB INFO    checkpoint saved as mix.iter150
    DEEPTB INFO    Epoch 15 summary:	train_loss: 0.010689	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as mix.ep15
    DEEPTB INFO    iteration:151	train_loss: 0.010646  (0.010690)	lr: 0.0009851
    DEEPTB INFO    checkpoint saved as mix.iter151
    DEEPTB INFO    iteration:152	train_loss: 0.010646  (0.010677)	lr: 0.0009851
    DEEPTB INFO    checkpoint saved as mix.iter152
    DEEPTB INFO    iteration:153	train_loss: 0.010739  (0.010696)	lr: 0.0009851
    DEEPTB INFO    checkpoint saved as mix.iter153
    DEEPTB INFO    iteration:154	train_loss: 0.010591  (0.010664)	lr: 0.0009851
    DEEPTB INFO    checkpoint saved as mix.iter154
    DEEPTB INFO    iteration:155	train_loss: 0.010420  (0.010591)	lr: 0.0009851
    DEEPTB INFO    checkpoint saved as mix.iter155
    DEEPTB INFO    iteration:156	train_loss: 0.010508  (0.010566)	lr: 0.0009851
    DEEPTB INFO    checkpoint saved as mix.iter156
    DEEPTB INFO    iteration:157	train_loss: 0.010529  (0.010555)	lr: 0.0009851
    DEEPTB INFO    checkpoint saved as mix.iter157
    DEEPTB INFO    iteration:158	train_loss: 0.010728  (0.010607)	lr: 0.0009851
    DEEPTB INFO    checkpoint saved as mix.iter158
    DEEPTB INFO    iteration:159	train_loss: 0.010432  (0.010554)	lr: 0.0009851
    DEEPTB INFO    checkpoint saved as mix.iter159
    DEEPTB INFO    iteration:160	train_loss: 0.010824  (0.010635)	lr: 0.0009851
    DEEPTB INFO    checkpoint saved as mix.iter160
    DEEPTB INFO    Epoch 16 summary:	train_loss: 0.010606	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as mix.ep16
    DEEPTB INFO    iteration:161	train_loss: 0.010781  (0.010679)	lr: 0.0009841
    DEEPTB INFO    checkpoint saved as mix.iter161
    DEEPTB INFO    iteration:162	train_loss: 0.010851  (0.010730)	lr: 0.0009841
    DEEPTB INFO    checkpoint saved as mix.iter162
    DEEPTB INFO    iteration:163	train_loss: 0.010366  (0.010621)	lr: 0.0009841
    DEEPTB INFO    checkpoint saved as mix.iter163
    DEEPTB INFO    iteration:164	train_loss: 0.010652  (0.010630)	lr: 0.0009841
    DEEPTB INFO    checkpoint saved as mix.iter164
    DEEPTB INFO    iteration:165	train_loss: 0.010539  (0.010603)	lr: 0.0009841
    DEEPTB INFO    checkpoint saved as mix.iter165
    DEEPTB INFO    iteration:166	train_loss: 0.010620  (0.010608)	lr: 0.0009841
    DEEPTB INFO    checkpoint saved as mix.iter166
    DEEPTB INFO    iteration:167	train_loss: 0.010762  (0.010654)	lr: 0.0009841
    DEEPTB INFO    checkpoint saved as mix.iter167
    DEEPTB INFO    iteration:168	train_loss: 0.010666  (0.010658)	lr: 0.0009841
    DEEPTB INFO    checkpoint saved as mix.iter168
    DEEPTB INFO    iteration:169	train_loss: 0.010456  (0.010597)	lr: 0.0009841
    DEEPTB INFO    checkpoint saved as mix.iter169
    DEEPTB INFO    iteration:170	train_loss: 0.010815  (0.010662)	lr: 0.0009841
    DEEPTB INFO    checkpoint saved as mix.iter170
    DEEPTB INFO    Epoch 17 summary:	train_loss: 0.010651	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    iteration:171	train_loss: 0.010799  (0.010703)	lr: 0.0009831
    DEEPTB INFO    checkpoint saved as mix.iter171
    DEEPTB INFO    iteration:172	train_loss: 0.010466  (0.010632)	lr: 0.0009831
    DEEPTB INFO    checkpoint saved as mix.iter172
    DEEPTB INFO    iteration:173	train_loss: 0.011049  (0.010757)	lr: 0.0009831
    DEEPTB INFO    checkpoint saved as mix.iter173
    DEEPTB INFO    iteration:174	train_loss: 0.010644  (0.010723)	lr: 0.0009831
    DEEPTB INFO    checkpoint saved as mix.iter174
    DEEPTB INFO    iteration:175	train_loss: 0.010827  (0.010755)	lr: 0.0009831
    DEEPTB INFO    checkpoint saved as mix.iter175
    DEEPTB INFO    iteration:176	train_loss: 0.010995  (0.010827)	lr: 0.0009831
    DEEPTB INFO    checkpoint saved as mix.iter176
    DEEPTB INFO    iteration:177	train_loss: 0.010421  (0.010705)	lr: 0.0009831
    DEEPTB INFO    checkpoint saved as mix.iter177
    DEEPTB INFO    iteration:178	train_loss: 0.010994  (0.010792)	lr: 0.0009831
    DEEPTB INFO    checkpoint saved as mix.iter178
    DEEPTB INFO    iteration:179	train_loss: 0.010644  (0.010747)	lr: 0.0009831
    DEEPTB INFO    checkpoint saved as mix.iter179
    DEEPTB INFO    iteration:180	train_loss: 0.010949  (0.010808)	lr: 0.0009831
    DEEPTB INFO    checkpoint saved as mix.iter180
    DEEPTB INFO    Epoch 18 summary:	train_loss: 0.010779	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    iteration:181	train_loss: 0.010564  (0.010735)	lr: 0.0009822
    DEEPTB INFO    checkpoint saved as mix.iter181
    DEEPTB INFO    iteration:182	train_loss: 0.011069  (0.010835)	lr: 0.0009822
    DEEPTB INFO    checkpoint saved as mix.iter182
    DEEPTB INFO    iteration:183	train_loss: 0.010779  (0.010818)	lr: 0.0009822
    DEEPTB INFO    checkpoint saved as mix.iter183
    DEEPTB INFO    iteration:184	train_loss: 0.010805  (0.010814)	lr: 0.0009822
    DEEPTB INFO    checkpoint saved as mix.iter184
    DEEPTB INFO    iteration:185	train_loss: 0.010968  (0.010860)	lr: 0.0009822
    DEEPTB INFO    checkpoint saved as mix.iter185
    DEEPTB INFO    iteration:186	train_loss: 0.010610  (0.010785)	lr: 0.0009822
    DEEPTB INFO    checkpoint saved as mix.iter186
    DEEPTB INFO    iteration:187	train_loss: 0.010745  (0.010773)	lr: 0.0009822
    DEEPTB INFO    checkpoint saved as mix.iter187
    DEEPTB INFO    iteration:188	train_loss: 0.010812  (0.010785)	lr: 0.0009822
    DEEPTB INFO    checkpoint saved as mix.iter188
    DEEPTB INFO    iteration:189	train_loss: 0.010453  (0.010685)	lr: 0.0009822
    DEEPTB INFO    checkpoint saved as mix.iter189
    DEEPTB INFO    iteration:190	train_loss: 0.010759  (0.010707)	lr: 0.0009822
    DEEPTB INFO    checkpoint saved as mix.iter190
    DEEPTB INFO    Epoch 19 summary:	train_loss: 0.010756	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    iteration:191	train_loss: 0.010650  (0.010690)	lr: 0.0009812
    DEEPTB INFO    checkpoint saved as mix.iter191
    DEEPTB INFO    iteration:192	train_loss: 0.010994  (0.010781)	lr: 0.0009812
    DEEPTB INFO    checkpoint saved as mix.iter192
    DEEPTB INFO    iteration:193	train_loss: 0.010403  (0.010668)	lr: 0.0009812
    DEEPTB INFO    checkpoint saved as mix.iter193
    DEEPTB INFO    iteration:194	train_loss: 0.010698  (0.010677)	lr: 0.0009812
    DEEPTB INFO    checkpoint saved as mix.iter194
    DEEPTB INFO    iteration:195	train_loss: 0.010413  (0.010598)	lr: 0.0009812
    DEEPTB INFO    checkpoint saved as mix.iter195
    DEEPTB INFO    iteration:196	train_loss: 0.010840  (0.010671)	lr: 0.0009812
    DEEPTB INFO    checkpoint saved as mix.iter196
    DEEPTB INFO    iteration:197	train_loss: 0.010360  (0.010577)	lr: 0.0009812
    DEEPTB INFO    checkpoint saved as mix.iter197
    DEEPTB INFO    iteration:198	train_loss: 0.010743  (0.010627)	lr: 0.0009812
    DEEPTB INFO    checkpoint saved as mix.iter198
    DEEPTB INFO    iteration:199	train_loss: 0.010700  (0.010649)	lr: 0.0009812
    DEEPTB INFO    checkpoint saved as mix.iter199
    DEEPTB INFO    iteration:200	train_loss: 0.010591  (0.010632)	lr: 0.0009812
    DEEPTB INFO    checkpoint saved as mix.iter200
    DEEPTB INFO    Epoch 20 summary:	train_loss: 0.010639	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    finished training
    DEEPTB INFO    wall time: 241.340 s



```python
# !dptb run band_2.json -i ./mix/checkpoint/mix.best.pth -stu ./data/kpathmd100.0/struct.vasp -o  band4
!dptb run band_2.json -i ./ref_ckpt/mix.md100.pth -stu ../data/kpathmd100.0/struct.vasp -o  band4

# display the band plot:
image_path = f'./band4/results/band.png'
img = mpimg.imread(image_path)
plt.figure(figsize=(10, 8))
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
     
     
    DEEPTB INFO    The ['hopping_param', 'overlap_param', 'onsite_param'] are frozen!
    /opt/mamba/lib/python3.10/site-packages/torch/nested/__init__.py:107: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. (Triggered internally at ../aten/src/ATen/NestedTensorImpl.cpp:178.)
      return torch._nested_tensor_from_tensor_list(ts, dtype, None, device, None)
    DEEPTB INFO    KPOINTS  klist: 354 kpoints
    DEEPTB INFO    The eigenvalues are already in data. will use them.
    DEEPTB INFO    Calculating Fermi energy in the case of spin-degeneracy.
    DEEPTB WARNING Fermi level bisection did not converge under tolerance 1e-10 after 55 iterations.
    DEEPTB INFO    q_cal: 31.999999998985263, total_electrons: 32.0, diff q: 1.0147367390800355e-09
    DEEPTB INFO    Estimated E_fermi: -4.093235731124878 based on the valence electrons setting nel_atom : {'Si': 4} .
    DEEPTB INFO    Using input Fermi energy: -4.1065 eV (estimated: -4.0932 eV)
    Figure(640x560)
    DEEPTB INFO    band calculation successfully completed.



    
![png](tu3_files/tu3_23_1.png)
    


<div style="color:black; background-color:#FFF3E9; border: 1px solid #FFE0C3; border-radius: 10px; margin-bottom:1rem">
    <p style="margin:1rem; padding-left: 1rem; line-height: 2.5;">
        Author: <a style="font-weight:normal" href="mailto:guqq@ustc.edu.cn">Gu, Qiangqiang : guqq@ustc.edu.cn</a>
    </p>
    <p style="margin:1rem; padding-left: 1rem; line-height: 2.5;">
        Thank you for reading!
    </p>
</div>


