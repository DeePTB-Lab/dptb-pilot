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
    ...
    ...
    ...
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
    ...
    ...
    ...
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
    ...
    ...
    ...
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


