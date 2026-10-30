# DeePTB Tutorial 2: data preparation for deeptb-sk model  [v2.2]

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
- **v2.0-2.1**: Added E3 equivariant networks to represent single-electron operators (Hamiltonian, density matrix, and overlap matrix) (DeePTB-E3)
- **v2.2**: Incorporated built-in SK empirical parameters covering commonly used elements across the periodic table

Through these capabilities, DeePTB provides multiple approaches to accelerate electronic structure simulations of materials.

### Learning Objectives

In this tutorial, you will:
1. Learn how to prepare data for DeePTB-SK models
2. Become familiar with the initial training scheme for DeePTB-SK

## 1. Prepare Data for DeePTB-SK Model
This section will show how to prepare data for DeePTB-SK model. The training label of DeePTB-SK model is the energy eigenvalue. This tutorial will show the usage of DFTIO with two DFT software, abacus and VASP.

### Pre-requisites
1. Install DFTIO, see https://github.com/deepmodeling/dftio.git
2. Run DFT Calculation:
   
   For a given structure, perform static calculations using either ABACUS or VASP. Self-consistent or non-self-consistent calculations are both acceptable. This tutorial will use ABACUS and VASP as examples to prepare data. The following sections will describe the usage of DFTIO for both software.
   
You can run `dftio -h` and `dftio <command> -h` to view the help documentation.


```python
!dftio -h
```

    usage: dftio [-h] [-v] {parse,band} ...
    
    dftio is to assist machine learning communities to transcript DFT output into
    a format that is easy to read or used by machine learning models.
    
    options:
      -h, --help     show this help message and exit
      -v, --version  show the dftio's version number and exit
    
    Valid subcommands:
      {parse,band}
        parse        parse dataset from DFT output
        band         plot band for eigenvalues data



```python
!dftio parse -h
```

    usage: dftio parse [-h] [-ll {DEBUG,3,INFO,2,WARNING,1,ERROR,0}]
                       [-lp LOG_PATH] [-m MODE] [-n NUM_WORKERS] [-r ROOT]
                       [-p PREFIX] [-o OUTROOT] [-f FORMAT] [-ham] [-ovp] [-dm]
                       [-eig] [-min BAND_INDEX_MIN]
    
    options:
      -h, --help            show this help message and exit
      -ll {DEBUG,3,INFO,2,WARNING,1,ERROR,0}, --log-level {DEBUG,3,INFO,2,WARNING,1,ERROR,0}
                            set verbosity level by string or number, 0=ERROR,
                            1=WARNING, 2=INFO and 3=DEBUG (default: INFO)
      -lp LOG_PATH, --log-path LOG_PATH
                            set log file to log messages to disk, if not
                            specified, the logs will only be output to console
                            (default: None)
      -m MODE, --mode MODE  The name of the DFT software, currently support
                            abacus/rescu/siesta/gaussian (default: abacus)
      -n NUM_WORKERS, --num_workers NUM_WORKERS
                            The number of workers used to parse the dataset. (For
                            n>1, we use the multiprocessing to accelerate io.)
                            (default: 1)
      -r ROOT, --root ROOT  The root directory of the DFT files. (default: ./)
      -p PREFIX, --prefix PREFIX
                            The prefix of the DFT files under root. (default:
                            frame)
      -o OUTROOT, --outroot OUTROOT
                            The output root directory. (default: ./)
      -f FORMAT, --format FORMAT
                            The output file format, should be dat, ase or lmdb.
                            (default: dat)
      -ham, --hamiltonian   Whether to parse the Hamiltonian matrix. (default:
                            False)
      -ovp, --overlap       Whether to parse the Overlap matrix (default: False)
      -dm, --density_matrix
                            Whether to parse the Density matrix (default: False)
      -eig, --eigenvalue    Whether to parse the kpoints and eigenvalues (default:
                            False)
      -min BAND_INDEX_MIN, --band_index_min BAND_INDEX_MIN
                            The initial band index for eigenvalues to
                            save.(0-band_index_min) bands will be ignored!
                            (default: 0)


###  1.1 ABACUS Case:

The following folder contains the result files from ABACUS calculations. We will demonstrate how to convert the ABACUS calculation data into the training data format for the DeePTB-SK model using a single command.


```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os
workdir='/root/soft/DeePTB/examples/GaAs_io_sk/data'
os.chdir(f"{workdir}")
!tree -L 1 ./
```

    [01;34m./[00m
    ├── [01;34mABACUS[00m
    ├── [01;34mVASP[00m
    ├── [01;34mabc_ase[00m
    └── [01;34mvasp_ase[00m
    
    4 directories, 0 files


The command to process the data is as follows:


```python
! dftio parse -m abacus -r ./ -p ABACUS -f ase -o abc_ase -eig
```

    /opt/mamba/lib/python3.10/site-packages/dpdata/system.py:1106: UserWarning: Data type spins is registered twice; only the newly registered one will be used.
      warnings.warn(
    Parsing the DFT files: 100%|█████████████████████| 1/1 [00:00<00:00, 121.39it/s]


The command above, the meaning of each parameter is as follows:
- `-m`: Specify the software, can be abacus, vasp, etc., default is abacus
- `-r`: root_dir, the root directory of the calculation results
- `-p`: prefix, the prefix of the calculation results, will search all file names: The retrieval rule is: `glob.glob(root + '/*' + prefix + '*')`
- `-f`: format output data format: dat ase lmdb; For eigenvalue training data, dat ase is a common format. lmdb is mainly for training quantum operator matrices.
- `-o`: out_dir output folder, root_dir/out_dir
- `-eig`: Whether to output eigenvalues, add -eig to set it to True
  
The following command provides a preview of the corresponding eigenvalue band structure, which is convenient for analysis:


```python
!dftio band -r ./abc_ase/AsGa.0 -f ase 

# display the band plot:
image_path = f'{workdir}/abc_ase/AsGa.0/band_structure.png'
img = mpimg.imread(image_path)
plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.axis('off')
plt.show()

```

    Figure(640x480)



    
![png](tu2_files/tu2_11_1.png)
    


The plotting command:
```python
dftio band -r ./abc_ase/AsGa.0 -f ase 
```
The meaning of the parameters is as follows:
- `-r` : root_dir, the root directory of the calculation results. Batch plotting is not supported here, so we directly specify the folder containing the parsed data from the previous step without using the prefix and search folder mode.
- `-f` : format, corresponding to the output format during parsing, ase or dat

### Band structure analysis

For the above band structure, we can see that some bands belong to core orbitals. We do not need these bands for training the TB model. We can remove these bands by adding an extra parameter to the above command.

We first check how many bands we want to remove. We can visualize the band structure to see this:


```python
!dftio band -r ./abc_ase/AsGa.0 -f ase -min 5

# display the band plot:

image_path = f'{workdir}/abc_ase/AsGa.0/band_structure.png'
img = mpimg.imread(image_path)
plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.axis('off')
plt.show()
```

    Figure(640x480)



    
![png](tu2_files/tu2_14_1.png)
    


Note that the above plotting command has an extra parameter:
- `-min` specifies the starting band for plotting. 

For example, `-min 5` means starting from the 5th band, as the counting starts from 0. This indicates that the first 5 bands are completely discarded.

Users can change the value of `-min` to check the band structure. After determining the appropriate value for `-min`, we can rerun the `dftio parse` command with this parameter. 

Additionally, users should ensure that the chosen `-min` value does not exceed the total number of bands available in the dataset.


```python
!dftio parse -m abacus -r ./ -p ABACUS -f ase -o abc_ase -eig -min 5
```

    /opt/mamba/lib/python3.10/site-packages/dpdata/system.py:1106: UserWarning: Data type spins is registered twice; only the newly registered one will be used.
      warnings.warn(
    Parsing the DFT files: 100%|█████████████████████| 1/1 [00:00<00:00, 390.02it/s]


The above command will remove the first 5 bands from the band structure plot. 


```python
!dftio band -r ./abc_ase/AsGa.0 -f ase # -min 0

# display the band plot:
image_path = f'{workdir}/abc_ase/AsGa.0/band_structure.png'

img = mpimg.imread(image_path)
plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.axis('off')
plt.show()
```

    Figure(640x480)



    
![png](tu2_files/tu2_18_1.png)
    


### Finally, the training data set format for DeePTB-SK model is obtained. 


```python
!tree -L 2 ./abc_ase
```

    [01;34m./abc_ase[00m
    └── [01;34mAsGa.0[00m
        ├── band_structure.png
        ├── eigenvalues.npy
        ├── kpoints.npy
        └── xdat.traj
    
    1 directory, 4 files




###  1.2. VASP Case:

The following folder contains the result files from VASP calculations.


```python
workdir='/root/soft/DeePTB/examples/GaAs_io_sk/data/'
os.chdir(f"{workdir}")
! tree ./VASP -L 1
```

    [01;34m./VASP[00m
    ├── EIGENVAL
    ├── KPOINTS
    ├── OUTCAR
    └── POSCAR
    
    0 directories, 4 files


using the same command as in the ABACUS case, but with the `-m` parameter set to `vasp`:


```python
! dftio parse -m vasp -r ./ -p VASP -f ase -o vasp_ase -eig
```

    DFTIO WARNING VASP parser only supports the static (SCF or NSCF) calculations. MD and RELAX is not supported yet.
    DFTIO WARNING VASP parser only supports the static (SCF or NSCF) calculations. MD and RELAX is not supported yet.
    Parsing the DFT files: 100%|██████████████████████| 1/1 [00:00<00:00, 59.79it/s]


Observe that the above command is the same as the abacus case, except for the `-m` parameter, which is set to `vasp`.

For visualizing the band structure of data already output in ASE format, the process is similar to that of the ABACUS case. 


```python
!dftio band -r ./vasp_ase/AsGa.0 -f ase 

# display the band plot:
image_path = f'{workdir}/vasp_ase/AsGa.0/band_structure.png'
img = mpimg.imread(image_path)
plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.axis('off')
plt.show()
```

    Figure(640x480)



    
![png](tu2_files/tu2_27_1.png)
    


Again, there are core orbitals in the band structure. We can use the same method as in the ABACUS case to remove these bands.

Again, by visualization, we can get the information about how many bands need to be removed:


```python
!dftio band -r ./vasp_ase/AsGa.0 -f ase  -min 5

# display the band plot:
image_path = f'{workdir}/vasp_ase/AsGa.0/band_structure.png'

img = mpimg.imread(image_path)
plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.axis('off')
plt.show()
```

    Figure(640x480)



    
![png](tu2_files/tu2_30_1.png)
    


Re-run the data processing command to discard the lower energy bands. The command is as follows:


```python
! dftio parse -m vasp -r ./ -p VASP -f ase -o vasp_ase -eig -min 5
```

    DFTIO WARNING VASP parser only supports the static (SCF or NSCF) calculations. MD and RELAX is not supported yet.
    DFTIO WARNING VASP parser only supports the static (SCF or NSCF) calculations. MD and RELAX is not supported yet.
    Parsing the DFT files: 100%|██████████████████████| 1/1 [00:00<00:00, 95.83it/s]


To confirm the band structure, we can visualize it again. The command is as follows:


```python
!dftio band -r ./vasp_ase/AsGa.0 -f ase # -min 5

# display the band plot:
image_path = f'{workdir}/vasp_ase/AsGa.0/band_structure.png'
img = mpimg.imread(image_path)
plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.axis('off')
plt.show()
```

    Figure(640x480)



    
![png](tu2_files/tu2_34_1.png)
    


### Finally, the training data set format for DeePTB-SK model is obtained. 

**Note:**

After the above operations, the obtained data only contains one frame structure. If you have multiple frame structures, you can perform the following two operations:

1. Place different structures in different folders, using the same prefix and different suffixes for naming. Different folders should be placed in the same directory. The data structure is as follows:
    ```bash
      .root_dir
      └── prefix.suffix1
          ├── info.json
          ├── eigenvalues.npy
          ├── kpoints.npy
          └── xdat.traj
      └── prefix.suffix2
          ├── info.json
          ├── eigenvalues.npy
          ├── kpoints.npy
          └── xdat.traj
      └── ...
    ```

    **Note**: You may notice that I added an `info.json` file! This file contains some parameter settings for the training set, which are required when training DeePTB-SK. Please refer to the DeePTB-SK tutorial for details.

2. In another case, if the data for different frames of structures is the same except for the atomic coordinates (e.g., different structures in the same MD trajectory), you can merge them into one file. The `xdat.traj` file is used to save different frames of structures, and the shape of `eigenvalues.npy` is [nframe,nk,nb]. The shape of `kpoints` can be [nframe,nk,nb] or [nk,nb], where the latter indicates that this structure uses the same kpoints.


## 2. **Training DeePTB-SK Model using vasp data**

In the previous step, we have prepared the data, and now we can start training the model.


```python
import os
workdir='/root/soft/DeePTB/examples/GaAs_io_sk/'
os.chdir(f"{workdir}")
!tree -L 1 ./data/vasp_ase
```

    [01;34m./data/vasp_ase[00m
    ├── [01;34mAsGa.0[00m
    └── [01;34mprocessed_dataset_22915fbb7a750e40346716ff89f45ffe8cc73ea3[00m
    
    2 directories, 0 files



First, we need to create an `info.json` file in the data folder with the following content:
```json
{
    "nframes": 1,
    "natoms": 2,
    "pos_type": "ase",
    "pbc": true,
    "bandinfo": {
        "band_min": 0,
        "band_max": 8,
        "emin": null,
        "emax": null
    }
}
```
`nframes` indicates the number of trajectory snapshots, `natoms` indicates the number of atoms in each snapshot, `pos_type` indicates the coordinate type, and `pbc` indicates whether periodic boundary conditions are applied. The `bandinfo` section contains information about the band window, which can be set according to the user's needs. The band window information can be sorted by band index or divided by energy range. Note that the value of emin is relative to min(eig[band_min]). Taking min(eig[band_min]) as the 0 energy.


```python
import json 

infodict = {
    "nframes": 1,
    "natoms": 2,
    "pos_type": "ase",
    "pbc": True,
    "bandinfo": {
        "band_min": 0,
        "band_max": 8,
        "emin": None,
        "emax": None
        }
    }

with open(f'{workdir}/data/vasp_ase/AsGa.0/info.json', 'w') as f:
    json.dump(infodict, f, indent=4)
```

### 2.1 Extract initial empirical SK parameters

Refer to tutorial 1, we extract the initial SK parameters of GaAs from the built-in baseline model.


```python
os.chdir(f"{workdir}/train")
!dptb esk gaas.json -m poly4 
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
     
     
    DEEPTB INFO    Extracting empirical SK parameters for GaAs
    DEEPTB INFO    dtype is not provided in the input json, set to the value torch.float32 in model ckpt.
    DEEPTB INFO    device is not provided in the input json, set to the value cpu in model ckpt.
    DEEPTB INFO    overlap is not provided in the input json, set to the value True in model ckpt.
    DEEPTB INFO    Empirical SK parameters are saved in ./sktb.json
    DEEPTB INFO    If you want to further train the model, please use `dptb config` command to generate input template.


We can compare the initial model band structure with the DFT band structure to see how well the model fits the DFT results.


```python
!dptb run band.json -i sktb.json -o  band 

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
     
     
    DEEPTB WARNING Warning! structure is not set in run option, read from input config file.
    /opt/mamba/lib/python3.10/site-packages/torch/nested/__init__.py:107: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. (Triggered internally at ../aten/src/ATen/NestedTensorImpl.cpp:178.)
      return torch._nested_tensor_from_tensor_list(ts, dtype, None, device, None)
    DEEPTB INFO    KPOINTS  klist: 180 kpoints
    DEEPTB INFO    The eigenvalues are already in data. will use them.
    DEEPTB INFO    Calculating Fermi energy in the case of spin-degeneracy.
    DEEPTB INFO    Fermi energy converged after 16 iterations.
    DEEPTB INFO    q_cal: 7.999999999921436, total_electrons: 8.0, diff q: 7.856382211457458e-11
    DEEPTB INFO    Estimated E_fermi: -4.927888815054766 based on the valence electrons setting nel_atom : {'As': 5, 'Ga': 3} .
    DEEPTB INFO    No Fermi energy provided, using estimated value: -4.9279 eV
    Figure(640x560)
    DEEPTB INFO    band calculation successfully completed.



    
![png](tu2_files/tu2_46_1.png)
    


### 2.1 model training (briefly introduce)


```python
# 94.116 s on NVIDIA V100
!dptb train input.json -i sktb.json -o nnsk
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
    DEEPTB INFO         r_max            : {'Ga-Ga': 6.220000000000001, 'Ga-As': 6.43, 'As-Ga': 6.43, 'As-As': 6.630000000000001}    
    DEEPTB INFO         er_max           : None                                    
    DEEPTB INFO         oer_max          : None                                    
    DEEPTB INFO    ------------------------------------------------------------------
    /opt/mamba/lib/python3.10/site-packages/torch/nested/__init__.py:107: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. (Triggered internally at ../aten/src/ATen/NestedTensorImpl.cpp:178.)
      return torch._nested_tensor_from_tensor_list(ts, dtype, None, device, None)
    DEEPTB WARNING The cutoffs in data and model are not checked. be careful!
    DEEPTB INFO    The ['overlap_param'] are frozen!
    DEEPTB INFO    The ['overlap_param'] are frozen!
    DEEPTB INFO    The ['overlap_param'] are frozen!
    DEEPTB INFO    iteration:1	train_loss: 4.588784  (1.376635)	lr: 0.01
    DEEPTB INFO    checkpoint saved as nnsk.iter1
    DEEPTB INFO    Epoch 1 summary:	train_loss: 4.588784	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep1
    DEEPTB INFO    iteration:2	train_loss: 3.226382  (1.931559)	lr: 0.00997
    DEEPTB INFO    checkpoint saved as nnsk.iter2
    DEEPTB INFO    Epoch 2 summary:	train_loss: 3.226382	
    ...
    ...
    ...
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep499
    DEEPTB INFO    iteration:500	train_loss: 0.014339  (0.014363)	lr: 0.002233
    DEEPTB INFO    checkpoint saved as nnsk.iter500
    DEEPTB INFO    Epoch 500 summary:	train_loss: 0.014339	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep500
    DEEPTB INFO    finished training
    DEEPTB INFO    wall time: 138.742 s



```python
!dptb run band.json -i ./nnsk/checkpoint/nnsk.best.pth -o band_train
# !dptb run band.json -i ./ref_ckpt/nnsk_tr1.pth -o band_train

# display the band plot:
image_path = f'./band_train/results/band.png'
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
    DEEPTB WARNING Warning! structure is not set in run option, read from input config file.
    /opt/mamba/lib/python3.10/site-packages/torch/nested/__init__.py:107: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. (Triggered internally at ../aten/src/ATen/NestedTensorImpl.cpp:178.)
      return torch._nested_tensor_from_tensor_list(ts, dtype, None, device, None)
    DEEPTB INFO    KPOINTS  klist: 180 kpoints
    DEEPTB INFO    The eigenvalues are already in data. will use them.
    DEEPTB INFO    Calculating Fermi energy in the case of spin-degeneracy.
    DEEPTB INFO    Fermi energy converged after 26 iterations.
    DEEPTB INFO    q_cal: 8.000000000121316, total_electrons: 8.0, diff q: 1.213162903468401e-10
    DEEPTB INFO    Estimated E_fermi: -5.508918695276289 based on the valence electrons setting nel_atom : {'As': 5, 'Ga': 3} .
    DEEPTB INFO    No Fermi energy provided, using estimated value: -5.5089 eV
    Figure(640x560)
    DEEPTB INFO    band calculation successfully completed.



    
![png](tu2_files/tu2_49_1.png)
    


We can also continue training from the previous step's training results for one more round.


```python
!dptb train input.json -i ./nnsk/checkpoint/nnsk.best.pth -o nnsk2
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
    DEEPTB INFO         r_max            : {'Ga-Ga': 6.220000000000001, 'Ga-As': 6.43, 'As-Ga': 6.43, 'As-As': 6.630000000000001}    
    DEEPTB INFO         er_max           : None                                    
    DEEPTB INFO         oer_max          : None                                    
    DEEPTB INFO    ------------------------------------------------------------------
    /opt/mamba/lib/python3.10/site-packages/torch/nested/__init__.py:107: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. (Triggered internally at ../aten/src/ATen/NestedTensorImpl.cpp:178.)
      return torch._nested_tensor_from_tensor_list(ts, dtype, None, device, None)
    DEEPTB WARNING The cutoffs in data and model are not checked. be careful!
    DEEPTB INFO    The ['overlap_param'] are frozen!
    DEEPTB INFO    The ['overlap_param'] are frozen!
    DEEPTB INFO    iteration:1	train_loss: 0.014329  (0.004299)	lr: 0.01
    DEEPTB INFO    checkpoint saved as nnsk.iter1
    DEEPTB INFO    Epoch 1 summary:	train_loss: 0.014329	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep1
    DEEPTB INFO    iteration:2	train_loss: 0.015183  (0.007564)	lr: 0.00997
    DEEPTB INFO    checkpoint saved as nnsk.iter2
    DEEPTB INFO    Epoch 2 summary:	train_loss: 0.015183	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    iteration:3	train_loss: 0.071176  (0.026647)	lr: 0.00994
    DEEPTB INFO    checkpoint saved as nnsk.iter3
    DEEPTB INFO    Epoch 3 summary:	train_loss: 0.071176	
    ...
    ...
    ...
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep499
    DEEPTB INFO    iteration:500	train_loss: 0.003409  (0.003414)	lr: 0.002233
    DEEPTB INFO    checkpoint saved as nnsk.iter500
    DEEPTB INFO    Epoch 500 summary:	train_loss: 0.003409	
    ---------------------------------------------------------------------------------
    DEEPTB INFO    checkpoint saved as nnsk.ep500
    DEEPTB INFO    finished training
    DEEPTB INFO    wall time: 138.178 s



```python
!dptb run band.json -i ./nnsk2/checkpoint/nnsk.best.pth -o band_train
#!dptb run band.json -i ./ref_ckpt/nnsk_tr2.pth -o band_train


# display the band plot:
image_path = f'./band_train/results/band.png'
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
    DEEPTB WARNING Warning! structure is not set in run option, read from input config file.
    /opt/mamba/lib/python3.10/site-packages/torch/nested/__init__.py:107: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. (Triggered internally at ../aten/src/ATen/NestedTensorImpl.cpp:178.)
      return torch._nested_tensor_from_tensor_list(ts, dtype, None, device, None)
    DEEPTB INFO    KPOINTS  klist: 180 kpoints
    DEEPTB INFO    The eigenvalues are already in data. will use them.
    DEEPTB INFO    Calculating Fermi energy in the case of spin-degeneracy.
    DEEPTB WARNING Fermi level bisection did not converge under tolerance 1e-10 after 55 iterations.
    DEEPTB INFO    q_cal: 7.999999985718575, total_electrons: 8.0, diff q: 1.4281424931539277e-08
    DEEPTB INFO    Estimated E_fermi: -5.590556383132935 based on the valence electrons setting nel_atom : {'As': 5, 'Ga': 3} .
    DEEPTB INFO    No Fermi energy provided, using estimated value: -5.5906 eV
    Figure(640x560)
    DEEPTB INFO    band calculation successfully completed.



    
![png](tu2_files/tu2_52_1.png)
    


<div style="color:black; background-color:#FFF3E9; border: 1px solid #FFE0C3; border-radius: 10px; margin-bottom:1rem">
    <p style="margin:1rem; padding-left: 1rem; line-height: 2.5;">
        Author: <a style="font-weight:normal" href="mailto:guqq@ustc.edu.cn">Gu, Qiangqiang : guqq@ustc.edu.cn</a>
    </p>
    <p style="margin:1rem; padding-left: 1rem; line-height: 2.5;">
        Thank you for reading!
    </p>
    <p style="margin:1rem; padding-left: 1rem; line-height: 2.5;">
        For more information about training, please refer to tutorial 3.
    </p>


</div>
