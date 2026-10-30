from pathlib import Path


def _hamiltonian_test(
        model_path: Path,
        test_dataset_root_path: Path,
        test_dataset_prefix: str,
        get_overlap: bool = True,
        device: str = "cpu",
        onsite_shift: bool = False,
        clean: bool = True
):
    """
    Evaluate DeePTB Hamiltonian and overlap prediction errors on a test dataset.

    Parameters
    ----------
    model_path : Path
        Path to a DeePTB model checkpoint or JSON model file.
    test_dataset_root_path : Path
        Root directory containing the test dataset folders.
    test_dataset_prefix : str
        Prefix used to select dataset entries under ``test_dataset_root_path``.
    get_overlap : bool, optional
        Whether the dataset contains overlap matrices and overlap errors should be
        analyzed.
    device : str, optional
        Device used for model evaluation, for example ``cpu`` or ``cuda``.
    onsite_shift : bool, optional
        Whether to enable onsite-shift handling in ``HamilLossAnalysis``.
    clean : bool, optional
        Whether to remove processed dataset cache directories after evaluation.

    Returns
    -------
    dict
        Nested statistics dictionary from ``HamilLossAnalysis.stats``.
    """
    from dptb.data import build_dataset
    from dptb.nn import build_model

    model = build_model(str(model_path.absolute()),
                        common_options={"device": device})
    model.eval()

    dataset = build_dataset(
        root=str(test_dataset_root_path.absolute()),
        type="DefaultDataset",
        prefix=test_dataset_prefix,
        get_overlap=get_overlap,
        get_Hamiltonian=True,
        basis=model.basis,
        r_max=model.model_options["embedding"]["r_max"]
    )

    import torch
    from dptb.nnops.loss import HamilLossAnalysis
    from dptb.data.dataloader import DataLoader
    from tqdm import tqdm
    from dptb.data import AtomicData

    ana = HamilLossAnalysis(idp=model.idp, device=device, decompose=True, overlap=True, onsite_shift=onsite_shift)

    loader = DataLoader(dataset, batch_size=10, shuffle=False, num_workers=0)

    for data in tqdm(loader, desc="doing error analysis"):
        with torch.no_grad():
            data = data.to(device)
            batch_info = data.get_batchinfo()
            ref_data = AtomicData.to_AtomicDataDict(data)
            data = model(ref_data)
            data.update(batch_info)
            ref_data.update(batch_info)
            ana(data, ref_data, running_avg=True)

    if clean:
        import re
        import os
        import shutil
        for item in os.listdir(test_dataset_root_path):
            if os.path.isdir(item) and re.match(r'^processed_dataset_.*', item):
                shutil.rmtree(item)
                print(f"已删除: {item}")

    # stats contain:
    # self.stats["mae"] = 0.
    # self.stats["rmse"] = 0.
    # self.stats["n_element"] = 0
    #
    # self.stats.setdefault("onsite", {})
    # self.stats.setdefault("hopping", {})
    # if self.overlap:
    #     self.stats.setdefault("overlap", {})
    #
    # for at, tp in self.idp.chemical_symbol_to_type.items():
    #     self.stats["onsite"][at] = {
    #         "rmse":0.,
    #         "mae":0.,
    #         "rmse_per_block_element":torch.zeros(1, dtype=self.dtype, device=self.device),
    #         "mae_per_block_element":torch.zeros(1, dtype=self.dtype, device=self.device),
    #         "rmse_per_irreps":torch.zeros(1, dtype=self.dtype, device=self.device),
    #         "mae_per_irreps":torch.zeros(1, dtype=self.dtype, device=self.device),
    #         "n_element":0,
    #     }
    #
    # for bt, tp in self.idp.bond_to_type.items():
    #     self.stats["hopping"][bt] = {
    #         "rmse":0.,
    #         "mae":0.,
    #         "rmse_per_block_element":torch.zeros(1, dtype=self.dtype, device=self.device),
    #         "mae_per_block_element":torch.zeros(1, dtype=self.dtype, device=self.device),
    #         "rmse_per_irreps":torch.zeros(1, dtype=self.dtype, device=self.device),
    #         "mae_per_irreps":torch.zeros(1, dtype=self.dtype, device=self.device),
    #         "n_element":0,
    #     }
    #
    #     if self.overlap:
    #         self.stats["overlap"][bt] = {
    #             "rmse":0.,
    #             "mae":0.,
    #             "rmse_per_block_element":torch.zeros(1, dtype=self.dtype, device=self.device),
    #             "mae_per_block_element":torch.zeros(1, dtype=self.dtype, device=self.device),
    #             "rmse_per_irreps":torch.zeros(1, dtype=self.dtype, device=self.device),
    #             "mae_per_irreps":torch.zeros(1, dtype=self.dtype, device=self.device),
    #             "n_element":0,
    #         }
    return ana.stats
