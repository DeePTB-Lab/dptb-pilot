from pathlib import Path

from dptb_pilot.tools.modules.deeptb.submodules.band import _band_predict, _band_gap

RESOURCE_ROOT = Path(__file__).parent / "resources"
MODEL_FILE_PATH = RESOURCE_ROOT / 'nnsk.best.pth'
STRUCTURE_FILE_PATH = RESOURCE_ROOT / '10_0.vasp'


def test_band_with_sk_model(tmp_path):
    OUTPUT_ROOT_PATH = tmp_path / 'band_with_sk_model'
    OUTPUT_ROOT_PATH.mkdir(exist_ok=True)
    band_structure_file_path, _, fermi_level = _band_predict(model_file_path=MODEL_FILE_PATH,
                                                             structure_file_path=STRUCTURE_FILE_PATH,
                                                             nel_atom={"C": 4},
                                                             kpath="[[0.0,0.0,0.0,50,G],[0.0,0.0,0.5,1,Z]]",
                                                             work_path=str(OUTPUT_ROOT_PATH)).values()
    band_gap = _band_gap(band_structure_file_path=band_structure_file_path,
                         fermi_level=fermi_level).values()
    print(band_gap)


E3_MODEL_FILE_PATH = RESOURCE_ROOT / 'e3env.pth'
OVERLAP_FILE_PATH = RESOURCE_ROOT / '-3_overlaps.h5'


def test_band_with_e3_model(tmp_path):
    OUTPUT_ROOT_PATH = tmp_path / 'band_with_e3_model'
    OUTPUT_ROOT_PATH.mkdir(exist_ok=True)
    band_structure_file_path, _, fermi_level = _band_predict(model_file_path=E3_MODEL_FILE_PATH,
                                                             structure_file_path=STRUCTURE_FILE_PATH,
                                                             nel_atom={"C": 4},
                                                             kpath="[[0.0,0.0,0.0,50,G],[0.0,0.0,0.5,1,Z]]",
                                                             work_path=str(OUTPUT_ROOT_PATH)).values()
    band_gap = _band_gap(band_structure_file_path=band_structure_file_path,
                         fermi_level=fermi_level).values()
    print(band_gap)
