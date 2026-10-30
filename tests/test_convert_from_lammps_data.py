from pathlib import Path

from dptb_pilot.tools.modules.deeptb.submodules.convert import convert_from_lammps_data

RESOURCE_ROOT = Path(__file__).parent / "resources"


def test_convert_from_lammps_data_to_vasp(tmp_path):
    result = convert_from_lammps_data(RESOURCE_ROOT / 'relaxed+3.data',
                             output_name='test_generated_vasp_poscar_file',
                             output_path=tmp_path,
                             format='vasp')
    assert result.is_file()

def test_convert_from_lammps_data_to_abacus(tmp_path):
    result = convert_from_lammps_data(RESOURCE_ROOT / 'relaxed+3.data',
                             output_name='test_generated_abacus_stru_file',
                             output_path=tmp_path,
                             format='abacus')
    assert result.is_file()
