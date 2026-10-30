from pathlib import Path

from dptb_pilot.tools.modules.deeptb.submodules.uniaxial_strain import _generate_uniaxial_strain_input_file

STRUCTURE_FILE = Path(__file__).parent / "resources" / "10_0.vasp"

TARGET_LENGTH = 20.0  # Å, target supercell axial length
AXIS = 'z'          # x,y,z or auto

# strain values in percent (same meaning as PRL 1999)
STRAIN_LIST = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]

MODEL_NAME = "model.pb"

RELAX_AFTER_STRAIN = True  # True -> LAMMPS minimize

def test_uniaxial_strain(tmp_path):
    result = _generate_uniaxial_strain_input_file(tmp_path / "uniaxial",
                                         STRUCTURE_FILE,
                                         strain_list=STRAIN_LIST,
                                         axis=AXIS,
                                         target_length=TARGET_LENGTH,
                                         model_name=MODEL_NAME,
                                         relax_after_strain=RELAX_AFTER_STRAIN)
    assert len(result["in_lammps_file_paths"]) == len(STRAIN_LIST)
    assert all(path.is_file() for path in result["in_lammps_file_paths"])
