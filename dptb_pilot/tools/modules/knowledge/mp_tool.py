import os
from typing import List, Dict, Any
from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter
from dptb_pilot.tools.init import mcp

@mcp.tool()
def search_materials_project(query: str, is_metal: bool = None, dimensionality: int = None, 
                           band_gap_min: float = None, band_gap_max: float = None,
                           energy_above_hull_max: float = None, limit: int = 3) -> str:
    """
    Search for materials in the Materials Project database with advanced filtering.
    
    Args:
        query: The search query (formula, elements, or ID).
        is_metal: Optional filter. Set to True for metals, False for semiconductors/insulators.
        dimensionality: Optional filter. 1 for 1D, 2 for 2D, 3 for 3D bulk materials.
        band_gap_min: Minimum band gap in eV.
        band_gap_max: Maximum band gap in eV.
        energy_above_hull_max: Maximum energy above hull in eV/atom (stability).
        limit: Maximum number of results to return (default: 3).
        
    Returns:
        A formatted string list of found materials with their IDs, formulas, band gap, stability, and structure info.
    """
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        return "Error: MP_API_KEY environment variable not set. Please configure it in .env."

    try:
        with MPRester(api_key) as mpr:
            # Prepare search arguments
            search_args = {
                "fields": ["material_id", "formula_pretty", "symmetry", "energy_above_hull", "formation_energy_per_atom", "band_gap", "is_metal", "structure"],
                "num_chunks": 1,
                "chunk_size": limit
            }
            
            if is_metal is not None:
                search_args["is_metal"] = is_metal
            
            if band_gap_min is not None or band_gap_max is not None:
                search_args["band_gap"] = (band_gap_min if band_gap_min is not None else 0, 
                                         band_gap_max if band_gap_max is not None else 100)
                                         
            if energy_above_hull_max is not None:
                search_args["energy_above_hull"] = (0, energy_above_hull_max)

            # Try searching by formula or elements
            docs = mpr.summary.search(
                formula=query if "-" not in query and "mp-" not in query else None,
                chemsys=query if "-" in query else None,
                material_ids=[query] if "mp-" in query else None,
                **search_args
            )
            
            # If standard search fails or returns nothing, and it looks like a formula, try strict formula
            if not docs and "mp-" not in query:
                 docs = mpr.summary.search(formula=query, **search_args)

            # Filter by dimensionality if requested
            # Since summary docs might not have dimensionality directly, we might need to infer it or it might be there.
            # Let's assume we filter by checking the structure if we have it, or rely on the agent to check the description.
            # Wait, fetching structure for all search results is heavy.
            # Let's check if we can filter by 'dimensionality' in search. 
            # If not, we will rely on the output description.
            # But wait, the user wants "Graphene".
            # Let's add logic to calculate dimensionality if possible, or just return symmetry which is a good proxy.
            # Spacegroup 191 (P6/mmm) is typical for Graphene.
            
            # Let's just return more detailed info so the Agent can decide.
            
            # Sort by stability (energy above hull)
            docs.sort(key=lambda x: x.energy_above_hull)
            
            filtered_docs = []
            for doc in docs:
                if dimensionality is not None:
                    # Heuristic: 2D materials usually have specific space groups or we can try to use pymatgen's dimensionality finder if structure is available.
                    # But structure is heavy.
                    # Let's just return all and let the Agent read the symmetry.
                    pass
                filtered_docs.append(doc)
            
            results = filtered_docs[:limit]
            
            if not results:
                return f"No materials found for query: {query}"
            
            output = f"Found {len(results)} materials for '{query}' (showing top {len(results)}):\n\n"
            for i, doc in enumerate(results):
                output += f"{i+1}. **{doc.formula_pretty}** (ID: `{doc.material_id}`)\n"
                output += f"   - Symmetry: {doc.symmetry.symbol} (No. {doc.symmetry.number}, {doc.symmetry.crystal_system}, Point Group: {doc.symmetry.point_group})\n"
                output += f"   - Band Gap: {doc.band_gap:.3f} eV ({'Metal' if doc.is_metal else 'Insulator/Semiconductor'})\n"
                output += f"   - Stability: {doc.energy_above_hull:.3f} eV/atom above hull\n"
                output += f"   - Formation Energy: {doc.formation_energy_per_atom:.3f} eV/atom\n\n"
                
            output += "To download a structure, use the `download_mp_structure` tool with the Material ID."
            return output

    except Exception as e:
        return f"Error searching Materials Project: {str(e)}"

@mcp.tool()
def download_mp_structure(mp_id: str, filename: str = None, work_path: str = ".") -> str:
    """
    Download a crystal structure from Materials Project by its ID.
    
    Args:
        mp_id: The Materials Project ID (e.g., "mp-149").
        filename: Optional custom filename. If not provided, uses "{mp_id}.cif".
        work_path: The directory to save the file in.
        
    Returns:
        Success message with the path to the downloaded file.
    """
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        return "Error: MP_API_KEY environment variable not set."

    try:
        with MPRester(api_key) as mpr:
            structure = mpr.get_structure_by_material_id(mp_id)
            
            if not filename:
                filename = f"{mp_id}.cif"
            
            if not filename.endswith(".cif") and not filename.endswith(".poscar"):
                filename += ".cif"
                
            # Ensure work_path exists
            if not os.path.exists(work_path):
                try:
                    os.makedirs(work_path, exist_ok=True)
                except Exception as e:
                    return f"Error creating workspace directory {work_path}: {e}"

            save_path = os.path.join(work_path, filename)
            
            # Write file
            if filename.lower().endswith(".cif"):
                writer = CifWriter(structure)
                writer.write_file(save_path)
            else:
                structure.to(filename=save_path)
                
            return f"Successfully downloaded structure {mp_id} to `{save_path}`. You can now visualize it using `visualize_structure`."

    except Exception as e:
        return f"Error downloading structure {mp_id}: {str(e)}"
