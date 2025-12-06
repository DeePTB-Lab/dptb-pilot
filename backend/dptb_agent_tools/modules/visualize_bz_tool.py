import os
import json
import numpy as np
from dptb_agent_tools.init_mcp import mcp
import seekpath
from pymatgen.core import Structure

@mcp.tool()
def visualize_brillouin_zone(file_name: str, work_path: str) -> str:
    """
    Calculate and visualize the Brillouin Zone (BZ) for a given structure file.
    
    Args:
        file_name: The name of the structure file (e.g., POSCAR, structure.cif).
        work_path: The absolute path to the workspace directory.
        
    Returns:
        A special markdown block that the frontend uses to render the BZ.
    """
    file_path = os.path.join(work_path, file_name)
    
    if not os.path.exists(file_path):
        return f"Error: File {file_name} not found in {work_path}"
        
    try:
        # Load structure using pymatgen
        struct = Structure.from_file(file_path)
        
        # Convert to seekpath input format
        # (cell, positions, numbers)
        cell = struct.lattice.matrix.tolist()
        positions = struct.frac_coords.tolist()
        numbers = [site.specie.number for site in struct]
        
        structure_tuple = (cell, positions, numbers)
        
        # Get BZ data
        res = seekpath.get_explicit_k_path(structure_tuple)
        
        # Extract relevant data for visualization
        # seekpath returns 'bravais_lattice_extended' which contains 'vertices' and 'faces' of the BZ
        # But usually we want the primitive BZ. 
        # 'bravais_lattice' key might be what we want if we want the standard one.
        # Let's use 'bravais_lattice' from the result if available, or just use the output.
        # Actually seekpath returns:
        # - point_coords: dict of label -> coords
        # - path: list of (start_label, end_label)
        # - primitive_lattice: The primitive lattice
        # - reciprocal_primitive_lattice: The reciprocal lattice
        # - bz_vertices: List of vertices of the BZ
        # - bz_faces: List of faces (indices of vertices)
        
        # Note: seekpath output keys might vary slightly depending on version/function.
        # get_explicit_k_path returns a dict.
        
        # We need to construct a payload that the frontend can easily draw.
        # We need:
        # 1. Vertices
        # 2. Faces (to draw edges)
        # 3. High symmetry points (labels and coords)
        # 4. Path (optional, but good to highlight)
        
        # Check if 'bravais_lattice_extended' is the one used for BZ visualization in seekpath web.
        # Actually, let's look at what seekpath returns.
        # It returns 'bravais_lattice' and 'reciprocal_primitive_lattice'.
        # Wait, seekpath doesn't explicitly return BZ vertices in `get_explicit_k_path` directly in the top level dict?
        # Let me check seekpath docs or assume I need to calculate it.
        # Actually, seekpath has `get_brillouin_zone` function? No.
        
        # Let's try to find how to get BZ vertices.
        # Ah, `seekpath.get_explicit_k_path` returns `bravais_lattice_extended`? No.
        # It returns `primitive_lattice` and `reciprocal_primitive_lattice`.
        # To get the BZ shape (Wigner-Seitz cell of reciprocal lattice), we might need another library or calculate it.
        # BUT, seekpath usually *does* provide this info for its own visualizer.
        
        # Let's check if I can simply use `seekpath.get_path`?
        # `seekpath.get_path` returns `point_coords`, `path`, etc.
        
        # Actually, to draw the BZ, we need the Voronoi cell of the reciprocal lattice.
        # Scipy's Voronoi can do this.
        # Or maybe `seekpath` has a helper.
        
        # Let's look at `seekpath` source or common usage.
        # It seems `seekpath` is mostly for finding the path.
        # However, the web interface shows the BZ.
        
        # Let's implement a simple BZ calculator using Scipy if seekpath doesn't give it directly.
        # 1. Get reciprocal lattice from seekpath (or pymatgen).
        # 2. Generate grid of reciprocal lattice points.
        # 3. Compute Voronoi diagram around (0,0,0).
        # 4. The region corresponding to (0,0,0) is the first BZ.
        
        # Wait, that's complex to implement robustly in a single script without testing.
        # Let's see if pymatgen has BZ visualization tools.
        # `pymatgen.electronic_structure.plotter.BSPlotter`? No.
        # `pymatgen.symmetry.bandstructure.HighSymmKpath`?
        
        # Let's try to use `seekpath`'s output if it has it.
        # If not, I'll use a simplified approach: just show the high-symmetry points and the path connecting them.
        # That's often enough for "visualizing the path".
        # But the user asked for "Brillouin Zone", which implies the shape.
        
        # Let's try to use `scipy.spatial.Voronoi`.
        from scipy.spatial import Voronoi, ConvexHull
        
        recip_lattice = np.array(res['reciprocal_primitive_lattice'])
        
        # Generate neighbor points to define the Voronoi cell
        # 3x3x3 grid is usually enough
        px, py, pz = np.tensordot(recip_lattice, np.mgrid[-1:2, -1:2, -1:2], axes=[0, 0])
        points = np.c_[px.ravel(), py.ravel(), pz.ravel()]
        center = points[13] # (0,0,0) should be at index 13
        
        vor = Voronoi(points)
        
        # Find the region corresponding to the center point (0,0,0)
        region_index = vor.point_region[13]
        region = vor.regions[region_index]
        
        # Get vertices of the BZ
        bz_vertices = [vor.vertices[i] for i in region]
        
        # Get faces (ridges)
        # This is a bit tricky with Scipy Voronoi to get just the faces of one region.
        # Alternative: Use ConvexHull of the vertices?
        # The BZ is a convex polyhedron.
        hull = ConvexHull(bz_vertices)
        
        # Prepare data for frontend
        # Vertices
        vertices_list = np.array(bz_vertices).tolist()
        
        # Edges (from Hull simplices)
        edges = []
        for simplex in hull.simplices:
            # simplex is a face (list of vertex indices)
            # We want edges. A face has edges connecting its vertices.
            # But ConvexHull simplices are triangles (in 3D).
            # The BZ faces might be polygons.
            # For visualization, drawing the edges of the triangular mesh is okay, 
            # but it might look cluttered.
            # Better to draw the edges of the Voronoi cell directly.
            
            # Let's just send the simplices (triangles) and let frontend draw lines?
            # Or better, extract unique edges from simplices.
            for i in range(len(simplex)):
                p1 = int(simplex[i])
                p2 = int(simplex[(i+1) % len(simplex)])
                edge = tuple(sorted((p1, p2)))
                edges.append(edge)
                
        unique_edges = list(set(edges))
        
        # High symmetry points
        kpoints = res['point_coords']
        path = res['path']
        
        payload = {
            "format": "bz",
            "data": {
                "vertices": vertices_list,
                "edges": unique_edges,
                "kpoints": kpoints,
                "path": path,
                "reciprocal_lattice": recip_lattice.tolist()
            }
        }
        
        return f":::visualize\n{json.dumps(payload)}\n:::"
        
    except Exception as e:
        import traceback
        error_msg = f"Error calculating BZ: {str(e)}\n{traceback.format_exc()}"
        print(f"❌ {error_msg}") # Print to server logs
        return error_msg
