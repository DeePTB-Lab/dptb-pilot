import os
import json
import numpy as np
import seekpath
from pymatgen.core import Structure
from scipy.spatial import Voronoi, ConvexHull
import traceback

def visualize_brillouin_zone(file_name: str):
    print(f"Processing {file_name}...")
    try:
        # Load structure using pymatgen
        struct = Structure.from_file(file_name)
        
        # Convert to seekpath input format
        cell = struct.lattice.matrix.tolist()
        positions = struct.frac_coords.tolist()
        numbers = [site.specie.number for site in struct]
        
        structure_tuple = (cell, positions, numbers)
        
        # Get BZ data
        print("Calling seekpath...")
        res = seekpath.get_explicit_k_path(structure_tuple)
        print("Seekpath finished.")
        
        recip_lattice = np.array(res['reciprocal_primitive_lattice'])
        print(f"Reciprocal lattice: {recip_lattice}")
        
        # Generate neighbor points to define the Voronoi cell
        # 3x3x3 grid is usually enough
        print("Generating grid...")
        px, py, pz = np.tensordot(recip_lattice, np.mgrid[-1:2, -1:2, -1:2], axes=[0, 0])
        points = np.c_[px.ravel(), py.ravel(), pz.ravel()]
        center = points[13] # (0,0,0) should be at index 13
        
        print(f"Grid points shape: {points.shape}")
        
        print("Calculating Voronoi...")
        vor = Voronoi(points)
        
        # Find the region corresponding to the center point (0,0,0)
        region_index = vor.point_region[13]
        region = vor.regions[region_index]
        
        if -1 in region:
            print("Warning: Infinite region found for center point!")
        
        # Get vertices of the BZ
        bz_vertices = [vor.vertices[i] for i in region]
        print(f"BZ vertices count: {len(bz_vertices)}")
        
        # Get faces (ridges) using ConvexHull
        print("Calculating ConvexHull...")
        hull = ConvexHull(bz_vertices)
        
        # Prepare data for frontend
        vertices_list = np.array(bz_vertices).tolist()
        
        edges = []
        for simplex in hull.simplices:
            for i in range(len(simplex)):
                p1 = simplex[i]
                p2 = simplex[(i+1) % len(simplex)]
                edge = tuple(sorted((p1, p2)))
                edges.append(edge)
                
        unique_edges = list(set(edges))
        print(f"Unique edges count: {len(unique_edges)}")
        
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
        
        print("Success! Payload generated.")
        # print(json.dumps(payload, indent=2))
        
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    # Create a dummy POSCAR for Si
    poscar_content = """Si diamond
3.867
0.0 0.5 0.5
0.5 0.0 0.5
0.5 0.5 0.0
Si
2
Direct
0.000000 0.000000 0.000000
0.250000 0.250000 0.250000
"""
    with open("POSCAR_Si", "w") as f:
        f.write(poscar_content)
        
    visualize_brillouin_zone("POSCAR_Si")
