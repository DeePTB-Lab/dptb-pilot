import React, { useEffect, useRef } from 'react';
import * as $3Dmol from '3dmol';

interface StructureViewerProps {
    data: any;
    format: string;
    height?: string;
    width?: string;
}

const StructureViewer: React.FC<StructureViewerProps> = ({ 
    data, 
    format, 
    height = '400px', 
    width = '100%' 
}) => {
    const viewerRef = useRef<HTMLDivElement>(null);
    const glViewerRef = useRef<any>(null);

    useEffect(() => {
        if (!viewerRef.current) return;

        // Initialize viewer if not already initialized
        if (!glViewerRef.current) {
            const config = { backgroundColor: 'white' };
            glViewerRef.current = $3Dmol.createViewer(viewerRef.current, config);
        }

        const viewer = glViewerRef.current;
        
        // Clear previous model
        viewer.clear();

        if (format === 'bz') {
            // Render Brillouin Zone
            const { vertices, edges, kpoints, path } = data;

            // Draw BZ edges
            edges.forEach((edge: number[]) => {
                const v1 = vertices[edge[0]];
                const v2 = vertices[edge[1]];
                viewer.addLine({
                    start: { x: v1[0], y: v1[1], z: v1[2] },
                    end: { x: v2[0], y: v2[1], z: v2[2] },
                    color: '#00BCD4',
                    linewidth: 3
                });
            });

            // Draw Path
            path.forEach(([startLabel, endLabel]: [string, string]) => {
                const start = kpoints[startLabel];
                const end = kpoints[endLabel];
                viewer.addLine({
                    start: { x: start[0], y: start[1], z: start[2] },
                    end: { x: end[0], y: end[1], z: end[2] },
                    color: '#FF5722',
                    linewidth: 15
                });
            });

            // Draw High Symmetry Points
            Object.entries(kpoints).forEach(([label, coords]: [string, any]) => {
                viewer.addSphere({
                    center: { x: coords[0], y: coords[1], z: coords[2] },
                    radius: 0.02,
                    color: '#FF5722'
                });
                viewer.addLabel(label, {
                    position: { x: coords[0], y: coords[1], z: coords[2] },
                    fontColor: 'black',
                    fontSize: 16,
                    showBackground: false
                });
            });

            viewer.zoomTo();
            viewer.render();
            return;
        }

        // Add new model
        // format mapping: POSCAR -> vasp, CIF -> cif
        let modelFormat = format.toLowerCase();
        if (modelFormat.includes('poscar') || modelFormat === 'vasp') {
            modelFormat = 'vasp';
        }

        try {
            viewer.addModel(data, modelFormat);
            
            // Style the model
            viewer.setStyle({}, { 
                sphere: { scale: 0.3 }, 
                stick: { radius: 0.15 } 
            });
            
            // Add unit cell box
            viewer.addUnitCell();
            
            // Zoom to fit
            viewer.zoomTo();
            
            // Render
            viewer.render();
        } catch (e) {
            console.error("Error rendering structure:", e);
            viewer.addLabel("Error rendering structure", { position: { x: 0, y: 0, z: 0 } });
            viewer.render();
        }

    }, [data, format]);

    return (
        <div 
            ref={viewerRef} 
            style={{ 
                height, 
                width, 
                position: 'relative', 
                border: '1px solid #d9d9d9', 
                borderRadius: '8px',
                overflow: 'hidden'
            }} 
        />
    );
};

export default StructureViewer;
