import React, { useEffect, useRef } from 'react';
import * as $3Dmol from '3dmol';

interface StructureViewerProps {
    data: string;
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
