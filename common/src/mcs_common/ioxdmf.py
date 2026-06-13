import os
import h5py
import numpy as np

def write_hdf5(step, u, x, y, z=None, *, unames, output_dir):
    """
    Direct HDF5 writer. Saves JAX arrays natively without transposing.
    """
    fname = f"{output_dir}/wave_{step:05d}.h5"
    with h5py.File(fname, "w") as f:
        f.create_dataset("X", data=x)
        f.create_dataset("Y", data=y)
        if z is not None:
            f.create_dataset("Z", data=z)
            
        for m in range(len(unames)):
            # Direct save: 2D is (Nx, Ny), 3D is (Nx, Ny, Nz)
            f.create_dataset(unames[m], data=u[m])

def write_xdmf(output_dir, Nt, Nx, Ny, Nz=None, *, unames, output_interval, dt):
    """
    Direct XDMF writer. Dimensions perfectly match Python array shapes.
    """
    fname = os.path.join(output_dir, "wave.xdmf")
    is_3d = Nz is not None
    
    with open(fname, "w") as f:
        f.write('<?xml version="1.0" ?>\n')
        f.write('<Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude">\n')
        f.write('  <Domain>\n')
        f.write('    <Grid Name="TimeSeries" GridType="Collection" CollectionType="Temporal">\n')
        
        for n in range(0, Nt + 1, output_interval):
            h5_file = f"wave_{n:05d}.h5"
            
            f.write(f'      <Grid Name="wave_{n}" GridType="Uniform">\n')
            f.write(f'        <Time Value="{n*dt}"/>\n')
            
            if is_3d:
                # Direct Python mapping: Nx Ny Nz
                f.write(f'        <Topology TopologyType="3DRectMesh" Dimensions="{Nx} {Ny} {Nz}"/>\n')
                f.write('        <Geometry GeometryType="VXVYVZ">\n')
                f.write(f'          <DataItem Name="X" Dimensions="{Nx}" NumberType="Float" Precision="8" Format="HDF">{h5_file}:/X</DataItem>\n')
                f.write(f'          <DataItem Name="Y" Dimensions="{Ny}" NumberType="Float" Precision="8" Format="HDF">{h5_file}:/Y</DataItem>\n')
                f.write(f'          <DataItem Name="Z" Dimensions="{Nz}" NumberType="Float" Precision="8" Format="HDF">{h5_file}:/Z</DataItem>\n')
                f.write('        </Geometry>\n')
            else:
                # Direct Python mapping: Nx Ny
                f.write(f'        <Topology TopologyType="2DRectMesh" Dimensions="{Nx} {Ny}"/>\n')
                f.write('        <Geometry GeometryType="VXVY">\n')
                f.write(f'          <DataItem Name="X" Dimensions="{Nx}" NumberType="Float" Precision="8" Format="HDF">{h5_file}:/X</DataItem>\n')
                f.write(f'          <DataItem Name="Y" Dimensions="{Ny}" NumberType="Float" Precision="8" Format="HDF">{h5_file}:/Y</DataItem>\n')
                f.write('        </Geometry>\n')
            
            for name in unames:
                dim_str = f"{Nx} {Ny} {Nz}" if is_3d else f"{Nx} {Ny}"
                f.write(f'        <Attribute Name="{name}" AttributeType="Scalar" Center="Node">\n')
                f.write(f'          <DataItem Dimensions="{dim_str}" NumberType="Float" Precision="8" Format="HDF">{h5_file}:/{name}</DataItem>\n')
                f.write('        </Attribute>\n')
                
            f.write('      </Grid>\n')
            
        f.write('    </Grid>\n  </Domain>\n</Xdmf>\n')