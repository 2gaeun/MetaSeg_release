import tempfile,unittest
from pathlib import Path
import numpy as np
from bms_deploy.geometry import Volume,prepare_pair,restore_native,require_same_grid
from bms_deploy.dicom_input import read_input_csv,read_series,scan_series


class GeometryTests(unittest.TestCase):
    def test_pair_channels_crop_and_native_inverse(self):
        a=np.zeros((8,9,10),np.float32);a[2:6,1:7,3:9]=3
        aff=np.diag([-2.,3.,-1.,1.]);aff[:3,3]=[12,4,21]
        real=Volume(a,aff);syn=Volume(a+4,aff.copy())
        for seq,index in [('T1CE',0),('BB',1)]:
            raw,props,info=prepare_pair(real,syn,seq)
            self.assertTrue(np.all(raw[index]==3));self.assertTrue(np.all(raw[1-index]==7))
            restored=restore_native(np.ones(raw.shape[1:],np.uint8),info)
            np.testing.assert_array_equal(np.asarray(restored.dataobj)>0,a>0)
            np.testing.assert_allclose(restored.affine,aff)
    def test_geometry_mismatch_fails(self):
        a=Volume(np.ones((3,3,3),np.float32),np.eye(4))
        b=Volume(a.data.copy(),np.eye(4));b.affine[0,3]=1
        with self.assertRaises(ValueError):require_same_grid(a,b)
    def test_csv_strict(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'a').mkdir();p=root/'input.csv'
            p.write_text('folder,sequence\na,bb\n');self.assertEqual(read_input_csv(root,p)[0][1],'BB')
            p.write_text('../a,BB\n')
            with self.assertRaises(ValueError):read_input_csv(root,p)
            p.write_text('a,T2\n')
            with self.assertRaises(ValueError):read_input_csv(root,p)


class DicomTests(unittest.TestCase):
    def test_two_classic_mr_volumes_no_nifti_intermediate(self):
        import SimpleITK as sitk
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for series in (1,2):
                folder=root/f's{series}';folder.mkdir()
                for z in range(3):
                    image=sitk.GetImageFromArray(np.full((8,10),z+series,np.int16));image.SetSpacing([.5,.75])
                    tags={'0008|0060':'MR','0008|0016':'1.2.840.10008.5.1.4.1.1.4',
                          '0008|0018':f'1.2.826.0.1.3680043.10.999.{series}.{z+1}',
                          '0020|000d':'1.2.826.0.1.3680043.10.999.99',
                          '0020|000e':f'1.2.826.0.1.3680043.10.999.{series}',
                          '0020|0032':f'10\\20\\{30+z*2}', '0020|0037':'1\\0\\0\\0\\1\\0',
                          '0028|0030':'0.75\\0.5','0020|0013':str(z+1)}
                    for key,value in tags.items():image.SetMetaData(key,value)
                    w=sitk.ImageFileWriter();w.KeepOriginalImageUIDOn();w.SetFileName(str(folder/f'{2-z}.dcm'));w.Execute(image)
            with self.assertRaisesRegex(ValueError, 'exactly one DICOM series'):
                scan_series(root)
            volumes=[scan_series(root/f's{series}')[0] for series in (1,2)]
            for uid,paths in volumes:
                v=read_series(paths);self.assertEqual(v.data.shape,(10,8,3))
                np.testing.assert_allclose(v.affine,np.array([[-.5,0,0,-10],[0,-.75,0,-20],[0,0,2,30],[0,0,0,1]]))
                # Duplicate a DICOM file must not silently merge a temporal stack.
                with self.assertRaises(ValueError):read_series(paths+[paths[0]])
            self.assertEqual(list(root.rglob('*.nii*')),[])

if __name__=='__main__':unittest.main()
