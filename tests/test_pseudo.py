"""Geometry regressions for the two-stage Dataset510 transform."""
import copy
import unittest
from types import SimpleNamespace
import numpy as np
import SimpleITK as sitk
import torch
from bms_deploy.geometry import Volume
from bms_deploy.da_adapter import prepare, from_sitk
from bms_deploy.pseudo import make_transform, network_reference, prepare_pair, restore_native, undo_crops, validate_transform

class PseudoTests(unittest.TestCase):
    def fixture(self, affine=None):
        if affine is None: affine=np.diag([.7,.9,1.5,1.]);affine[:3,3]=[32,-11,9]
        real=Volume(np.arange(7*9*5,dtype=np.float32).reshape(7,9,5),affine)
        original,work,source,_=prepare(real)
        meta=make_transform(original,work,'new_case','T1CE')
        grid=from_sitk(network_reference(work)).affine
        return real,original,work,meta,grid
    def test_center_grid_and_nearest_floor_restore(self):
        for aff in (np.diag([.7,.9,1.5,1.]),np.array([[0,0,-.7,40],[0,1.2,0,-20],[2,0,0,5],[0,0,0,1.]])):
            native,original,work,meta,grid=self.fixture(aff)
            validate_transform(meta)
            shape=(256,256,work.GetSize()[2]);x=np.zeros(shape,np.uint8);x[::3,::5,::2]=1
            output=restore_native(Volume(x,grid),meta)
            # Independent floor index oracle, then SITK physical sampling.
            ix=np.floor(np.arange(work.GetSize()[0])*256/work.GetSize()[0]).astype(int)
            iy=np.floor(np.arange(work.GetSize()[1])*256/work.GetSize()[1]).astype(int)
            xy=x[ix[:,None],iy[None,:],:]
            w=sitk.GetImageFromArray(xy.transpose(2,1,0));w.CopyInformation(work)
            expected=sitk.Resample(w,original,sitk.Transform(),sitk.sitkNearestNeighbor,0,sitk.sitkUInt8)
            np.testing.assert_array_equal(np.asarray(output.dataobj),sitk.GetArrayFromImage(expected).transpose(2,1,0))
            np.testing.assert_allclose(output.affine,native.affine,atol=1e-4,rtol=0)
    def test_real_background_channels_two_crop_inverse(self):
        _,_,work,meta,aff=self.fixture()
        shape=(256,256,work.GetSize()[2]);r=np.full(shape,-1.,np.float32);r[2:7,4:11,1:3]=.2;r[0,0,0]=-1+5e-7
        s=np.ones(shape,np.float32)*.8
        for seq,index in [('T1CE',0),('BB',1)]:
            m=copy.deepcopy(meta);m['input_sequence']=seq;m['real_channel_index']=index
            raw,props,info=prepare_pair(Volume(r,aff),Volume(s,aff),seq,m)
            self.assertEqual(info['real_bbox_xyz'],[[2,7],[4,11],[1,3]])
            self.assertTrue(np.all(raw[index]==.2));self.assertTrue(np.all(raw[1-index]==.8))
            pp=dict(shape_before_cropping=[2,7,5],shape_after_cropping_and_before_resampling=[2,3,4],bbox_used_for_cropping=[[0,2],[2,5],[1,5]])
            full=undo_crops(np.full((2,3,4),.75,np.float32),pp,SimpleNamespace(transpose_backward=[0,1,2]),info)
            self.assertEqual(np.count_nonzero(full.data),24)
            mask=restore_native(Volume((full.data>=.5).astype(np.uint8),aff),m)
            prob=restore_native(full,m,probability=True)
            np.testing.assert_array_equal(np.asarray(mask.dataobj)>0,np.asarray(prob.dataobj)>=.5)
    def test_wrong_schema_grid_and_provenance_are_rejected(self):
        _,_,work,meta,aff=self.fixture();shape=(256,256,work.GetSize()[2]);a=Volume(np.zeros(shape,np.float32),aff)
        for key in ['schema_version','pseudo_model_space']:
            bad=copy.deepcopy(meta)
            if key=='schema_version':bad[key]='i2sb-spatial-transform-v1'
            else:bad[key]['origin'][0]+=1
            with self.assertRaises(ValueError):validate_transform(bad)
        with self.assertRaises(ValueError):prepare_pair(a,a,'BB',meta)
        wrong=Volume(a.data,aff.copy());wrong.affine[0,3]+=1
        with self.assertRaises(ValueError):prepare_pair(a,wrong,'T1CE',meta)
