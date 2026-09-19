import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from evaluator import _pixel_change_maps, _change_heatmap, _save_visualization


class VisualizationTest(unittest.TestCase):
    def test_zero_and_signed_changes(self):
        original=torch.full((3,4,4),0.4)
        for change in _pixel_change_maps(original,original):
            np.testing.assert_array_equal(change,0)
            heatmap,limit=_change_heatmap(change)
            self.assertEqual(limit,0)
            self.assertTrue((np.asarray(heatmap)==255).all())
        brighter=original+0.2
        brightness,saturation,warmth=_pixel_change_maps(original,brighter)
        self.assertTrue((brightness>0).all())
        np.testing.assert_allclose(saturation,0)
        np.testing.assert_allclose(warmth,0)
        warm=original.clone();warm[0]+=0.2
        self.assertTrue((_pixel_change_maps(original,warm)[1]>0).all())
        self.assertTrue((_pixel_change_maps(original,warm)[2]>0).all())
        cool=original.clone();cool[2]+=0.2
        self.assertTrue((_pixel_change_maps(original,cool)[2]<0).all())
        heatmap,_=_change_heatmap(np.array([[-0.5,0,0.5]],dtype=np.float32))
        np.testing.assert_array_equal(np.asarray(heatmap),[[[0,0,255],[255,255,255],[255,0,0]]])

    def test_rows_and_plain_first_row(self):
        original=torch.full((3,24,32),0.2)
        output=torch.full_like(original,0.4);target=torch.full_like(original,0.6)
        for count in (1,3):
            refs=original.unsqueeze(0).repeat(count,1,1,1)
            experts=target.unsqueeze(0).repeat(count,1,1,1)
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'panel.png'
                _save_visualization(path,original,refs,experts,output,target,30,0.9,
                    [torch.eye(4) if i%2==0 else None for i in range(count)],2,torch.ones(count,4)/count)
                with Image.open(path) as image:
                    self.assertEqual(image.size,(768,24+(count+3)*90+116))
                    predicted_y=12+(count+1)*90+54+12
                    target_y=predicted_y+90+58
                    predicted_pixel=image.getpixel((132,predicted_y))
                    target_pixel=image.getpixel((132,target_y))
                    self.assertEqual(predicted_pixel[0],255)
                    self.assertTrue(120<=predicted_pixel[1]<=135)
                    self.assertEqual(target_pixel,(255,0,0))
                    for column,value in enumerate((51,102,153)):
                        self.assertEqual(image.getpixel((12+column*252+120,66+12)),(value,)*3)


if __name__=='__main__':unittest.main()
