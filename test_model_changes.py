import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from attention import CrossAttention
from models import LatentRetouchModel, EnhancementUNet
from trainer import Trainer
from config import Config


torch.set_num_threads(1)


class MemoryDataset:
    def __init__(self, count):
        self.records=[{'id':f'{i}_A'} for i in range(count)]
    def __len__(self):return len(self.records)
    def load(self,index):
        x=np.full((3,16,16),index/max(1,len(self)),dtype=np.float32)
        return x,x,self.records[index]['id']
    def __getitem__(self,index):
        x,y,key=self.load(index)
        return {'original':x,'expert':y,'index':index,'id':key}


def model(**kwargs):
    return LatentRetouchModel(8,8,2,2,8,8,8,8,**kwargs)


class ChangesTest(unittest.TestCase):
    def test_full_attention_equivalence_and_gradients(self):
        attn=CrossAttention(3,8,2)
        q=torch.randn(2,4,3,requires_grad=True)
        k=torch.randn(2,4,3,requires_grad=True)
        v=torch.randn(2,3,4,5,requires_grad=True)
        context,a=attn(q,k,values=v,query_size=(2,2),key_size=(2,2))
        projected_q=attn.q(q).reshape(2,4,2,4).transpose(1,2)
        projected_k=attn.k(k).reshape(2,4,2,4).transpose(1,2)
        heads=(projected_q@projected_k.transpose(-1,-2)*attn.scale).abs().softmax(-1)
        torch.testing.assert_close(a,heads.mean(1))
        keys=F.interpolate(heads.reshape(4,4,2,2),size=(4,5),mode='bilinear',align_corners=False).flatten(2)
        keys=keys/keys.sum(-1,keepdim=True)
        full=F.interpolate(keys.transpose(1,2).reshape(4,20,2,2),size=(3,4),mode='bilinear',align_corners=False).flatten(2).transpose(1,2).reshape(2,2,12,20)
        projected_v=attn.v(v.flatten(2).transpose(1,2)).reshape(2,20,2,4).transpose(1,2)
        expected=attn.out((full@projected_v).transpose(1,2).reshape(2,12,8)).transpose(1,2).reshape(2,8,3,4)
        actual=F.interpolate(context,size=(3,4),mode='bilinear',align_corners=False)
        torch.testing.assert_close(actual,expected)
        actual.square().mean().backward()
        for value in (q,k,v,attn.v.weight,attn.out.weight):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(),0)

    def test_model_full_features_and_ablations(self):
        for options in ({},{'ablation_uniform_weight':True},{'ablation_cnn_fusion':True}):
            net=model(**options)
            x=torch.rand(1,3,32,32);refs=torch.rand(1,2,3,32,32)
            captured=[]
            handle=net.enhancer.register_forward_pre_hook(lambda module,args:captured.append(args[1]))
            out,scores,weights=net(x,None,refs,refs)
            handle.remove()
            self.assertEqual(captured[0].shape,(1,8,8,8))
            self.assertEqual(out.shape,x.shape)
            torch.testing.assert_close(weights.sum(1),torch.ones(1,4))
            out.mean().backward()
            if options.get("ablation_cnn_fusion"):
                self.assertIsNone(net.a.a.net[0].weight.grad)
            else:
                self.assertGreater(net.a.a.net[0].weight.grad.abs().sum().item(),0)
                for layer in (net.cross_attention.v,net.cross_attention.out,net.context_edit):
                    self.assertGreater(layer.weight.grad.abs().sum().item(),0)
            encoder=net.cnn_fusion if options.get("ablation_cnn_fusion") else net.b
            self.assertGreater(encoder.a.net[0].weight.grad.abs().sum().item(),0)
        net=model()
        with torch.no_grad():
            for param in net.b.parameters():param.zero_()
            net.cross_attention.v.bias.zero_();net.cross_attention.out.bias.zero_();net.context_edit.bias.zero_()
        captured=[]
        net.enhancer.register_forward_pre_hook(lambda module,args:captured.append(args[1]))
        net(x,None,refs,refs)
        torch.testing.assert_close(captured[0],torch.zeros_like(captured[0]))

    def test_residual_rgb(self):
        net=EnhancementUNet(8,8,8)
        with torch.no_grad():
            net.output.weight.zero_();net.output.bias.zero_()
        x=torch.rand(1,3,16,16)
        edit=torch.rand(1,8,4,4)
        torch.testing.assert_close(net(x,edit),x)
        with torch.no_grad():net.output.bias.fill_(1.0)
        torch.testing.assert_close(net(x,edit),(x+torch.tanh(torch.tensor(1.0))*0.25).clamp(0,1))

    def test_chunks_refresh_only_current_queries(self):
        data=MemoryDataset(203)
        with tempfile.TemporaryDirectory() as folder:
            cfg=Config(device='cpu',num_workers=0,batch_size=4,batches_per_chunk=25,output_dir=folder)
            Path(cfg.paths()['similarity']).mkdir()
            trainer=Trainer(model(),cfg,data,data)
            calls=[];original=trainer.update_grouping
            def refresh(*args,**kwargs):
                result=original(*args,**kwargs)
                calls.append((list(kwargs['query_indices']),trainer.model.b.a.net[0].weight.clone()))
                return result
            trainer.update_grouping=refresh
            seen=[]
            for batch in trainer.epoch_batches(data,'train',1,True):
                self.assertTrue(trainer.model.training)
                seen.extend(batch['index'].tolist())
                with torch.no_grad():trainer.model.b.a.net[0].weight.add_(0.001)
            self.assertEqual([len(c[0]) for c in calls],[100,100,3])
            self.assertEqual(sorted(seen),list(range(203)))
            self.assertFalse(torch.equal(calls[0][1],calls[1][1]))
            matrices=sorted(Path(cfg.paths()['similarity']).glob('*.npy'))
            self.assertEqual([np.load(p).shape for p in matrices],[(100,203),(100,203),(3,203)])
            for index,refs in trainer.references['train'].items():self.assertNotIn(index,refs)
            cfg.ablation_random_reference=True
            trainer.update_grouping(data,'train',2,query_indices=[7,2],save_results=False)
            self.assertEqual(len(trainer.references['train']),203)


if __name__=='__main__':unittest.main()
