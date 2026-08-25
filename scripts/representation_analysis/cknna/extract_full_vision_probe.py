#!/usr/bin/env python3
"""Stream all full-chunk LIBERO anchors into per-layer vision probe shards."""
from __future__ import annotations

import argparse, dataclasses, hashlib, json, math, os, time
from pathlib import Path

from safetensors.torch import save_file
import torch
from torch.utils.data import Subset

from openpi.models_pytorch import pi0_pytorch
from openpi.training import config as config_lib
from openpi.training import data_loader as data_loader_lib
from extract_policy_features import load_policy
from scripts import train_pytorch as train_lib

LAYERS = 18
PATCHES = 256

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()

def rows(path: Path):
    with path.open() as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def capture(model, observation):
    images,masks,tokens,token_masks,_=model._preprocess_observation(observation,train=False)
    if len(images)!=3: raise ValueError(f"expected 3 views, got {len(images)}")
    embs,pad,att=model.embed_prefix(images,masks,tokens,token_masks)
    (embs,)=model._align_prefix_embeddings_dtype(embs)
    nvision=len(images)*PATCHES
    vmask=torch.cat([m[:,None].expand(-1,PATCHES) for m in masks],dim=1)
    got={}; handles=[]
    for i,layer in enumerate(model.paligemma_with_expert.paligemma.language_model.layers):
        def hook(_m,_inp,out,*,idx=i):
            hidden=out[0] if isinstance(out,tuple) else out
            v=hidden[:,:nvision].float(); w=vmask.to(v.dtype).unsqueeze(-1)
            got[idx]=((v*w).sum(1)/w.sum(1).clamp_min(1)).detach().cpu()[0]
        handles.append(layer.register_forward_hook(hook))
    try:
        mask2=pi0_pytorch.make_att_2d_masks(pad,att)
        pos=torch.cumsum(pad,dim=1)-1
        mask4=model._prepare_attention_masks_4d(mask2,dtype=embs.dtype)
        model.paligemma_with_expert(attention_mask=mask4,position_ids=pos,past_key_values=None,
            inputs_embeds=[embs,None],use_cache=False)
    finally:
        for h in handles: h.remove()
    if sorted(got)!=list(range(LAYERS)): raise RuntimeError(f"captures={sorted(got)}")
    out=torch.stack([got[i] for i in range(LAYERS)])
    if not torch.isfinite(out).all(): raise RuntimeError("non-finite feature")
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config-name",required=True); p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--external-encoder-path",type=Path,required=True); p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--dataset-root",type=Path,required=True); p.add_argument("--assets-base",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--device",default="cuda:0")
    p.add_argument("--num-workers",type=int,default=4); p.add_argument("--target-shard-rows",type=int,default=512)
    p.add_argument("--max-samples",type=int)
    a=p.parse_args(); os.environ["OPENPI_DEPTH_MODEL_PATH"]=str(a.external_encoder_path)
    manifest=list(rows(a.manifest))
    if a.max_samples is not None: manifest=manifest[:a.max_samples]
    indices=[int(x["global_frame_index"]) for x in manifest]
    if [x["sample_index"] for x in manifest]!=list(range(len(manifest))): raise ValueError("bad sample order")
    if a.max_samples is None and len(manifest)!=193569: raise ValueError(f"unexpected samples {len(manifest)}")
    a.output_dir.mkdir(parents=True,exist_ok=True)
    final=a.output_dir/"feature_manifest.json"; progress=a.output_dir/"progress_manifest.json"
    if final.exists(): raise FileExistsError(final)
    manifest_hash=sha(a.manifest); records=[]; completed=0
    if progress.exists():
        state=json.loads(progress.read_text())
        expected={"schema_version":"guidedvla-action-probe-vision-only-progress-v1","source_manifest_sha256":manifest_hash,
                  "config_name":a.config_name,"checkpoint":str(a.checkpoint),"num_samples":len(manifest)}
        if any(state.get(k)!=v for k,v in expected.items()): raise ValueError("progress metadata mismatch")
        records=state["shards"]; completed=sum(x["num_samples"] for x in records)
        if completed and records[-1]["last_sample_index"]!=completed-1: raise ValueError("non-contiguous progress")
        for rec in records:
            path=Path(rec["path"])
            if not path.exists() or sha(path)!=rec["sha256"]: raise ValueError(f"invalid completed shard: {path}")
    elif any(a.output_dir.glob("shard_*.safetensors")):
        raise FileExistsError("orphan shards without progress manifest")
    cfg=config_lib.get_config(a.config_name); base=cfg.data.base_config or config_lib.DataConfig()
    base=dataclasses.replace(base,local_root_dir=str(a.dataset_root)); factory=dataclasses.replace(cfg.data,base_config=base)
    cfg=dataclasses.replace(cfg,data=factory,assets_base_dir=str(a.assets_base),batch_size=1,num_workers=a.num_workers)
    dc=cfg.data.create(cfg.assets_dirs,cfg.model); dc=dataclasses.replace(dc,use_object_loss=False)
    ds=data_loader_lib.create_torch_dataset(dc,action_horizon=cfg.model.action_horizon,model_config=cfg.model,split="all")
    transformed=data_loader_lib.transform_dataset(ds,dc); subset=Subset(transformed,indices[completed:])
    tl=data_loader_lib.TorchDataLoader(subset,local_batch_size=1,shuffle=False,num_batches=len(indices)-completed,num_workers=a.num_workers,seed=20260825,framework="pytorch")
    loader=data_loader_lib.DataLoaderImpl(dc,tl,framework="pytorch")
    device=torch.device(a.device); torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
    model,load_report=load_policy(a.config_name,a.checkpoint,device); started=time.monotonic(); feats=[]; ids=[]
    def flush():
        nonlocal feats,ids
        if not feats:return
        tensor=torch.stack(feats,dim=1).to(torch.bfloat16).contiguous(); sid=torch.tensor(ids,dtype=torch.int64)
        path=a.output_dir/f"shard_{len(records):05d}.safetensors"; tmp=path.with_suffix(".tmp")
        save_file({"vision_features":tensor,"sample_index":sid},tmp); tmp.replace(path)
        records.append({"path":str(path),"sha256":sha(path),"num_samples":len(ids),"first_sample_index":ids[0],"last_sample_index":ids[-1]})
        state={"schema_version":"guidedvla-action-probe-vision-only-progress-v1","source_manifest_sha256":manifest_hash,
               "config_name":a.config_name,"checkpoint":str(a.checkpoint),"num_samples":len(manifest),"shards":records}
        progress_tmp=progress.with_suffix(".tmp"); progress_tmp.write_text(json.dumps(state,indent=2,sort_keys=True)+"\n"); progress_tmp.replace(progress)
        print(json.dumps({"shard":len(records),"samples":ids[-1]+1,"elapsed":time.monotonic()-started}),flush=True)
        feats=[];ids=[]
    prev_ep=None
    with torch.no_grad(),torch.autocast(device_type="cuda",dtype=torch.bfloat16):
        for local_cursor,(obs,_actions,obj) in enumerate(loader):
            if obj is not None: raise ValueError("object targets enabled")
            obs=train_lib.move_observation_to_device(obs,device)
            cursor=completed+local_cursor
            ep=int(manifest[cursor]["episode_index"])
            if prev_ep is not None and ep!=prev_ep and len(ids)>=a.target_shard_rows: flush()
            feats.append(capture(model,obs)); ids.append(cursor); prev_ep=ep
    flush()
    if sum(x["num_samples"] for x in records)!=len(manifest): raise RuntimeError("incomplete")
    result={"schema_version":"guidedvla-action-probe-vision-only-v1","config_name":a.config_name,"load_report":load_report,
      "source_manifest":str(a.manifest),"source_manifest_sha256":manifest_hash,"num_samples":len(manifest),"num_episodes":len({x["episode_index"] for x in manifest}),
      "num_layers":LAYERS,"feature_dim":2048,"feature_location":"paligemma_post_block","pooling":"masked_mean_over_all_valid_camera_patch_tokens",
      "saved_dtype":"torch.bfloat16","elapsed_seconds":time.monotonic()-started,"peak_cuda_memory_bytes":torch.cuda.max_memory_allocated(device),"shards":records}
    tmp=final.with_suffix(".tmp");tmp.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");tmp.replace(final)
    print(json.dumps({"complete":True,"samples":len(manifest),"shards":len(records)}),flush=True)

if __name__=="__main__": main()
