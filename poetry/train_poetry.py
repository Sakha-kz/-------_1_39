"""Modern Persian GPT-2 adaptation; compare against fetched upstream before claiming exact reproduction."""
import argparse,csv,hashlib,json,math,random,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

def dump(p,obj):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def prepare(tsv,text_column,poet_column=None,limit=None,seed=42):
    # Split entire poems, not tokens or lines from the same poem.
    with Path(tsv).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f,delimiter='\t'); fields=reader.fieldnames
        if text_column not in fields: raise ValueError(f'Text column {text_column!r} missing. Actual columns: {fields}')
        if poet_column and poet_column not in fields: raise ValueError(f'Poet column missing: {fields}')
        poems={}
        for r in reader:
            text=r[text_column].strip()
            if len(text)<30: continue
            key=hashlib.sha256(text.encode()).hexdigest()
            poems[key]=dict(id=key,text=text,poet=r[poet_column].strip() if poet_column else 'ناشناس')
    items=sorted(poems.values(),key=lambda r:r['id']); random.Random(seed).shuffle(items)
    if limit: items=items[:limit]
    if len(items)<30: raise ValueError('Need >=30 distinct full poems')
    n=max(1,int(len(items)*.1)); return items[n:],items[:n],fields

class Poems(Dataset):
    def __init__(self,rows,tokenizer,block=128):
        self.samples=[]
        for row in rows:
            text='<s>'+row['poet']+'<|startoftext|>'+row['text']+tokenizer.eos_token
            ids=tokenizer(text,add_special_tokens=False)['input_ids']
            # Non-overlapping chunks, all from one split; keep final chunk.
            for i in range(0,len(ids),block):
                part=ids[i:i+block]
                if len(part)>1: self.samples.append(dict(input_ids=part,attention_mask=[1]*len(part)))
    def __len__(self): return len(self.samples)
    def __getitem__(self,i): return self.samples[i]

def collator(tokenizer):
    def pad(batch):
        out=tokenizer.pad(batch,return_tensors='pt'); labels=out['input_ids'].clone()
        # Mask padding via attention mask, not token ID: pad may equal EOS.
        labels[out['attention_mask']==0]=-100; out['labels']=labels; return out
    return pad

@torch.inference_mode()
def evaluate_tokens(model,ds,pad,device):
    from torch.utils.data import DataLoader
    total=count=0; model.eval()
    for batch in DataLoader(ds,batch_size=2,collate_fn=pad):
        batch={k:v.to(device) for k,v in batch.items()}; outputs=model(**batch)
        n=int((batch['labels'][:,1:]!=-100).sum()); total+=outputs.loss.item()*n; count+=n
    loss=total/max(count,1); return dict(loss=loss,perplexity=math.exp(min(loss,80)),predicted_tokens=count)

def run(args):
    from transformers import AutoTokenizer,AutoModelForCausalLM,Trainer,TrainingArguments,set_seed
    from transformers.trainer_utils import get_last_checkpoint
    from huggingface_hub import HfApi
    set_seed(args.seed); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    train_rows,val_rows,fields=prepare(args.tsv,args.text_column,args.poet_column,args.limit,args.seed)
    model_id='HooshvareLab/gpt2-fa'; sha=HfApi().model_info(model_id).sha
    tok=AutoTokenizer.from_pretrained(model_id,revision=sha)
    tok.add_special_tokens({'bos_token':'<s>','additional_special_tokens':['<|startoftext|>']})
    if tok.eos_token is None: tok.add_special_tokens({'eos_token':'<|endoftext|>'})
    tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(model_id,revision=sha); model.resize_token_embeddings(len(tok))
    model.config.pad_token_id=tok.pad_token_id; model.config.use_cache=False
    device='cuda' if torch.cuda.is_available() else 'cpu'; model.to(device); model.gradient_checkpointing_enable()
    tr=Poems(train_rows,tok,args.block); va=Poems(val_rows,tok,args.block); pad=collator(tok)
    dataset_hash=hashlib.sha256(Path(args.tsv).read_bytes()).hexdigest()
    provenance=dict(model=model_id,revision=sha,seed=args.seed,tsv_sha256=dataset_hash,columns=fields,
        text_column=args.text_column,poet_column=args.poet_column,train_poems=len(train_rows),val_poems=len(val_rows),
        train_ids=[r['id'] for r in train_rows],val_ids=[r['id'] for r in val_rows],block=args.block,limit=args.limit)
    old=out/'provenance.json'
    if old.exists() and json.loads(old.read_text(encoding='utf-8'))!=provenance: raise ValueError('Model or dataset revision changed. Use another output folder to avoid invalid resume.')
    dump(old,provenance)
    prompts=['<s>'+r['poet']+'<|startoftext|>' for r in val_rows[:3]]
    def generate():
        model.eval(); model.config.use_cache=True; rows=[]
        for p in prompts:
            set_seed(args.seed); inp=tok(p,return_tensors='pt').to(device)
            ids=model.generate(**inp,max_new_tokens=80,do_sample=True,temperature=.8,top_p=.9,pad_token_id=tok.pad_token_id)
            rows.append(dict(prompt=p,output=tok.decode(ids[0],skip_special_tokens=False)))
        model.config.use_cache=False; return rows
    # Save genuine baseline before training; preserve it across checkpoint resumes.
    if not (out/'before.json').exists(): dump(out/'before.json',dict(evaluation=evaluate_tokens(model,va,pad,device),generations=generate()))
    training=TrainingArguments(output_dir=str(out/'checkpoints'),num_train_epochs=args.epochs,per_device_train_batch_size=1,
        per_device_eval_batch_size=2,gradient_accumulation_steps=8,learning_rate=5e-5,warmup_ratio=.05,
        eval_strategy='epoch',save_strategy='epoch',logging_steps=10,load_best_model_at_end=True,
        save_total_limit=2,fp16=device=='cuda',report_to='none',seed=args.seed,data_seed=args.seed,
        dataloader_num_workers=0,remove_unused_columns=False)
    trainer=Trainer(model=model,args=training,train_dataset=tr,eval_dataset=va,data_collator=pad,processing_class=tok)
    latest=get_last_checkpoint(str(out/'checkpoints')) if (out/'checkpoints').exists() else None
    start=time.perf_counter(); trainer.train(resume_from_checkpoint=latest)
    model.save_pretrained(out/'final'); tok.save_pretrained(out/'final')
    dump(out/'after.json',dict(evaluation=evaluate_tokens(model,va,pad,device),generations=generate(),train_seconds=time.perf_counter()-start,resumed_from=latest))
    dump(out/'training_history.json',trainer.state.log_history)
    before=json.loads((out/'before.json').read_text(encoding='utf-8')); after=json.loads((out/'after.json').read_text(encoding='utf-8'))
    report=['### Сахтай Е.Н. · CSE8842\n### Преподаватель Ботайбеков А.О.',
        '# Persian Poetry Fine-Tuning',f'Базовая модель: {model_id}; revision: {sha}. Seed {args.seed}.',
        f'Данные: {len(train_rows)} полных стихотворений train, {len(val_rows)} validation; block size {args.block}.',
        '| Этап | Token loss | Perplexity |\n|---|---:|---:|',
        f'| До | {before["evaluation"]["loss"]:.5f} | {before["evaluation"]["perplexity"]:.5f} |',
        f'| После | {after["evaluation"]["loss"]:.5f} | {after["evaluation"]["perplexity"]:.5f} |',
        '## Генерации при одинаковых prompts и seed']
    for a,b in zip(before['generations'],after['generations']):
        report+=['### Промпт\n'+a['prompt'],'До:\n```text\n'+a['output']+'\n```','После:\n```text\n'+b['output']+'\n```']
    report+=['## Интерпретация\nСравните связность, повторения, соответствие авторскому стилю. Перечислите отличия от исходного notebook. Perplexity не измеряет художественное качество.']
    (out/'README.md').write_text('\n\n'.join(report),encoding='utf-8')
    import matplotlib.pyplot as plt
    logs=trainer.state.log_history; fig,ax=plt.subplots(figsize=(8,4))
    for key in ['loss','eval_loss']:
        rs=[r for r in logs if key in r]; ax.plot([r.get('epoch',0) for r in rs],[r[key] for r in rs],label=key)
    ax.set(xlabel='Epoch',ylabel='Loss'); ax.legend(); fig.tight_layout(); fig.savefig(out/'loss.png',dpi=140); plt.close(fig)
    import importlib.metadata
    dump(out/'environment.json',{p:importlib.metadata.version(p) for p in ['torch','transformers','datasets','accelerate','tokenizers']})
    print('Finished. Inspect before/after generations; perplexity is not a poetry-quality score.')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--tsv',required=True); p.add_argument('--text-column',required=True); p.add_argument('--poet-column'); p.add_argument('--out',default='poetry/results')
    p.add_argument('--limit',type=int,default=5000); p.add_argument('--block',type=int,default=128); p.add_argument('--epochs',type=int,default=1); p.add_argument('--seed',type=int,default=42)
    run(p.parse_args())
