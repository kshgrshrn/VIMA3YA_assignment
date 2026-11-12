"""Small, explicit training loop and unchanged-input greedy inference."""
import gc
import json
import random
import time
from pathlib import Path

MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
TARGETS = ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']


def encode_example(row, tokenizer):
    prefix = tokenizer.apply_chat_template(row['messages'][:2], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(row['messages'], tokenize=True, add_generation_prompt=False)
    if full[:len(prefix)] != prefix:
        raise ValueError('Chat-template prefix does not match: cannot safely mask labels.')
    labels = [-100]*len(prefix) + full[len(prefix):]
    if not any(x != -100 for x in labels):
        raise ValueError('Empty training target')
    return {'input_ids':full,'labels':labels}


def sample_indices(weights, count, seed):
    return random.Random(seed).choices(range(len(weights)), weights=weights, k=count)


def validate_predictions(predictions, expected_ids):
    ids = [p['id'] for p in predictions]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_ids) or len(ids) != len(expected_ids):
        raise ValueError('Prediction IDs do not match the input IDs exactly.')
    if any(not isinstance(p.get('output'),str) or set(p) != {'id','output'} for p in predictions):
        raise ValueError('Invalid prediction envelope')
    return predictions  # Raw invalid model JSON remains invalid, as required.


def load_model(revision, quantized=False, train=False, rank=16):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    kwargs = dict(revision=revision, torch_dtype=torch.float16, device_map={'':0}, attn_implementation='sdpa')
    if quantized:
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID,**kwargs)
    if train:
        if quantized:
            model = prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)
        model = get_peft_model(model,LoraConfig(r=rank,lora_alpha=rank*2,lora_dropout=0.05,
            bias='none',task_type='CAUSAL_LM',target_modules=TARGETS))
        # fp32 trainable adapters work with AMP's gradient scaler on T4.
        for parameter in model.parameters():
            if parameter.requires_grad: parameter.data = parameter.data.float()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.config.use_cache = False
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if not 0 < count <= 50_000_000: raise ValueError(f'Invalid trainable parameter count: {count}')
        print('Trainable parameters:',count,flush=True)
    return model


def collate(examples, pad_id):
    import torch
    length=max(len(x['input_ids']) for x in examples)
    ids=[];masks=[];labels=[]
    for example in examples:
        n=len(example['input_ids']);pad=length-n
        ids.append(example['input_ids']+[pad_id]*pad)
        masks.append([1]*n+[0]*pad)
        labels.append(example['labels']+[-100]*pad)
    return {k:torch.tensor(v,dtype=torch.long,device='cuda') for k,v in
            [('input_ids',ids),('attention_mask',masks),('labels',labels)]}


def train_adapter(rows, tokenizer, revision, config, output_dir):
    import torch
    from transformers import set_seed
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    set_seed(config['seed'])
    model=load_model(revision,config['quantized'],train=True,rank=config['rank'])
    encoded=[encode_example(row,tokenizer) for row in rows]
    max_length=max(len(x['input_ids']) for x in encoded)
    if max_length > model.config.max_position_embeddings:
        raise ValueError('Training example exceeds model context; never silently truncate.')
    weights=[]
    for row in rows:
        try: amount=json.loads(row['messages'][-1]['content']).get('amount_paise')
        except json.JSONDecodeError:amount=None
        near=type(amount) is int and any(abs(amount-threshold)<=threshold*0.1 for threshold in (500000,5000000))
        weights.append(3 if config['boundary_sampling'] and near else 1)
    total=config['steps']*config['effective_batch']
    order=sample_indices(weights,total,config['seed'])
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['learning_rate'],weight_decay=0.01)
    scaler=torch.amp.GradScaler('cuda')
    model.train();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.monotonic()
    logs=[];successful_steps=0
    # Same exact sample budget for all runs, including raw versus cleaned.
    for step in range(config['steps']):
        selected=order[step*config['effective_batch']:(step+1)*config['effective_batch']]
        optimizer.zero_grad(set_to_none=True)
        learning_rate=config['learning_rate']*min(1.0,(step+1)/max(1,int(config['steps']*0.05)))
        learning_rate*=0.5*(1+__import__('math').cos(__import__('math').pi*step/config['steps']))
        for group in optimizer.param_groups:group['lr']=learning_rate
        # Use pairs for short examples; process long quoted threads individually.
        batches=[];position=0
        while position<len(selected):
            remaining=selected[position:position+config['micro_batch']]
            if max(len(encoded[i]['input_ids']) for i in remaining)>768:remaining=remaining[:1]
            batches.append(remaining);position+=len(remaining)
        # Normalize across supervised tokens in the entire effective batch.
        target_count=sum(sum(t!=-100 for t in encoded[i]['labels'][1:]) for i in selected)
        loss_sum=0.0
        for group in batches:
            batch=collate([encoded[i] for i in group],tokenizer.pad_token_id)
            n_targets=(batch['labels'][:,1:]!=-100).sum().item()
            with torch.autocast('cuda',dtype=torch.float16):
                loss=model(**batch).loss*(n_targets/target_count)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss; stop rather than report a fake run.')
            scaler.scale(loss).backward();loss_sum+=loss.detach().float().item()
            del batch,loss
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
        old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
        successful_steps+=int(scaler.get_scale()>=old_scale)
        logs.append({'step':step+1,'loss':loss_sum,'learning_rate':learning_rate})
        if (step+1)%10==0 or step==0:
            print(f"{config['name']}: {step+1}/{config['steps']}, loss={loss_sum:.4f}, elapsed={(time.monotonic()-started)/60:.1f} min",flush=True)
    torch.cuda.synchronize()
    metrics={'training_minutes':(time.monotonic()-started)/60,'peak_allocated_gb':torch.cuda.max_memory_allocated()/2**30,
             'peak_reserved_gb':torch.cuda.max_memory_reserved()/2**30,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
             'examples_sampled':total,'successful_optimizer_steps':successful_steps,'max_training_tokens':max_length}
    model.save_pretrained(output_dir/'adapter',safe_serialization=True)
    tokenizer.save_pretrained(output_dir/'adapter')
    (output_dir/'loss_log.json').write_text(json.dumps(logs,indent=2))
    (output_dir/'training_metrics.json').write_text(json.dumps(metrics,indent=2))
    (output_dir/'config.json').write_text(json.dumps(dict(config,base_model=MODEL_ID,revision=revision),indent=2))
    del optimizer,scaler
    model.gradient_checkpointing_disable();model.config.use_cache=True;model.eval()
    return model,metrics


def predict(model, tokenizer, rows, batch_size=8, rules=None):
    """No added rules except the explicitly designated reference baseline."""
    import torch
    from transformers import GenerationConfig
    model.eval();model.config.use_cache=True
    prompts=[]
    for row in rows:
        messages=[dict(m) for m in row['messages'] if m['role'] in {'system','user'}]
        if rules is not None:messages[0]['content']+='\n\n'+rules
        prompts.append(tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True))
    # Same public decoding contract in every run. No grammar or JSON repair.
    generation=GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=200,
        eos_token_id=model.generation_config.eos_token_id,pad_token_id=tokenizer.pad_token_id,
        bos_token_id=model.generation_config.bos_token_id,use_cache=True)
    results=[];tokenizer.padding_side='left'
    index=0;started=time.monotonic()
    while index<len(rows):
        group_size=min(batch_size,len(rows)-index)
        # Long input threads need smaller batches; inputs themselves are untouched.
        if max(len(tokenizer.encode(p,add_special_tokens=False)) for p in prompts[index:index+group_size])>1024:
            group_size=1
        batch=tokenizer(prompts[index:index+group_size],return_tensors='pt',padding=True,add_special_tokens=False).to('cuda')
        if batch['input_ids'].shape[1]+200>model.config.max_position_embeddings:
            raise ValueError('Input exceeds available context.')
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            generated=model.generate(**batch,generation_config=generation)
        new_tokens=generated[:,batch['input_ids'].shape[1]:]
        outputs=tokenizer.batch_decode(new_tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        for row,output in zip(rows[index:index+group_size],outputs):
            results.append({'id':row['id'],'output':output.strip()})
        index+=group_size
        if index%40==0 or index==len(rows):print(f'Predicted {index}/{len(rows)} in {(time.monotonic()-started)/60:.1f} min',flush=True)
        del batch,generated,new_tokens
    return validate_predictions(results,[r['id'] for r in rows])


def release_memory():
    import torch
    gc.collect();torch.cuda.empty_cache()
