"""Run the assessment pipeline on a free Colab T4 using local inference."""
import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from cleaning import load_rows, save_rows, audit_file
from pipeline import MODEL_ID, encode_example, train_adapter, predict, load_model, release_memory

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results'
DATA=ROOT/'private_data'
BASE=dict(seed=42,steps=100,effective_batch=16,micro_batch=2,learning_rate=2e-4,rank=16,quantized=True,boundary_sampling=False)
RUNS=[dict(BASE,name='A_raw',data='raw'),dict(BASE,name='B_clean',data='clean'),
      dict(BASE,name='C_fp16',data='clean',quantized=False),dict(BASE,name='D_rank32',data='clean',rank=32),
      dict(BASE,name='E_boundary',data='clean',boundary_sampling=True)]


def save_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False))


def prepare(archive):
    DATA.mkdir(parents=True,exist_ok=True);RESULTS.mkdir(parents=True,exist_ok=True)
    # Extract an exact allowlist; ignore path traversal or unexpected archive files.
    names=['data/train.jsonl','data/dev.jsonl','data/test_inputs.jsonl','SCHEMA.md','score.py']
    with zipfile.ZipFile(archive) as z:
        for name in names:
            path=DATA/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(z.read('candidate_pack/'+name))
    clean,summary=audit_file(DATA/'data/train.jsonl',DATA/'audit')
    shutil.copy(DATA/'audit/cleaning_log.csv',ROOT/'cleaning_log.csv')
    shutil.copy(DATA/'audit/audit_summary.json',RESULTS/'audit_summary.json')
    shutil.copy(DATA/'score.py',ROOT/'score.py')
    print('AUDIT:',json.dumps(summary),flush=True)
    return load_rows(DATA/'data/train.jsonl'),clean,load_rows(DATA/'data/dev.jsonl')


def scorer():
    spec=importlib.util.spec_from_file_location('provided_scorer',DATA/'score.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def environment():
    import torch
    import importlib.metadata
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(0):
        raise RuntimeError('This assessment requires a Colab T4 GPU.')
    info={'python':platform.python_version(),'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,
          'packages':{p:importlib.metadata.version(p) for p in ['torch','transformers','peft','accelerate','bitsandbytes','tokenizers','huggingface-hub']}}
    print('ENVIRONMENT:',json.dumps(info),flush=True)
    save_json(RESULTS/'environment.json',info)
    return info


def tokenizer_and_revision(reproduce=False):
    from huggingface_hub import model_info
    from transformers import AutoTokenizer
    config_path=ROOT/'final_config.json'
    if reproduce:
        if not config_path.exists():raise RuntimeError('Run experiments first to select a final configuration.')
        revision=json.loads(config_path.read_text())['revision']
    elif (RESULTS/'model_revision.json').exists():
        revision=json.loads((RESULTS/'model_revision.json').read_text())['revision']
    else:
        revision=model_info(MODEL_ID).sha
        save_json(RESULTS/'model_revision.json',{'model_id':MODEL_ID,'revision':revision})
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=revision)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    return tokenizer,revision


def baseline(tokenizer,revision,dev,rules=False):
    name='baseline_rules' if rules else 'baseline_fixed'
    output=RESULTS/name;output.mkdir(exist_ok=True)
    if (output/'scores.json').exists():
        print('Reuse completed',name,flush=True);return
    model=load_model(revision,False)
    raw=predict(model,tokenizer,dev,rules=(DATA/'SCHEMA.md').read_text() if rules else None)
    save_rows(output/'dev_predictions.jsonl',raw)
    scores=scorer().score(dev,raw);save_json(output/'scores.json',scores)
    print(name,scores['all'],flush=True)
    del model;release_memory()


def experiment(config,raw,clean,dev,tokenizer,revision):
    output=RESULTS/config['name'];output.mkdir(exist_ok=True)
    if (output/'scores.json').exists():
        old=json.loads((output/'config.json').read_text())
        if any(old.get(k)!=v for k,v in config.items()):raise RuntimeError('Existing run config differs; use a new results directory.')
        print('Reuse completed',config['name'],flush=True);return
    rows=raw if config['data']=='raw' else clean
    model,metrics=train_adapter(rows,tokenizer,revision,config,output)
    predictions=predict(model,tokenizer,dev)
    save_rows(output/'dev_predictions.jsonl',predictions)
    scores=scorer().score(dev,predictions);save_json(output/'scores.json',scores)
    print('RESULT',config['name'],scores['all'],flush=True)
    del model;release_memory()


def select_best():
    candidates=[]
    for config in RUNS:
        output=RESULTS/config['name']
        if not (output/'scores.json').exists():raise RuntimeError('Required experiment missing: '+config['name'])
        s=json.loads((output/'scores.json').read_text())['all']
        candidates.append((s['exact_match'],s['mean_field_acc'],-config['rank'],config))
    best=max(candidates,key=lambda x:x[:3])[-1]
    best=dict(best,base_model=MODEL_ID,revision=json.loads((RESULTS/'model_revision.json').read_text())['revision'])
    save_json(ROOT/'final_config.json',best)
    shutil.copytree(RESULTS/best['name']/'adapter',ROOT/'adapter',dirs_exist_ok=True)
    return best


def final_predictions(config,tokenizer,dev):
    from peft import PeftModel
    model=load_model(config['revision'],config['quantized'])
    # Match training-time frozen-layer preparation for QLoRA inference reload.
    if config['quantized']:
        from peft import prepare_model_for_kbit_training
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)
    model=PeftModel.from_pretrained(model,ROOT/'adapter')
    # Compare a fixed random dev sample against the saved trained model's outputs.
    saved={p['id']:p['output'] for p in load_rows(RESULTS/config['name']/'dev_predictions.jsonl')}
    import random
    sample=random.Random(937).sample(dev,16)
    regenerated=predict(model,tokenizer,sample,batch_size=1)
    matches=sum(p['output']==saved[p['id']] for p in regenerated)
    save_json(RESULTS/'reload_check.json',{'matched':matches,'sample_size':len(sample),'batch_size':1})
    if matches!=len(sample):
        raise RuntimeError('Reload/batch-size reproducibility mismatch. Inspect before producing test outputs.')
    # Test is loaded only after selection. No test labels exist or are inferred.
    test=load_rows(DATA/'data/test_inputs.jsonl')
    predictions=predict(model,tokenizer,test,batch_size=1)
    save_rows(ROOT/'predictions_test.jsonl',predictions)
    del model;release_memory()


def save_evaluation_artifacts(config):
    """Save scores and error examples without rewriting the submission documents."""
    rows = []
    run_names = ['baseline_fixed', 'baseline_rules'] + [run['name'] for run in RUNS]

    for name in run_names:
        folder = RESULTS / name
        score_path = folder / 'scores.json'
        if not score_path.exists():
            continue

        scores = json.loads(score_path.read_text())['all']
        metrics_path = folder / 'training_metrics.json'
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        rows.append({
            'run': name,
            'exact_match': scores['exact_match'],
            'mean_field_accuracy': scores['mean_field_acc'],
            'json_valid': scores['json_valid'],
            'peak_gpu_gb': metrics.get('peak_allocated_gb'),
            'training_minutes': metrics.get('training_minutes'),
        })

    if rows:
        with (RESULTS / 'experiment_table.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    dev_rows = load_rows(DATA / 'data/dev.jsonl')
    predictions = {
        row['id']: row['output']
        for row in load_rows(RESULTS / config['name'] / 'dev_predictions.jsonl')
    }
    evaluator = scorer()
    errors = []
    for row in dev_rows:
        expected = json.loads(row['messages'][-1]['content'])
        actual_text = predictions[row['id']]
        actual = evaluator.parse(actual_text)
        wrong = [
            key for key in evaluator.KEYS
            if actual is None or key not in actual or not evaluator.same(actual[key], expected[key])
        ]
        if not wrong and set(actual) == set(evaluator.KEYS):
            continue

        reasons = []
        if actual is None:
            reasons.append('Invalid JSON: generation did not satisfy the output contract.')
        if 'amount_paise' in wrong:
            reasons.append('Amount extraction or unit conversion failed.')
        if 'txn_date' in wrong:
            reasons.append('Date extraction or resolution failed; inspect relative or quoted dates.')
        if any(key in wrong for key in ['priority', 'needs_human']):
            reasons.append('Priority or escalation rule was missed; check the amount and ticket cues.')
        if 'category' in wrong:
            reasons.append('Intent classification failed; inspect overlapping intent and fraud precedence.')
        if 'txn_id' in wrong:
            reasons.append('Transaction-ID copying or normalization failed.')
        if 'channel' in wrong:
            reasons.append('Channel extraction or null decision failed.')
        if 'language' in wrong:
            reasons.append('English/Hinglish classification failed.')
        if actual is not None and set(actual) != set(evaluator.KEYS):
            reasons.append('Output key set differs from the required eight keys.')

        errors.append({
            'id': row['id'],
            'fields': wrong,
            'expected': expected,
            'actual': actual_text,
            'diagnostic_hypothesis': ' '.join(reasons),
        })

    save_json(RESULTS / 'dev_errors.json', errors)
    score_path = RESULTS / config['name'] / 'scores.json'
    final_scores = json.loads(score_path.read_text())['all']
    save_json(RESULTS / 'dev_field_scores.json', final_scores['fields'])

def bundle():
    path=ROOT.parent/'nimbuspay_results.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for file in ROOT.rglob('*'):
            relative=file.relative_to(ROOT)
            if not file.is_file() or any(p in {'private_data','__pycache__','.git'} for p in relative.parts):continue
            # Non-final adapters stay in a separate local backup, not the submission archive.
            if relative.parts[0]=='results' and 'adapter' in relative.parts:continue
            z.write(file,str(relative))
    print('Saved checkpoint archive:',path,flush=True)
    return path


def main(mode,archive):
    started=time.monotonic();raw,clean,dev=prepare(archive);environment()
    tokenizer,revision=tokenizer_and_revision(mode=='reproduce')
    lengths=sorted(len(encode_example(row,tokenizer)['input_ids']) for row in raw)
    stats={'min':lengths[0],'median':lengths[len(lengths)//2],'p95':lengths[int(len(lengths)*.95)],'p99':lengths[int(len(lengths)*.99)],'max':lengths[-1],
           'policy':'No truncation; dynamic padding; microbatch 1 above 768 tokens.'}
    save_json(RESULTS/'length_statistics.json',stats);print('LENGTHS',stats,flush=True)
    if mode=='smoke':
        smoke=dict(BASE,name='smoke',steps=2,data='clean')
        model,_=train_adapter(clean[:32],tokenizer,revision,smoke,RESULTS/'smoke')
        print('SMOKE predictions:',predict(model,tokenizer,dev[:2],batch_size=1),flush=True)
        del model;release_memory();return
    if mode=='experiments':
        baseline(tokenizer,revision,dev,False);bundle()
        baseline(tokenizer,revision,dev,True);bundle()
        for config in RUNS:
            experiment(config,raw,clean,dev,tokenizer,revision);bundle()
        best=select_best();save_evaluation_artifacts(best);bundle()
        # Test generation is deferred to the fresh selected-model reproduction.
    else:
        best=json.loads((ROOT/'final_config.json').read_text())
        model,metrics=train_adapter(raw if best['data']=='raw' else clean,tokenizer,revision,best,RESULTS/'reproduction')
        reproduced=predict(model,tokenizer,dev)
        save_rows(RESULTS/'reproduction/dev_predictions.jsonl',reproduced)
        save_json(RESULTS/'reproduction/scores.json',scorer().score(dev,reproduced))
        # Use the reproduced adapter for the final submitted model and predictions.
        shutil.copytree(RESULTS/'reproduction/adapter',ROOT/'adapter',dirs_exist_ok=True)
        save_rows(RESULTS/best['name']/'dev_predictions.jsonl',reproduced)
        del model;release_memory()
        final_predictions(best,tokenizer,dev)
        elapsed=time.monotonic()-started
        save_json(RESULTS/'reproduction_timing.json',{'runner_seconds':elapsed,'within_3_hours_excluding_notebook_setup':elapsed<10800})
        save_evaluation_artifacts(best);bundle()
        if elapsed>=10800:raise RuntimeError('Reproduction exceeded three hours; report and shorten the recipe.')
    print('DONE',mode,round((time.monotonic()-started)/60,2),'minutes',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['smoke','experiments','reproduce'],required=True)
    parser.add_argument('--archive',required=True);args=parser.parse_args();main(args.mode,args.archive)
