"""Package local pilot evidence and adapters; no credentials or Gemini outputs."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import numpy as np

ROOT=Path('data/openjev-training-pilot')

def main():
    runs=[json.loads((ROOT/d/'analysis.json').read_text()) for d in ['outputs','outputs-seed819','outputs-seed820']]
    summary={'training_seeds':[818,819,820],'validation_records':96,'calibration_records':48,
        'training_records_consumed_per_run':256,'steps_per_run':128,'trainable_parameters':5591040,
        'baseline':runs[0]['baseline'],'baseline_calibrated':runs[0]['baseline_calibrated'],
        'adapted_mean':{},'adapted_calibrated_mean':{},'per_seed':runs,
        'conclusion':'Feasible and promising for probability quality; accuracy gain not established; no game-skill gain; canvas-seed stability worsened.'}
    for key in ['accuracy','nll','brier','ece_10_bins','two_seed_prediction_disagreement']:
        for src,dst in [('adapted','adapted_mean'),('adapted_calibrated','adapted_calibrated_mean')]:
            values=[r[src][key] for r in runs]
            summary[dst][key]={'mean':float(np.mean(values)),'min':min(values),'max':max(values)}
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
    docs=Path('dataset/training-pilot');docs.mkdir(exist_ok=True)
    for name in ['summary.json','data-report.json','cost-and-cleanup.json','fresh-process-reload.json']:
        shutil.copy2(ROOT/name,docs/name)
    text='''# OpenJev pilot results — 2026-09-20

The pipeline works on one RTX PRO 6000, but this pilot supports a probability-quality improvement, not a demonstrated improvement in decision accuracy or game strength.

Three rank-16 decoder-only LoRA runs used the same 256 training records, 128 optimizer updates, and seeds 818/819/820. All were evaluated on the same 96 held-out records (199 slots, each with two canvas seeds). Temperatures were fitted on a separate 48-record calibration set. No checkpoint or seed was selected for best validation performance.

| Model | Accuracy | NLL ↓ | Brier ↓ | ECE ↓ |
|---|---:|---:|---:|---:|
| Frozen base | 64.82% | 1.295 | 0.558 | 0.233 |
| Frozen base + temperature | 64.82% | 0.822 | 0.457 | 0.108 |
| Adapter, mean of 3 runs | MEANACC | MEANNLL | MEANBRIER | MEANECE |
| Adapter + temperature, mean of 3 runs | CALACC | CALNLL | CALBRIER | CALECE |

Accuracy improvements were only 0.75–1.01 percentage points; every paired record-bootstrap 95% interval included zero. After calibrating both models, the adapter NLL improved by 0.049–0.088. Two of three per-run NLL intervals excluded zero; these are correlated results on one small evaluation set, not three independent evaluation datasets.

Probability calibration is therefore worth testing first: it delivers much of the benefit without training. The adapter adds a smaller, promising improvement. None of this establishes production reliability.

The games slice contains only six chess and six Gomoku records. Solver-acceptable move accuracy was 4.17% for the base and 4.17%, 4.17%, and 0% for the three adapters. Do not claim stronger game play. Simple policy-based workflows were already 100% correct before training, so that subset is too easy to demonstrate a gain.

Canvas-seed prediction disagreement worsened from 4.52% to 9.05–11.06%. The actual serving engine's reread rate was not measured. A next pilot should include a consistency objective or richer noise coverage and a larger untouched game evaluation, alongside a temperature-only control. Do not scale to a large paid run on these results alone.

One 96 GB RTX PRO 6000 completed the experiment. Observed Runpod balance decrease: $0.5834; estimated Gemini generation: $0.0518; combined approximately $0.64. Runpod's itemized billing had not populated at cleanup, so the GPU figure is a balance delta. The pod was deleted and the provider reported no remaining pods and zero hourly spend. No network volume was created.

All three adapters and logits are saved locally. Remote/local adapter SHA-256 hashes match. A fresh-process reload of seed 818 reproduced 28 held-out slot-logit vectors across 12 records exactly; seeds 819/820 passed in-process reload checks. Repository tests: 68 passed.

The adapters use frozen BF16 DiffusionGemma with decoder-only query/value updates. They are custom PyTorch adapters, not directly usable in the current NVFP4/vLLM server. No images were trained on or evaluated in this experiment. Deployment parity and larger held-out evaluation remain necessary.

Gemini 3.7 Flash generated 96 examples for about five cents. Those samples are excluded from training and this release because of the [Gemini Developer API restriction on developing competing models](https://ai.google.dev/gemini-api/terms). Original, mechanically labeled workflow records replaced them. Source records retain Apache-2.0, CC BY 4.0, BSD-3-Clause, or CC0 terms; `sources.jsonl` records provenance. DiffusionGemma itself and these adapter modifications are Apache-2.0.

See [the experiment methodology](../PILOT-TRAINING.md) and `summary.json` for exact metrics and uncertainty. The separate 100K-image pool is complete, audited, and packaged; its locked 10K evaluation split was untouched.
'''
    for prefix,key in [('MEAN','adapted_mean'),('CAL','adapted_calibrated_mean')]:
        for suffix,metric in [('ACC','accuracy'),('NLL','nll'),('BRIER','brier'),('ECE','ece_10_bins')]:
            value=summary[key][metric]['mean'];formatted=f'{value*100:.2f}%' if metric=='accuracy' else f'{value:.3f}'
            text=text.replace(prefix+suffix,formatted)
    (docs/'RESULTS.md').write_text(text)
    shutil.copy2(docs/'RESULTS.md',ROOT/'RESULTS.md')
    release=ROOT/'release';release.mkdir(exist_ok=True)
    for name in ['train.jsonl','validation.jsonl','calibration.jsonl','sources.jsonl','data-report.json','summary.json',
                 'cost-and-cleanup.json','fresh-process-reload.json','adapter-sha256.txt','environment.txt','gpu.txt',
                 'train.log','replications.log','verify.log','install.log','RESULTS.md']:
        shutil.copy2(ROOT/name,release/name)
    for folder in ['outputs','outputs-seed819','outputs-seed820','procedural']:
        shutil.copytree(ROOT/folder,release/folder,dirs_exist_ok=True)
    for name in ['LICENSE','pyproject.toml','README.md','dataset/PILOT-TRAINING.md','dataset/LICENSE-DATA.md','dataset/training_sources.yaml',
                 'dataset/games/LICENSE-DATA.md','dataset/games/training_sources.yaml',
                 'scripts/train_slot_pilot.py','scripts/verify_saved_pilot.py','scripts/prepare_training_pilot.py',
                 'scripts/generation_replay.py','scripts/collect_thinking_replay.py',
                 'scripts/build_procedural_pilot.py','scripts/analyze_training_pilot.py','scripts/package_training_pilot.py',
                 'scripts/provision_pilot.py','tests/test_training_pilot.py']:
        src=Path(name)
        if src.exists():
            dest=release/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    shutil.copytree('openjev',release/'openjev',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    for folder in ['dataset/notices','dataset/games/notices']:
        if Path(folder).exists():shutil.copytree(folder,release/folder,dirs_exist_ok=True)
    (release/'RESULTS.md').write_text(text.replace('../PILOT-TRAINING.md','dataset/PILOT-TRAINING.md'))
    files=sorted(p for p in release.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
    (release/'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(release)}\n' for p in files))
    archive=ROOT/'openjev-slot-pilot.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:tar.add(release,arcname='openjev-slot-pilot')
    metadata={'path':str(archive),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}
    (ROOT/'release-summary.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata))

if __name__=='__main__':main()
