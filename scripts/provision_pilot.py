"""Provision exactly one 96GB RTX PRO 6000 with provider-side termination.

CLI 2.14 lacks the documented terminate-after flag, so creation uses the
documented GraphQL input. All subsequent lifecycle operations use runpodctl.
No credentials are written to artifacts or sent into the pod.
"""
import datetime
import json
from pathlib import Path
import subprocess
import tomllib
import requests

ROOT=Path('data/openjev-training-pilot')
def main():
    if (ROOT/'pod.json').exists():raise RuntimeError('Pod already provisioned: inspect before any retry')
    key=tomllib.loads((Path.home()/'.runpod/config.toml').read_text())['apikey']
    deadline=datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(hours=2)
    payload={'name':'openjev-slot-pilot','cloudType':'SECURE','gpuTypeId':'NVIDIA RTX PRO 6000 Blackwell Server Edition',
        'gpuCount':1,'imageName':'runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404','containerDiskInGb':100,
        'volumeInGb':0,'ports':'22/tcp','startSsh':True,'supportPublicIp':True,
        'terminateAfter':deadline.isoformat(),'minMemoryInGb':96,'minVcpuCount':8,
        'env':[{'key':'PUBLIC_KEY','value':(Path.home()/'.runpod/ssh/runpodctl-ssh-key.pub').read_text().strip()}]}
    response=requests.post('https://api.runpod.io/graphql',headers={'Authorization':'Bearer '+key},
        json={'query':'mutation($input: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input:$input) { id costPerHr gpuCount imageName createdAt desiredStatus } }',
              'variables':{'input':payload}},timeout=90)
    response.raise_for_status();result=response.json()
    if result.get('errors'):raise RuntimeError(str(result['errors']))
    pod=result['data']['podFindAndDeployOnDemand'];pod['terminate_after']=deadline.isoformat()
    (ROOT/'pod.json').write_text(json.dumps(pod,indent=2));print(json.dumps(pod),flush=True)
    if pod['costPerHr']>2.3 or pod['gpuCount']!=1:
        subprocess.run(['runpodctl','pod','delete',pod['id']],check=True)
        raise RuntimeError('Pod exceeds experiment resource/price cap; deleted')

if __name__=='__main__':main()
