"""4CH support wrapper binding immutable geometry to the same handoff/scaler."""
from pathlib import Path
import json
from geas35.models.transition.handoff_four_channel_v1 import BoundFourChannelRuntime,checked,contract,CONTRACT_HASH
from geas35.models.transition.four_channel_phase0 import sha
from geas35.rl.support_v1 import StateActionSupportModel
VERSION='geas35.step15.hlv15_b60_r6.4ch.v1'
def load_bound_support(path,*,manifest_sha256,crop,root):
    c=contract(root);path=Path(path)
    if sha(path)!=manifest_sha256:raise ValueError('Support binding hash')
    m=json.loads(path.read_text())
    if m['version']!=VERSION or m['crop']!=crop or m['contract_sha256']!=CONTRACT_HASH:raise ValueError('Support version/crop')
    for record in m['files'].values():checked(root,record)
    for record in m['code']:checked(root,record)
    for key,authority in [('schema','schema'),('scaler','observation_scaler')]:
        if m['files'][key]!=c['selected'][crop][authority]:raise ValueError('Support scale/schema substitution')
    h=checked(root,m['files']['handoff']);rt=BoundFourChannelRuntime(h.parent,crop=crop,manifest_sha256=m['files']['handoff']['sha256'],root=root)
    model=StateActionSupportModel(checked(root,m['files']['support']))
    if model.config['crop']!=crop or model.config['state_columns']!=rt.obs or model.config['action_columns']!=rt.actions:raise ValueError('Support feature mismatch')
    if (model.artifact_path.parent/model.config['arrays_file']).resolve()!=checked(root,m['files']['reference']):raise ValueError('Unbound support arrays')
    if model.reference_state_raw.shape[1]!=56 or model.reference_action_raw.shape[1]!=6:raise ValueError('Legacy geometry')
    return rt,model
