"""Load research support only through its explicit B60/R6 hash binding."""
from pathlib import Path
import json
from geas35.models.transition.handoff_b60_r6_v1 import BoundTransitionRuntime,sha
from geas35.rl.support_v1 import StateActionSupportModel

def load_bound_research_artifacts(binding_path, *, binding_sha256, crop, root):
    root=Path(root);binding_path=Path(binding_path)
    if sha(binding_path)!=binding_sha256:raise ValueError('Unbound support manifest')
    binding=json.loads(binding_path.read_text())
    if binding['version']!='geas35.step15.b60_r6.v1' or binding['crop']!=crop:raise ValueError('Support crop/version mismatch')
    for key in ('handoff_manifest','input_scaler','schema','train','support','reference','fit_registry','protocol'):
        record=binding[key]
        if sha(root/record['path'])!=record['sha256']:raise ValueError('Support dependency mismatch: '+key)
    handoff=binding['handoff_manifest']
    runtime=BoundTransitionRuntime((root/handoff['path']).parent,crop=crop,manifest_sha256=handoff['sha256'],root=root)
    support=StateActionSupportModel(root/binding['support']['path'])
    if support.config['crop']!=crop or support.config['state_columns']!=runtime.obs or support.config['action_columns']!=runtime.actions:
        raise ValueError('Support/runtime schema mismatch')
    if binding['input_scaler']['sha256']!=runtime.manifest['bundle_hashes']['observation_scaler.json'] or binding['schema']['sha256']!=runtime.manifest['bundle_hashes']['schema.json']:
        raise ValueError('Support/runtime scale mismatch')
    expected=(root/binding['support']['path']).parent/support.config['arrays_file']
    if expected.resolve()!=(root/binding['reference']['path']).resolve():raise ValueError('Unbound support array path')
    return runtime,support
