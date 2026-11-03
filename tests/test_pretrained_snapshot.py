from pathlib import Path
import pytest
from scripts.refine_supervised import resolve_backbone


def weights(path):
    path.mkdir();(path/'config.json').write_text('{}');(path/'model.safetensors').write_bytes(b'fixture-only')
    return path


def test_local_backbone_never_downloads_and_tracks_content(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('unexpected network access')
    monkeypatch.setattr('huggingface_hub.snapshot_download',forbidden)
    root=weights(tmp_path/'local')
    directory,meta=resolve_backbone(str(root))
    assert Path(directory)==root and meta['revision'] is None
    assert set(meta['file_sha256'])=={'config.json','model.safetensors'}
    with pytest.raises(ValueError,match='revision'):resolve_backbone(str(root),'main')


def test_hub_revision_is_resolved_once_then_trials_use_local_files(tmp_path,monkeypatch):
    root=weights(tmp_path/'fixed-commit');calls=[]
    def snapshot(*args,**kwargs):calls.append((args,kwargs));return str(root)
    monkeypatch.setattr('huggingface_hub.snapshot_download',snapshot)
    directory,meta=resolve_backbone('org/backbone','immutable-sha')
    assert len(calls)==1 and calls[0][1]['revision']=='immutable-sha'
    assert meta['revision']=='fixed-commit'
    resolve_backbone(directory)
    assert len(calls)==1
