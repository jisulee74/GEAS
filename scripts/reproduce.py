#!/usr/bin/env python3
"""Verify a source clone, or restore approved private data without overwriting files."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'reproducibility/project.json'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_archive(archive, checksum=None):
    archive = Path(archive).resolve()
    checksum = Path(checksum) if checksum else Path(str(archive) + '.sha256')
    expected = checksum.read_text().split()[0]
    if len(expected) != 64 or sha256(archive) != expected:
        raise ValueError('Backup SHA-256 mismatch')
    return archive


def evidence():
    for entry in json.loads((ROOT / 'reproducibility/evidence_manifest.json').read_text()):
        if sha256(ROOT / entry['published_file']) != entry['published_sha256']:
            raise ValueError('Published evidence changed: ' + entry['published_file'])


def destination_for(member_name, destination, mappings):
    path = PurePosixPath(member_name)
    if path.is_absolute() or '..' in path.parts or '.git' in path.parts:
        raise ValueError('Unsafe backup member path')
    name = path.as_posix()
    for mapping in mappings:
        prefix = mapping['member_prefix']
        if name.startswith(prefix):
            target = destination / mapping['restore_prefix'] / name[len(prefix):]
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError('Restore path escapes destination')
            return target
    return None


def restore(archive, checksum, destination, mappings, overlays=()):
    archives = [verify_archive(archive, checksum), *(verify_archive(p) for p in overlays)]
    superseded = [set() for _ in archives]
    for i in range(len(archives) - 1):
        for later in archives[i + 1:]:
            with tarfile.open(later, "r:*") as layer:
                superseded[i].update(m.name for m in layer)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    copied = skipped = 0
    for index, archive in enumerate(archives):
        with tarfile.open(archive, 'r:*') as tar:
            for member in tar:
                if member.name in superseded[index] or Path(member.name).name == "running.json" or Path(member.name).suffix in {".pid", ".lock", ".tmp"}:
                    continue
                target = destination_for(member.name, destination, mappings)
                if target is None or member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError('Refusing non-regular restore member: ' + member.name)
                source = tar.extractfile(member)
                if source is None:
                    raise ValueError('Unreadable backup member: ' + member.name)
                if target.exists():
                    h = hashlib.sha256()
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                        h.update(block)
                    if not target.is_file() or sha256(target) != h.hexdigest():
                        raise FileExistsError('Existing file differs; choose a fresh clone: ' + str(target))
                    skipped += 1
                    continue
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                with target.open('xb') as output:
                    target.chmod(0o600)
                    shutil.copyfileobj(source, output, 8 * 1024 * 1024)
                copied += 1
    print(json.dumps({'restored_files': copied, 'identical_existing_files': skipped}))


def smoke(config):
    evidence()
    env = dict(os.environ)
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    env['PYTHONPATH'] = os.pathsep.join(str(ROOT / p) for p in config['python_paths'])
    result = subprocess.run([sys.executable, '-m', 'pytest', *config['smoke_tests'], '-q', '--tb=short'], cwd=ROOT / config['working_directory'], env=env)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    actions.add_parser('smoke', help='Run synthetic CPU checks; no DB, real training, control or publishing')
    verify = actions.add_parser('verify-backup')
    verify.add_argument('archive', type=Path)
    verify.add_argument('--checksum', type=Path)
    restore_parser = actions.add_parser('restore', help='Restore only approved private-data paths; never overwrite differing files')
    restore_parser.add_argument('archive', type=Path)
    restore_parser.add_argument('--checksum', type=Path)
    restore_parser.add_argument('--overlay', type=Path, action='append', default=[], help='Newer delta backup; takes precedence without overwriting existing files')
    restore_parser.add_argument('--destination', type=Path, default=ROOT)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    if args.action == 'smoke':
        return smoke(config)
    if args.action == 'verify-backup':
        print('SHA-256 verified:', verify_archive(args.archive, args.checksum).name)
    else:
        restore(args.archive, args.checksum, args.destination, config['restore_mappings'], args.overlay)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
