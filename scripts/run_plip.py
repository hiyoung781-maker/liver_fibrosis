"""PLIP 실행과 XML 리포트 파싱 (spec Sec 8.0).

PLIP은 GPU 구현이 없는 CPU 도구다. K-BDS에서는 cpu64 파티션에서 포즈 단위로
병렬 실행한다. 8gpu는 GPU 사용 여부와 무관하게 시간당 8 node-hour를 과금하므로
이 배치를 8gpu에서 돌리는 것은 비용 낭비다.

상호작용이 하나도 없는 포즈는 정상적인 결과다. 예외가 아니라 빈 리스트로
돌려줘야 배치가 멈추지 않는다.

실제 8W30 산출물로 검증한 바로는, PLIP은 다섯 글자 CCD 코드(A1AFA)를 세 글자
hetid(A1A)로 자르고, 인접한 칼슘과 리간드를 합쳐 사이트를 구성하므로 longname은
"A1A-CA"가 된다. 또한 13개 결합 부위 중 대부분은 칼슘/글리칸 등 리간드가 아닌
부위이므로, 리포트를 파싱할 때는 하나의 사이트를 선택할 방법이 필요하다.
"""

from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

__all__ = ["INTERACTION_KINDS", "parse_report", "run_plip"]

INTERACTION_KINDS = ("metal_complexes", "hbonds", "hydrophobic_contacts",
                     "pi_stacking", "salt_bridges", "water_bridges",
                     "halogen_bonds")

# PLIP XML의 컨테이너/항목 태그와 우리 키의 매핑.
_TAGS = {
    "metal_complexes": ("metal_complexes", "metal_complex"),
    "hbonds": ("hydrogen_bonds", "hydrogen_bond"),
    "hydrophobic_contacts": ("hydrophobic_interactions", "hydrophobic_interaction"),
    "pi_stacking": ("pi_stacks", "pi_stack"),
    "salt_bridges": ("salt_bridges", "salt_bridge"),
    "water_bridges": ("water_bridges", "water_bridge"),
    "halogen_bonds": ("halogen_bonds", "halogen_bond"),
}

_INT_FIELDS = {"resnr"}
_FLOAT_FIELDS = {"dist", "dist_h-a", "dist_d-a", "don_angle", "centdist",
                 "angle", "offset", "dist_a-w", "dist_d-w"}


def _coerce(tag: str, text: str):
    if tag in _INT_FIELDS:
        return int(text)
    if tag in _FLOAT_FIELDS:
        return float(text)
    return text


def _record_from_site(site) -> dict:
    record = {kind: [] for kind in INTERACTION_KINDS}
    interactions = site.find("interactions")
    if interactions is None:
        return record
    for kind, (container, item) in _TAGS.items():
        node = interactions.find(container)
        if node is None:
            continue
        for entry in node.findall(item):
            record[kind].append({
                child.tag: _coerce(child.tag, (child.text or "").strip())
                for child in entry if child.text
            })
    return record


def _merge(records: list) -> dict:
    merged = {kind: [] for kind in INTERACTION_KINDS}
    for record in records:
        for kind in INTERACTION_KINDS:
            merged[kind].extend(record[kind])
    return merged


def parse_report(xml_text: str, hetid: Optional[str] = None) -> dict:
    """Parse a PLIP XML report into per-kind interaction lists.

    A real PLIP report can contain many <bindingsite> elements (13 for the
    deposited 8W30 structure), most of them calcium or glycan sites that are
    not the ligand of interest. Without `hetid`, every site is merged, which
    is the default batch behaviour. Pass `hetid` (PLIP's truncated,
    three-character hetid -- not the full CCD code) to isolate the single
    site whose <identifiers><hetid> matches, so an unrelated calcium site's
    metal complex is not mistaken for the ligand's own coordination shell.

    Returns empty lists for every kind on a pose with no detected
    interactions -- that is a normal result, not an error, and must not
    raise so a single such pose cannot stop a batch of ~85,000.
    """
    root = ET.fromstring(xml_text)
    sites = root.findall(".//bindingsite")

    if hetid is not None:
        sites = [
            site for site in sites
            if (site.findtext("identifiers/hetid") or "").strip() == hetid
        ]

    return _merge(_record_from_site(site) for site in sites)


def run_plip(complex_pdb: str, out_dir: str, plip_bin: str = "plip",
             timeout: int = 300) -> Optional[str]:
    """Run PLIP on a receptor-ligand complex PDB and return the XML report path.

    Returns None (never raises) on any failure -- PLIP binary missing,
    non-zero exit, or timeout -- so a batch of ~85,000 poses can skip one
    bad pose instead of dying on it. `plip_bin` should be the absolute path
    to the PLIP executable in the docking conda env, since PLIP is not on
    the default PATH on this system.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [plip_bin, "-f", str(complex_pdb), "-o", str(out), "-x", "-q"],
            check=True, capture_output=True, timeout=timeout,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None
    # PLIP names its XML report "<input-stem>_report.xml", not "report.xml"
    # -- verified against real output on 8W30.pdb (8W30_report.xml).
    candidates = sorted(out.glob("*_report.xml"))
    return str(candidates[0]) if candidates else None
