"""Print the structured data audit and re-check local public files.
    python scripts/audit_data.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentmamba.config import PROJECT_ROOT
from agentmamba.data.document_loader import load_uci_198, load_uci_851
from agentmamba.data.source_audit import AUDIT_DATE, SOURCES


def main() -> None:
    raw = PROJECT_ROOT / "data" / "raw"
    local = {}
    p851 = raw / "uci_851_steel_energy" / "Steel_industry_data.csv"
    if p851.exists():
        evs = load_uci_851(p851)
        local["UCI #851"] = {"daily_events": len(evs), "first": evs[0].timestamp, "last": evs[-1].timestamp}
    p198 = raw / "uci_198_steel_plates" / "Faults.NNA"
    if p198.exists():
        local["UCI #198"] = {"records": len(load_uci_198(p198, p198.with_name("Faults27x7_var")))}
    print(json.dumps({"audit_date": AUDIT_DATE, "sources": SOURCES, "local_files": local}, indent=2))


if __name__ == "__main__":
    main()
