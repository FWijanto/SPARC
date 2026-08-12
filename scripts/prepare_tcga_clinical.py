#!/usr/bin/env python3
"""Build the SPARC DSS table from the public TCGA-CDR supplement."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


SHEET_NAME = "TCGA-CDR"
EXPECTED_PATIENTS = 10_445
EXPECTED_EVENTS = 2_481
EXPECTED_SHA256 = "c520a9ee0aa3c8a4332d4b92a15442cef01ed91480b33def02558557cebb1e6d"


def prepare_dss_table(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Read TCGA-CDR Table S1 and write the clinical table used by SPARC."""
    source = pd.read_excel(input_path, sheet_name=SHEET_NAME)
    required = {"bcr_patient_barcode", "type", "DSS.time", "DSS"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"TCGA-CDR sheet is missing columns: {sorted(missing)}")

    clinical = pd.DataFrame(
        {
            "patient_id": source["bcr_patient_barcode"],
            "project_id": "TCGA-" + source["type"],
            "time": source["DSS.time"],
            "event": source["DSS"],
        }
    )
    clinical = clinical.dropna(subset=["time", "event"])
    clinical = clinical.loc[clinical["time"] > 0].copy()
    clinical["event"] = clinical["event"].astype(int)

    n_patients = len(clinical)
    n_events = int(clinical["event"].sum())
    if (n_patients, n_events) != (EXPECTED_PATIENTS, EXPECTED_EVENTS):
        raise RuntimeError(
            "Unexpected TCGA-CDR contents: "
            f"found {n_patients} patients and {n_events} events; expected "
            f"{EXPECTED_PATIENTS} and {EXPECTED_EVENTS}. Verify that the input "
            "is the official Liu et al. TCGA-CDR Supplemental Table S1."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clinical.to_csv(output_path, index=False)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()

    print(f"Wrote {n_patients:,} patients ({n_events:,} DSS events) to {output_path}")
    print(f"SHA256: {digest}")
    if digest != EXPECTED_SHA256:
        print(
            "Note: row/event counts match the paper input, but byte-level output "
            "differs. This can reflect CSV formatting across pandas versions."
        )
    return clinical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare data/clinical_dss.csv from TCGA-CDR Table S1."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to TCGA-CDR-SupplementalTableS1.xlsx",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/clinical_dss.csv"),
        help="Output CSV (default: data/clinical_dss.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_dss_table(args.input, args.output)


if __name__ == "__main__":
    main()
