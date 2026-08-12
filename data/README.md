# Data layout

This directory ships the exact cross-validation split used for the paper.
The clinical outcome table is reconstructed locally from the public TCGA
Clinical Data Resource (TCGA-CDR); patient-level outcome rows are not
redistributed in this repository. Patch-level features are also not committed.

## Cross-validation split

`mmp_hybrid_splits_v2_20cancer.csv` contains 44,175 fold-specific slide rows
covering 7,284 patients, 19 TCGA disease cohorts, and the 18 reporting groups
used in the paper. The legacy filename is retained for checkpoint and config
compatibility. Its columns are:

| Column | Description |
|---|---|
| `slide_id` | TCGA diagnostic-slide identifier |
| `patient_id` | TCGA participant identifier |
| `cancer_type` | TCGA project identifier, for example `TCGA-BRCA` |
| `cancer_group` | Lowercase grouping label used by the original split |
| `fold` | Outer fold, from 0 through 4 |
| `split` | `train` or `val`; the legacy `val` partition is the held-out outer test set |

Reference checksum:

```text
SHA256  cc231ad6c9e5f5e3fc88c54e45c0db896c86dec70813a081cf335a8eb6890343
```

## Prepare disease-specific survival labels

1. Download `TCGA-CDR-SupplementalTableS1.xlsx` from the
   [NCI Genomic Data Commons TCGA-CDR publication page](https://gdc.cancer.gov/about-data/publications/PanCan-Clinical-2018).
2. Generate the local table:

```bash
python scripts/prepare_tcga_clinical.py \
    --input /path/to/TCGA-CDR-SupplementalTableS1.xlsx
```

This writes the config-compatible `data/clinical_dss.csv` with columns
`patient_id`, `project_id`, `time`, and `event`. From the official supplement,
the output contains 10,445 patients and 2,481 DSS events. It remains ignored by
Git and can be checked against:

```text
SHA256  c520a9ee0aa3c8a4332d4b92a15442cef01ed91480b33def02558557cebb1e6d
```

## Patch-level features

The configs reference two feature directories, neither of which is committed:

| Path | Content |
|---|---|
| `features/hoptimus1` | Per-slide HDF5 files of patch features extracted with the H-optimus-1 foundation model at 20× / 224 px. One `.h5` per slide containing `features` (`N×1536`) and `coords` (`N×2`). |
| `predicted_programs_transformers` | Per-slide HDF5 files of the 40 SPARC-Map gene-program scores per patch. One `.h5` per slide containing `features` (`N×40`) and `coords` (`N×2`). |

To run training or inference, point either:

* the YAML config (`data.img_feature_dir`, `data.gep_feature_dir`), or
* the environment variables `SPARC_IMG_FEATURE_DIR` and `SPARC_GEP_FEATURE_DIR`,

at the directories holding these files on your machine.

### How to obtain the features

* **Image features** are produced by running the H-optimus-1 backbone on
  trident-detected tissue patches. See `inference/feature_extraction.md`.
* **Gene-program scores** are produced by SPARC-Map (this repo) - run
  inference with the SPARC-Map architecture on the same tissue patches
  to obtain a 40-dim score per patch.

## External cohorts

External cohorts (NLST, SurGen, Yale Breast, Ovarian) follow the same
two-directory convention. Cohort-specific layouts and DICOM/CZI handling
are described in `inference/cohorts.py`.
