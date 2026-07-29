# Third-Party Notices

ScAI is licensed under the Apache License 2.0. This file records third-party
material that ScAI incorporates or derives content from, together with the
licence terms that material is distributed under.

Runtime dependencies installed from PyPI and npm are not listed here; their
licences travel with the packages and are recorded in `requirements.txt`,
`requirements-nodeps.txt` and `frontend/package.json`.

---

## bioSkills

- **Project**: bioSkills
- **Source**: https://github.com/GPTomics/bioSkills
- **Commit referenced**: `076137114dda666202a17d2f6b9a7215cc00c888`
- **Licence**: MIT
- **Copyright**: Copyright (c) 2026 GPTomics

### What ScAI derives from it

`backend/data/method_rules.yaml` is a corpus of methodological rules applied to
bioinformatics analyses. The statistical claims and the literature citations in
that file are distilled from bioSkills SKILL.md documents, specifically those
under `pathway-analysis/`, `single-cell/`, `differential-expression/` and
`experimental-design/`.

The derivation is not a copy. bioSkills targets an R/Bioconductor stack
(clusterProfiler, limma, edgeR, Seurat), whereas ScAI executes Python in a
Jupyter sandbox. Each rule was therefore re-expressed against the installed
Python packages: the `python` field of every rule is ScAI's own work and is not
upstream content. Rule identifiers, the gate/guidance classification, the
enforcement mechanisms in `backend/services/method_rules.py`, and the code gates
that cite these rules are likewise ScAI's own.

No bioSkills source file is vendored into this repository.

### Licence compatibility

The MIT licence is permissive and compatible with Apache License 2.0
redistribution. MIT requires that its copyright notice and permission notice be
retained in copies and substantial portions of the material; both are reproduced
in full below, and the derived file carries a provenance header pointing here.

Apache-2.0 and MIT differ in that Apache-2.0 additionally grants patent rights
and requires a NOTICE-style attribution. Distributing MIT-derived content within
an Apache-2.0 project is permitted; the MIT terms continue to govern the
MIT-derived portions, which is what this notice records.

### MIT License

```
MIT License

Copyright (c) 2026 GPTomics

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## External data resources

ScAI fetches gene sets and marker tables from external resources at runtime.
Nothing from these resources is redistributed with this repository; the caches
under `backend/data/` are gitignored. Users are responsible for observing each
resource's own terms, which differ -- some restrict commercial use.

| Resource | Used by | Fetched at runtime |
|---|---|---|
| MSigDB | `backend/services/gene_set_fetch.py` | Collection GMTs, per pinned release |
| Ensembl REST | `backend/services/gene_set_fetch.py` | Orthology and gene-symbol lookup |
| PanglaoDB | `scripts/fetch_markers.py` | Cell-type marker table |
| CellMarker 2.0 | `scripts/fetch_markers.py` | Cell-type marker table |
| OmniPath / CollecTRI | `decoupler` (runtime dependency) | Transcription-factor regulons |
