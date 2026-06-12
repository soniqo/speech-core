# fdb_mini

Synthetic 4-sample fixture mirroring the FDB v1.0 directory
layout — one sample per category — consumed by
`test_fdb_bench_smoke` so the FDB driver's smoke test has
something to point at without downloading the real 727-
sample corpus. Not real FDB data; do not benchmark against
this. Regenerate with:

    python3 scripts/make_fdb_mini.py
