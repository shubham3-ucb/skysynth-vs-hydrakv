# SkySynth vs HydraKV

Workload: YCSB-A 50% read / 50% blind upsert, Zipf 0.99, 250M keys x 100 B values, 16 GiB memory budget, 16 threads,
30 s timed run. Both stores ran under the same conditions: the same benchmark on the same dedicated machine type (AWS
m6id.16xlarge) and the same hidden test suite (16 tests). `kvstore_impl.cc` here is SkySynth's store. HydraKV:
[ksenxx/hydrakv](https://github.com/ksenxx/hydrakv) at commit 934b8499c559, unmodified (it was tuned for Zipf 0.95 at 8 GiB).

| | SkySynth | HydraKV |
|---|---|---|
| Hidden tests passed | **16/16** | 12/16 (13/16 on a re-run) |
| Hidden tests failed | none | crash during writes, crash during read-modify-write, repeated crash and recovery, power loss |
| Throughput, benchmark, median of 3 (Mops/s) | **22.2** | 13.3 |
| Throughput, hidden replays on random 64-bit keys: exact / Twitter ops (Mops/s) | 11.5 / 12.7 | **11.8 / 16.9** |
| Disk-space check | pass (141 MiB for 47 MiB live) | pass (2,049 MiB for 47 MiB live) |
| Memory in the benchmark (budget 16 GiB) | 15.9 GiB | 15.3 GiB |
