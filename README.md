# SkySynth vs HydraKV

Workload: YCSB-A 50% read / 50% blind upsert, Zipf 0.99, 250M keys x 100 B values, 16 GiB memory budget, 16 threads,
30 s timed run. Both stores ran under the same conditions: the same benchmark on the same dedicated machine type (AWS
m6id.16xlarge) and the same hidden test suite (16 tests). `kvstore_impl.cc` here is SkySynth's store. HydraKV:
[ksenxx/hydrakv](https://github.com/ksenxx/hydrakv) at commit 934b8499c559, unmodified (it was tuned for Zipf 0.95 at 8 GiB).

| | SkySynth | HydraKV |
|---|---|---|
| Hidden tests passed | **16/16** and **16/16** (two gradings; each crash test run 3 times per grading, all passed) | 12/16 and 13/16 (two gradings) |
| Throughput, benchmark (Mops/s) | **22.2** (runs: 22.19, 22.23) | 13.3 (runs: 13.21, 13.28, 13.32) |
| Throughput, hidden replays on random 64-bit keys: exact / YCSB-A trace / Twitter ops (Mops/s) | 11.5 / 11.6 / 12.7 | **11.8 / 22.3 / 16.9** |
| Disk-space check | pass (141 MiB peak for 47 MiB live) | pass (2,049 MiB peak for 47 MiB live) |
| Peak memory in the benchmark (budget 16 GiB) | 15.9 GiB | 15.3 GiB |

## HydraKV's failed tests

Verbatim from the hidden-suite logs. SkySynth passes all four in both gradings, 3 of 3 runs each.

| Test | What it checks | HydraKV (grading 1 / grading 2) | Log message |
|---|---|---|---|
| Crash during writes | kill -9 while 8 threads write and another thread calls Checkpoint() every 0.1-0.4 s; after reopen, each thread's surviving writes must be a prefix that includes everything a completed Checkpoint() covered | fail (1 of 3 runs) / pass | "lost 2394 ops that a completed Checkpoint() had covered" |
| Crash during read-modify-write | Same, with counters | fail (1 of 3 runs) / fail (1 of 2 runs) | "lost 3394 ops that a completed Checkpoint() had covered" |
| Repeated crash and recovery | Write, Checkpoint(), kill -9, reopen, repeated | fail / fail | "no prefix of its 16163 ops since the last recovery (9269 covered by a completed Checkpoint()) matches the recovered state" |
| Power loss | Same, and data the OS had not written to disk is dropped | fail / fail | "no prefix of its 14895 ops since the last recovery (7072 covered by a completed Checkpoint()) matches the recovered state, but a shorter prefix does" |

Cause, from HydraKV's code and its own audit ("Bug 6"): Checkpoint() writes only the calling thread's write buffer, so
new keys still in other threads' buffers are not made durable, and writes a completed Checkpoint() covered are lost
after a crash.
