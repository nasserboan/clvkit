# Benchmarks

Reproduce anything here with:

```console
uv run --extra dask python benchmarks/dask_crossover.py
```

## `engine="dask"`: where it starts to win

`CustomerBase.from_transactions` is the only step in clvkit whose cost scales
with *transactions* rather than with customers, which is why it is the only
step Dask is offered for. The question this benchmark answers is when that
offer is worth taking.

Synthetic logs in the documented input contract, read from Parquet by both
engines, summarised at `time_unit="W", collapse="D"`. Each cell is a fresh
subprocess, because peak RSS is a high-water mark that never comes back down
inside one. MacBook Pro, M-series, 10 cores; Dask's default scheduler, which is
threads in this same process.

Three transactions per customer, a CDNOW-shaped log where the summary barely
compresses:

| rows | pandas | Dask | |
|---:|---:|---:|---|
| 100,000 | 0.06 s · 192 MB | 0.22 s · 216 MB | pandas 3.7× faster |
| 1,000,000 | 0.35 s · 474 MB | 0.50 s · 556 MB | pandas 1.4× faster |
| 4,000,000 | 1.99 s · 1,349 MB | 1.68 s · 1,698 MB | **Dask 1.2× faster** |
| 16,000,000 | 17.40 s · 4,973 MB | 9.59 s · 5,589 MB | **Dask 1.8× faster** |

Fifty transactions per customer. The summary now compresses 50:1, which is the
case Dask should like most:

| rows | pandas | Dask | |
|---:|---:|---:|---|
| 1,000,000 | 0.17 s · 414 MB | 0.42 s · 531 MB | pandas 2.5× faster |
| 4,000,000 | 1.56 s · 1,213 MB | 1.32 s · 1,497 MB | **Dask 1.2× faster** |
| 16,000,000 | 16.06 s · 4,354 MB | 7.34 s · 4,757 MB | **Dask 2.2× faster** |

Both engines returned identical customer counts at every size, which is the
cross-check the script runs on itself.

## What the numbers say

**The crossover is wall-clock, and it is at about 4 million transactions.**
Below that, Dask's graph construction and shuffle cost more than the work they
distribute, and pandas wins by up to 3.7×. Above it Dask uses all the cores and
pulls ahead, reaching 1.8–2.2× by 16 million rows. Compressing harder moves the
speedup but not the crossover.

**There is no memory crossover, and that is the more useful result.** Dask's
peak RSS is *higher* than pandas' at every size measured, by 12–25%. The
default scheduler runs the workers as threads in this same process, so the
shuffled partitions, the finished summary, and Dask's own machinery all sit in
one address space, and nothing is ever handed off anywhere it could be freed.
The raw log never being held whole doesn't help when the shuffle buffers
replace it.

So the honest recommendation is narrower than the feature sounds:

- **Under ~4M transactions, use `engine="pandas"`.** That's the default, and
  it's faster.
- **Over ~4M, `engine="dask"` halves the wall-clock** if the log is already on
  disk in a format Dask can read in parallel (Parquet, partitioned CSV).
- **If the problem is that pandas runs out of memory, this doesn't fix it on
  one machine.** It would take a distributed scheduler with workers in separate
  processes, which this benchmark doesn't measure and this library doesn't
  configure for you. `dask.distributed` in front of the same
  `engine="dask"` call is where that story continues.

## Rerunning

```console
# the two tables above
uv run --extra dask python benchmarks/dask_crossover.py --sizes 100_000 1_000_000 4_000_000 16_000_000
uv run --extra dask python benchmarks/dask_crossover.py --per-customer 50 --sizes 1_000_000 4_000_000 16_000_000
```

Absolute times are machine-specific; the crossover point is the part worth
comparing.
