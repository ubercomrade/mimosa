"""Benchmark native and generic input relabeling without changing model scores."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
from mimosa import (
    EncodedSequences, InputRelabeledModel, alphabet_permutations,
    prepare_profile, read_fasta, read_model, relabel_model,
)
from mimosa.cache import Cache
from mimosa.io.bundles import model_fingerprint, sequence_fingerprint
from mimosa.scan import _scan_batch_into


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(function):
    gc.collect()
    start = time.perf_counter()
    value = function()
    elapsed = time.perf_counter()-start
    del value
    return elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+")
    parser.add_argument("--orders", default="0,4", help="comma-separated BaMM orders")
    parser.add_argument("--fasta")
    parser.add_argument("--num-sequences", type=int, default=10_000)
    parser.add_argument("--seq-length", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=112)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"choose a new benchmark output: {args.output}")
    if args.repeats < 1 or args.num_sequences < 1 or args.seq_length < 1:
        parser.error("repeats and sequence dimensions must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(111)
    batch = (
        read_fasta(args.fasta)[0] if args.fasta else
        EncodedSequences(rng.integers(0,4,args.num_sequences*args.seq_length,dtype=np.uint8),
                         np.arange(args.num_sequences+1,dtype=np.int64)*args.seq_length)
    )
    warmup = EncodedSequences.from_strings(["ACGT"*50, "TGCA"*50])
    orders = [int(x) for x in args.orders.split(",")]
    records, source_models, equalities = [], [], []
    methods = ["native_original","native_relabel","generic_wrapper"]
    start_all = time.perf_counter()
    for path in args.models:
        for order in (orders if Path(path).suffix==".ihbcp" else [None]):
            model = read_model(path, order=order)
            permutations = alphabet_permutations(model.motif_length,args.seed)
            native = relabel_model(model,permutations=permutations)
            wrapped = InputRelabeledModel(model,permutations)
            variants = dict(zip(methods,(model,native,wrapped)))
            for variant in variants.values():
                _scan_batch_into(variant,warmup,"both",True)
                prepare_profile(variant,warmup,min_logerr=1,circular=True)
            # Validate the two implementations on the actual full scoring DNA.
            a,b = [_scan_batch_into(v,batch,"both",True) for v in (native,wrapped)]
            for x,y in ((a.forward,b.forward),(a.reverse,b.reverse)):
                np.testing.assert_array_equal(x.data,y.data)
            del a,b
            source_models.append({"path":str(Path(path).resolve()),"sha256":sha(path),
                                  "name":model.name,"order":order,"length":model.motif_length,
                                  "fingerprint":model_fingerprint(model)})
            equalities.append({"model":model.name,"order":order,"raw_scores_checked":2*len(batch.data),"exact":True})
            cache = Cache(args.output.parent/"profile_cache")
            prepare_profile(native,batch,min_logerr=3,circular=True,cache=cache)
            # Ensure cache verification and first mmap opening are outside warm timings.
            prepare_profile(native,batch,min_logerr=3,circular=True,cache=cache)
            for repeat in range(args.repeats):
                rotated = methods[repeat%3:]+methods[:repeat%3]
                for method in rotated:
                    variant = variants[method]
                    for stage,function in [
                        ("scan_both",lambda v=variant:_scan_batch_into(v,batch,"both",True)),
                        ("prepare_no_cache",lambda v=variant:prepare_profile(v,batch,min_logerr=3,circular=True)),
                    ]:
                        seconds = measure(function)
                        records.append({"model":model.name,"order":order,"length":model.motif_length,
                                        "method":method,"stage":stage,"repeat":repeat,"seconds":seconds})
                for method,function,stage in [
                    ("native_relabel",lambda:relabel_model(model,permutations=permutations),"construct"),
                    ("generic_wrapper",lambda:InputRelabeledModel(model,permutations),"construct"),
                    ("native_relabel",lambda:prepare_profile(native,batch,min_logerr=3,circular=True,cache=cache),"warm_cache"),
                ]:
                    records.append({"model":model.name,"order":order,"length":model.motif_length,
                                    "method":method,"stage":stage,"repeat":repeat,"seconds":measure(function)})
            print(f"{model.name} order={order}: completed {args.repeats} repeats",file=sys.stderr,flush=True)
    implementation = Path(__file__).resolve().parents[1]/"src/mimosa"
    result = {
        "records":records,"source_models":source_models,"equivalence_checks":equalities,
        "positions_per_strand":len(batch.data),"n_rows":len(batch),"repeats":args.repeats,
        "seed":args.seed,"sequence_fingerprint":sequence_fingerprint(batch),
        "elapsed_seconds":time.perf_counter()-start_all,
        "runtime":{"python":sys.version,"numpy":np.__version__,"platform":platform.platform(),
                   "openblas_threads":os.environ.get("OPENBLAS_NUM_THREADS"),
                   "load_average":list(os.getloadavg()),"max_rss_KiB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
        "source_sha256":{str(p):sha(p) for p in [Path(__file__).resolve(),implementation/"randomization.py",implementation/"scan.py"]},
        "limitations":"Warm JIT, one process; OS cache is not flushed; no CNN/GPU timing or biological replication."
    }
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"output":str(args.output),"timings":len(records),"elapsed_seconds":result["elapsed_seconds"]}))


if __name__=="__main__":
    main()
