"""Run the actual benchmark CLI on reproducible random local CPU checkpoints.

No trained accuracy or large-model/GPU performance claim is implied. The
temporary weights are deleted after the report is written.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/tiny-cpu-benchmark.json"))
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    torch.set_num_threads(1)
    fixture = {
        "target": {"seed": 101, "layers": 2, "hidden": 32},
        "draft": {"seed": 202, "layers": 1, "hidden": 16},
        "vocabulary": 64,
        "context": 128,
        "heads": 2,
    }
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        for name in ("target", "draft"):
            destination = temporary / name
            destination.mkdir()
            vocabulary = {"<bos>": 0, "<eos>": 1, "<unk>": 2, **{f"t{index}": index + 3 for index in range(61)}}
            raw = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
            raw.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=raw, bos_token="<bos>", eos_token="<eos>", unk_token="<unk>"
            )
            tokenizer.save_pretrained(destination)
            torch.manual_seed(fixture[name]["seed"])
            model = GPT2LMHeadModel(
                GPT2Config(
                    vocab_size=64,
                    n_positions=128,
                    n_ctx=128,
                    n_embd=fixture[name]["hidden"],
                    n_layer=fixture[name]["layers"],
                    n_head=2,
                    bos_token_id=0,
                    eos_token_id=1,
                )
            )
            model.save_pretrained(destination)
        report_path = temporary / "report.json"
        command = [
            sys.executable,
            "-m",
            "speculative_decoding",
            "hf-benchmark",
            "--target",
            str(temporary / "target"),
            "--draft",
            str(temporary / "draft"),
            "--prompt",
            "t0 t1 t2",
            "--max-new-tokens",
            "24",
            "--draft-tokens",
            "4",
            "--warmups",
            str(args.warmups),
            "--repeats",
            str(args.repeats),
            "--threads",
            "1",
            "--output",
            str(report_path),
        ]
        subprocess.run(
            command, cwd=root, env=dict(os.environ, PYTHONPATH=str(root / "src")), check=True, stdout=subprocess.DEVNULL
        )
        report = json.loads(report_path.read_text())
        report["target_checkpoint"] = "temporary random target fixture; weights removed after run"
        report["draft_checkpoint"] = "temporary random draft fixture; weights removed after run"
        report["fixture"] = fixture
        report["trained_model_quality_evaluated"] = False
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "modes": {
                    name: {
                        key: value[key]
                        for key in ("median_seconds", "observed_median_time_ratio_vs_serial", "same_tokens_as_serial")
                    }
                    for name, value in report["modes"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
