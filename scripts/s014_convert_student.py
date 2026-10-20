"""Convert the trained S014 student to Core ML with a flexible token length.

    checkpoint -> Student (CPU, eval) -> wrapper with named outputs -> trace at T=32
    -> ct.convert(input_ids [1, RangeDim(1, 512)] int32, style [1,256] fp32; fp16, iOS18)
    -> artifacts/s014/student_coreml/kokoro_student_w256.mlpackage
    -> xcrun coremlcompiler compile -> exact logical bytes of the .mlmodelc
    -> artifacts/s014/student_coreml/size.json
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from logical_size import logical_bytes  # noqa: E402
from s014_student import Student, parameter_count  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/s014/student_coreml"


class Wrapped(nn.Module):
    def __init__(self, student: Student):
        super().__init__()
        self.student = student

    def forward(self, input_ids, style):
        hidden, logits, text = self.student(input_ids, style, None)
        return hidden, logits, text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/s014/runs/w256-b/step10000.pt")
    args = parser.parse_args()

    saved = torch.load(args.checkpoint, map_location="cpu")
    student = Student(**saved["config"])
    student.load_state_dict(saved["model"])
    student = student.eval()
    params = parameter_count(student)
    wrapped = Wrapped(student).eval()

    ids = torch.randint(1, 178, (1, 32), dtype=torch.int32)
    style = torch.zeros((1, 256), dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, (ids, style), strict=True)
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(1, 512)), dtype=np.int32),
            ct.TensorType(name="style", shape=(1, 256), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="duration_hidden"),
            ct.TensorType(name="duration_logits"),
            ct.TensorType(name="text_features"),
        ],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    package = OUT / "kokoro_student_w256.mlpackage"
    compiled_dir = OUT / "compiled"
    for p in (package, compiled_dir):
        if p.exists():
            shutil.rmtree(p)
    OUT.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(package))
    compiled_dir.mkdir()
    subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(package), str(compiled_dir)],
        check=True, stdout=subprocess.DEVNULL,
    )
    compiled = next(compiled_dir.glob("*.mlmodelc"))
    total = logical_bytes(compiled)
    weights = sum(p.stat().st_size for p in compiled.rglob("weight.bin"))
    result = dict(
        checkpoint=str(args.checkpoint.relative_to(ROOT)),
        step=saved.get("step"),
        config=student.config,
        parameters=params,
        compiled_bytes=total,
        compiled_weight_bin=weights,
        compiled_other=total - weights,
        compiled=str(compiled.relative_to(ROOT)),
        package=str(package.relative_to(ROOT)),
    )
    (OUT / "size.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
