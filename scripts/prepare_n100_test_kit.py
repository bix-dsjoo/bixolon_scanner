"""Write the PowerShell-free 0.2.1 USB launchers and freeze its copied files."""

import argparse
from pathlib import Path

from bixolon_scanner.contracts.artifact import directory_content_manifest
from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.training.three_bakery_data import read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.kit_root.resolve()
    source = Path("artifacts/n100/0.2.1/worker-payload")
    if directory_content_manifest(root / "Benchmark/worker") != directory_content_manifest(
        source / "worker"
    ):
        raise ValueError("Copied Worker differs from verified 0.2.1")
    if directory_content_manifest(root / "Benchmark/licenses") != directory_content_manifest(
        source / "licenses"
    ):
        raise ValueError("Copied licenses differ")
    installer = "BixolonBakeryAIScannerLite-0.2.1-N100-Setup.exe"
    if sha256_file(root / installer) != sha256_file(source.parent / installer):
        raise ValueError("Copied installer differs")
    if sha256_file(root / "N100-MEASURE.exe") != sha256_file(
        source.parent / "test-kit-build/N100-MEASURE.exe"
    ):
        raise ValueError("Copied measurement runner differs")
    for filename in ("provenance.json", "version.json"):
        if sha256_file(root / "Benchmark" / filename) != sha256_file(source / filename):
            raise ValueError("Copied execution profile differs")
    common = '@echo off\nsetlocal\n"%~dp0N100-MEASURE.exe" {args}\nset "RESULT=%ERRORLEVEL%"\necho.\npause\nexit /b %RESULT%\n'
    files = {
        "1_INSTALL.cmd": f'@echo off\nstart "" /wait "%~dp0{installer}"\n',
        "2_MEASURE.cmd": common.replace("{args}", "%*"),
        "3_VERIFY_FILES.cmd": common.replace("{args}", "--verify-only %*"),
    }
    for name, text in files.items():
        (root / name).write_bytes(text.replace("\n", "\r\n").encode("ascii"))
    (root / "README_KO.txt").write_text(
        "N100 0.2.1 설치·측정 키트\n\n"
        "PowerShell과 Python을 따로 설치할 필요가 없습니다. 0.2.1만 측정합니다.\n\n"
        "1. 이 N100 폴더를 N100 내부 SSD의 C:\\N100으로 복사하세요.\n"
        "   previous-0.2.0 폴더는 이전 키트 보관용이므로 복사하지 않아도 됩니다.\n"
        "2. 앱을 설치하려면 1_INSTALL.cmd를 더블클릭하세요.\n"
        "   설치된 앱 사용과 별개로, 2_MEASURE.cmd는 포함된 측정용 Worker를 직접 실행합니다.\n"
        "3. 측정 전 Scanner Lite와 다른 Scanner Worker·무거운 작업을 종료하세요.\n"
        "4. 2_MEASURE.cmd를 더블클릭하세요. PowerShell 대신 포함된 N100-MEASURE.exe를 실행합니다.\n"
        "   파일 검증 → Worker 준비 대기 → warmup 10회 → 132장 3회(396개 응답) → 결과 ZIP 순서입니다.\n"
        "   GPU 최초 준비에는 몇 분이 걸릴 수 있습니다. 완료될 때까지 창을 닫지 마세요.\n"
        "5. results 폴더에 새로 만들어진 N100-날짜-시간-식별자.zip을 이 PC에 가져오세요.\n"
        "   실패해도 ZIP을 보존합니다. FAILED.txt와 Worker 로그로 원인을 확인할 수 있습니다.\n\n"
        "3_VERIFY_FILES.cmd는 측정 없이 전체 파일 checksum만 확인합니다.\n"
        "측정 결과에는 회차별 p50/p95/p99, 상태 개수, 전체 응답, CPU·드라이버 정보, 실제 provider가 들어갑니다.\n"
        "정답 승인·오승인·미검출은 결과를 고정 GT와 대조해 평가합니다. APPROVED 수를 정확도로 간주하지 않습니다.\n"
        "CPU 검출 → Intel UHD GPU 분류 → CPU verifier 및 CPU fallback을 유지했습니다.\n"
        "GPU를 쓸 수 없으면 실제 provider가 CPU로 기록됩니다. CPU fallback 시간을 GPU 성능으로 표시하지 않습니다.\n"
        "이 키트는 실제 N100 측정을 위한 도구이며, 빌드 PC 확인을 N100 실측으로 간주하지 않습니다.\n"
        "Checksum은 손상·변경 탐지이며 발행자 인증은 아닙니다.\n",
        encoding="utf-8-sig",
    )
    (root / "CURRENT-RESULTS-KO.md").write_text(
        "# 0.2.1 현재 검증 상태\n\n"
        "개발 로그132장: 정답 승인1,078/1,096 · 오승인0 · 미검출0 · UNKNOWN15 · 재촬영19 · 추가 검출16.\n\n"
        "개발 PC의 배포 EXE HTTP p50/p95/p99(ms): CPU201.56/253.67/291.13, CUDA58.47/74.25/84.99.\n"
        "이 값은 Core Ultra 9 285K / RTX5080 결과이며 N100 성능이 아닙니다.\n\n"
        "N100 실측은 아직 완료하지 않았습니다. 이 키트의 2_MEASURE.cmd로0.2.1만 측정합니다.\n"
        "실제 provider·회차별 시간·상태·전체 응답은 results ZIP에 보존합니다.\n",
        encoding="utf-8",
    )
    by_hash = {sha256_file(p): p for p in (root / "log132").iterdir() if p.is_file()}
    records = read_jsonl(
        Path("artifacts/retraining/three-bakery-improvement-0.1.18/log-inputs.jsonl")
    )
    if len(by_hash) != 132 or set(by_hash) != {r["image_sha256"] for r in records}:
        raise ValueError("USB images differ from frozen log132")
    inputs = [
        {
            "image_id": r["image_id"],
            "image_sha256": r["image_sha256"],
            "path": by_hash[r["image_sha256"]].relative_to(root).as_posix(),
        }
        for r in records
    ]
    entries = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or relative.parts[0] in {
            "previous-0.2.0",
            "results",
            "KIT-MANIFEST.json",
        }:
            continue
        entries.append(
            {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    write_json(
        root / "KIT-MANIFEST.json",
        {
            "version": "0.2.1",
            "benchmark": "Benchmark",
            "powershell_required": False,
            "image_count": 132,
            "warmup": 10,
            "repetitions": 3,
            "files": entries,
            "inputs": inputs,
        },
    )
    print(f"Kit frozen: {root}; {len(entries)} files; 132 images; 0.2.1 only", flush=True)


if __name__ == "__main__":
    main()
