"""Write a Korean handoff report and exact GT/prediction failure overlays."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from bixolon_scanner.configuration import load_json_config
from bixolon_scanner.contracts.artifact import directory_content_manifest
from bixolon_scanner.contracts.catalog import sha256_file
from bixolon_scanner.training.three_bakery_data import read_jsonl, write_json


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    version = root / "artifacts/versions/0.1.17"
    final = version / "final-evaluation"
    output = root / "artifacts/distributions/0.1.17"
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    result = load_json_config(final / "report.json")
    rows = read_jsonl(final / "cpu/responses.jsonl")
    records = {r["image_id"]: r for r in read_jsonl(final / "final-inputs.jsonl")}
    names = {
        p["class_id"]: p["display_name_ko"]
        for p in load_json_config(root / "apps/product_scanner/assets/catalog/bread_ko.json")[
            "products"
        ]
    }
    font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 24)
    small = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 19)
    failures = []
    cards = []
    for row in rows:
        if row["repetition"] or not row["metrics"]["wrong_approved_count"]:
            continue
        record = records[row["image_id"]]
        incorrect = {
            item["segmentation_id"]: item
            for item in row["metrics"]["items"]
            if item["status"] == "APPROVED"
            and item["predicted_class_id"] != item["target_class_id"]
        }
        with Image.open(record["image_path"]) as original:
            original = ImageOps.exif_transpose(original).convert("RGB")
            canvas = Image.new("RGB", (1200, 1000), "#f7f8fa")
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (24, 18),
                f"최종 평가 {row['image_id']}번 — 오승인 {len(incorrect)}건",
                font=font,
                fill="#20252d",
            )
            draw.text(
                (24, 55),
                "왼쪽: GT(초록) / 오른쪽: Worker(오승인 빨강, 정상 승인 파랑, 기타 주황)",
                font=small,
                fill="#454b56",
            )
            for column in range(2):
                preview = ImageOps.contain(original, (570, 780))
                ox, oy = 20 + column * 600 + (570 - preview.width) // 2, 100
                canvas.paste(preview, (ox, oy))
                scale = preview.width / original.width
                boxes = []
                if column == 0:
                    for index, annotation in enumerate(record["annotations"]):
                        category = int(annotation["category_id"])
                        boxes.append(
                            (
                                annotation["bbox"],
                                f"GT {index + 1}: {names.get(f'bread_{category:02d}', str(category))}",
                                "#078643",
                            )
                        )
                else:
                    for item in row["response"]["segmentations"]:
                        b = item["bbox"]
                        label = (item.get("prediction") or {}).get("class_id")
                        color = (
                            "#d52835"
                            if item["segmentation_id"] in incorrect
                            else "#246ac1"
                            if item["status"] == "APPROVED"
                            else "#db8b10"
                        )
                        boxes.append(
                            (
                                [b["x"], b["y"], b["width"], b["height"]],
                                f"{item['segmentation_id'].split('_')[-1]} {names.get(label, item['status'])}",
                                color,
                            )
                        )
                for (x, y, w, h), label, color in boxes:
                    x1, y1 = ox + x * scale, oy + y * scale
                    draw.rectangle((x1, y1, x1 + w * scale, y1 + h * scale), outline=color, width=3)
                    bounds = draw.textbbox((x1, y1), label, font=small)
                    draw.rectangle(bounds, fill="white")
                    draw.text((x1, y1), label, font=small, fill=color)
            explanations = []
            for item in incorrect.values():
                predicted = names.get(item["predicted_class_id"], item["predicted_class_id"])
                target = names.get(item["target_class_id"], "GT 미대응 추가 검출")
                explanations.append(f"{item['segmentation_id']}: {target} → {predicted} APPROVED")
                failures.append({"image_id": row["image_id"], **item})
            for index, explanation in enumerate(explanations):
                draw.text((24, 890 + index * 28), explanation, font=small, fill="#a21d2c")
            draw.text(
                (24, 961),
                "GT와 클래스 무관 IoU ≥ 0.5 일대일 대응. GT/정책은 평가 후 수정하지 않음.",
                font=small,
                fill="#454b56",
            )
            canvas.save(reports / f"false-approval-{row['image_id']}.jpg", quality=92)
            cards.append(ImageOps.contain(canvas, (600, 500)))
    if cards:
        sheet = Image.new("RGB", (1200, 500 * ((len(cards) + 1) // 2)), "white")
        for index, card in enumerate(cards):
            sheet.paste(card, ((index % 2) * 600, (index // 2) * 500))
        sheet.save(reports / "false-approvals.jpg", quality=92)
    write_json(reports / "false-approvals.json", failures)
    (reports / "0.1.16-vs-0.1.17.md").write_text(
        (root / "docs/architecture/scanner-0.1.17.md")
        .read_text(encoding="utf-8")
        .replace("../../configs/versions/0.1.17.json", "version-config.json"),
        encoding="utf-8",
    )
    shutil.copy2(root / "configs/versions/0.1.17.json", reports / "version-config.json")
    text = [
        "# 0.1.17 최종 검증 결과",
        "",
        "사용자가 지정한 현재 1등 `ssdlite-margin-dense-20260908`을 그대로 배포한다.",
        "최종 300장·1,410 GT는 학습·validation·후보/CPU 설정 선택에 쓰지 않았다.",
        "정답 대응은 클래스 무관 bbox IoU ≥0.5, 최대 대응 수 우선·총 IoU 차선이다.",
        "정확도 분모에는 누락·UNKNOWN·재촬영·ERROR도 포함한다. 같은 결과를 세 번 측정한 것으로 독립 표본 900장이라고 표현하지 않는다.",
        "",
        "**최종 세 목표 동시 달성: 미달.** 모델·Catalog·정책·CPU 설정은 최종 평가 후 변경하지 않았다.",
        "",
        "| Provider / 반복 | 올바른 APPROVED / GT | 승인율 | 오승인 | 완전 성공 이미지 | HTTP p50 | p95 | p99 | 최대 | n |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for provider, report in result["providers"].items():
        for i, value in enumerate(report["repetitions"], 1):
            latency = value["latency"]["all"]
            text.append(
                f"| {provider.upper()} / {i} | {value['correct_approved_count']}/{value['ground_truth_count']} | {value['correct_approved_rate']:.2%} | {value['wrong_approved_count']} | {value['complete_images']}/300 | {latency['p50_ms']:.2f}ms | {latency['p95_ms']:.2f}ms | {latency['p99_ms']:.2f}ms | {latency['max_ms']:.2f}ms | {latency['count']} |"
            )
    text += [
        "",
        "전체 요청과 full-path의 지연 통계를 각각 JSON에 보존했다. 이미지 전체 재촬영·ERROR 표본이 없으므로 해당 p95는 null이며 full-path와 전체 요청은 같다.",
        "CPU 기준: Intel Core Ultra 9 285K, ONNX Runtime CPU 1.29.0, Detector 8 / Embedder 12 threads, 동시 요청 1개, warmup 10회, 실제 EXE HTTP 왕복. 파일 사전 읽기와 시작 모델 로딩은 제외한다.",
        "CUDA: RTX 5080, ONNX Runtime GPU 1.28.0. CPU/CUDA는 같은 ONNX·Catalog·정책이다.",
        "",
        f"CPU/CUDA 상태·품목 순위 parity: **{result['provider_parity']['status_rank_parity']}**; 비교 요청 {result['provider_parity']['compared_request_count']}개.",
        "",
        "## 오승인 사례",
        "",
        "![오승인 4개 이미지](false-approvals.jpg)",
        "",
    ]
    for f in failures:
        text.append(
            f"- [{f['image_id']}번](false-approval-{f['image_id']}.jpg): {names.get(f['target_class_id'], 'GT 미대응 추가 검출')} → {names.get(f['predicted_class_id'], f['predicted_class_id'])} APPROVED"
        )
    text += [
        "",
        "145번은 IoU 대응 기준상 GT 미대응 추가 승인으로 집계한다. 이 분류명이 사진에 실제 빵이 없다는 뜻은 아니다. GT가 틀렸다고 단정하거나 수정하지 않았다.",
        "같은 실물 개발 성적 645/649(99.38%)·오승인 0건과 최종 성적은 구분한다. 이전 0.1.16은 학습·평가 데이터가 달라 개선 배수를 직접 비교하지 않는다.",
        "",
        "## 배포 검증",
        "",
        "Python 전체 1,147개, Scanner Flutter 193개, Lite 24개, SDK 6개 테스트 통과. 세 Flutter analyze, Ruff check/format, git diff --check 통과.",
        "실제 CPU/CUDA Worker EXE의 readiness·정상 스캔·손상 이미지·누락 입력·미지원 형식 smoke 통과. Runtime/Catalog/Worker API 버전 0.1.17 및 checksum을 검증했다.",
        "설치 EXE 생성과 portable Worker 실행을 검증했다. 이번 실행에서 시스템에 설치·제거하거나 실제 카메라 하드웨어를 시험하지는 않았다.",
        "체크섬은 손상·변경을 검출하지만 발행자 인증은 제공하지 않는다. 배포 EXE는 Authenticode 미서명이다.",
        "",
    ]
    (reports / "verification.md").write_text("\n".join(text), encoding="utf-8")
    for name in ("packaged-cpu-smoke.json", "packaged-cuda-smoke.json"):
        shutil.copy2(version / name, reports / name)
    shutil.copy2(final / "report.json", reports / "final-evaluation.json")
    shutil.copy2(final / "final-candidate.json", reports / "final-candidate.json")
    write_json(
        reports / "release-source.json",
        {
            "source_checkout": str(root),
            "product_version": "0.1.17",
            "version_config_sha256": sha256_file(root / "configs/versions/0.1.17.json"),
            "source_model_directory": directory_content_manifest(version / "source/runtime"),
            "source_catalog_directory": directory_content_manifest(version / "source/catalog"),
        },
    )
    print(
        json.dumps(
            {"report": str(reports / "verification.md"), "failures": failures}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
