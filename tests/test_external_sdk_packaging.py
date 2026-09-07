from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_external_sdk_and_store_model_versions_are_packaged_independently() -> None:
    script = (ROOT / "scripts" / "build_external_sdk.ps1").read_text(encoding="utf-8")

    assert '[string]$ModelVersion = "0.1.16"' in script
    assert '[string]$SdkVersion = "1.1.0"' in script
    assert '[ValidateSet("All", "Sdk", "StoreModel")]' in script
    assert '"BIXOLON-Scanner-SDK-Windows-x64-$SdkVersion"' in script
    assert '"BIXOLON-Store-Model-$storeId-$ModelVersion"' in script
    assert "Join-Path $resolvedSdkOutputRoot $SdkVersion" in script
    assert 'Join-Path $resolvedStoreModelOutputRoot "$storeId/$ModelVersion"' in script
    assert 'distribution = "BIXOLON_SCANNER_SDK"' in script
    assert "supported_worker_runtime_schemas" in script
    assert "supported_catalog_schemas" in script
    assert "BIXOLON-Scanner-External-SDK" not in script
    assert script.index('$buildSdk = $Package -in @("All", "Sdk")') < script.index(
        "$requiredDirectories"
    )
    assert "if ($buildSdk) {\n    $requiredDirectories +=" in script


def test_store_model_installer_accepts_new_identity_and_legacy_bundle() -> None:
    script = (ROOT / "sdk" / "external" / "deployment" / "install-store-bundle.ps1").read_text(
        encoding="utf-8"
    )

    assert '$identitySchema -notin @("1.0", "1.1")' in script
    assert "[string]$identity.model_version" in script
    assert "[string]$identity.product_version" in script
    assert 'schema_version = "1.1"' in script
    assert "model_version = $modelVersion" in script


def test_external_sdk_docs_describe_independent_versions_and_selection() -> None:
    readme = (ROOT / "sdk" / "external" / "README-KO.md").read_text(encoding="utf-8")

    assert "BIXOLON-Scanner-SDK-Windows-x64-<sdk-version>.zip" in readme
    assert "BIXOLON-Store-Model-<store>-<model-version>.zip" in readme
    assert "SDK와 Store Model의 version은 독립적입니다." in readme
    assert "storeBundleRoot" in readme
