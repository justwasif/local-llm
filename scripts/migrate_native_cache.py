"""Use a standard site-packages path so dill fingerprints dependencies by reference."""

from pathlib import Path

root = Path.home() / ".cache/local-llm"
for environment in ["train", "serve"]:
    source = root / "overlays" / environment
    target = root / "native-packages" / environment / "site-packages"
    assert source.resolve().is_relative_to(root.resolve())
    assert target.resolve().is_relative_to(root.resolve())
    if source.exists() and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
    print(f"Native cache: {target}")
