"""Downloads the ONCET checkpoint at image build time and checks its SHA-256.

A script file, not a RUN heredoc: the App Store builds with kaniko, which does not
support Dockerfile heredocs and ran the heredoc version as `python3 -` with no input,
so the checkpoint was silently missing from platform-built images."""
import hashlib, os, urllib.request
base = "https://github.com/wojemann/DynaSD/releases/download/v0.1.0-checkpoints"
files = {
    "ONCET/best_model.pth": ("best_model.pth", "34afd9c56ccd11d66f97bd224e3e0447ceae07ac9b3e503c30939d37f59a5c2f"),
    "ONCET/final_training_config.json": ("final_training_config.json", "0481bbcc0e69c855112e5aa7d56737725f05662dd35162952d63983ca5e07dda"),
}
for rel, (name, sha) in files.items():
    dst = os.path.join(os.environ["DYNASD_CACHE_DIR"], rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    data = urllib.request.urlopen(f"{base}/{name}").read()
    got = hashlib.sha256(data).hexdigest()
    if got != sha:
        raise SystemExit(f"{name}: sha256 {got} != {sha}")
    open(dst, "wb").write(data)
    print("checkpoint", rel, len(data), "bytes, sha256 ok")
