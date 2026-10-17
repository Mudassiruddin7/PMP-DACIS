"""Paper build helpers: Tectonic bootstrap, LaTeX compile, results bundle."""
import io
import os
import platform
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile

TECTONIC_VERSION = "0.17.0"


def _tectonic_asset():
    sysname, mach = platform.system(), platform.machine().lower()
    if sysname == "Linux":
        arch = "aarch64" if mach in ("aarch64", "arm64") else "x86_64"
        return f"tectonic-{TECTONIC_VERSION}-{arch}-unknown-linux-musl.tar.gz"
    if sysname == "Darwin":
        arch = "aarch64" if mach in ("arm64", "aarch64") else "x86_64"
        return f"tectonic-{TECTONIC_VERSION}-{arch}-apple-darwin.tar.gz"
    return f"tectonic-{TECTONIC_VERSION}-x86_64-pc-windows-msvc.zip"


def ensure_tectonic(bin_dir):
    """Path to a working `tectonic` executable (downloaded once into bin_dir), or None."""
    found = shutil.which("tectonic")
    if found:
        return found
    os.makedirs(bin_dir, exist_ok=True)
    exe = os.path.join(bin_dir, "tectonic.exe" if platform.system() == "Windows" else "tectonic")
    if os.path.exists(exe):
        return exe
    asset = _tectonic_asset()
    url = f"https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%40{TECTONIC_VERSION}/{asset}"
    try:
        data = urllib.request.urlopen(url, timeout=120).read()
        if asset.endswith(".zip"):
            zipfile.ZipFile(io.BytesIO(data)).extractall(bin_dir)
        else:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
                tf.extractall(bin_dir)
        os.chmod(exe, 0o755)
        return exe
    except Exception as e:  # network or platform problem: the caller reports and continues
        print("could not fetch tectonic:", e)
        return None


def compile_pdf(tectonic, paper_dir, tex_name, timeout=900):
    """Compile paper_dir/tex_name with Tectonic. Returns (pdf_path or None, log text)."""
    if tectonic is None:
        return None, "tectonic unavailable"
    p = subprocess.run([tectonic, tex_name], cwd=paper_dir, capture_output=True, text=True, timeout=timeout)
    pdf = os.path.join(paper_dir, os.path.splitext(tex_name)[0] + ".pdf")
    return (pdf if p.returncode == 0 and os.path.exists(pdf) else None), (p.stdout or "") + (p.stderr or "")


def zip_results(root, extra_files=(), skip_ext=(".pt",)):
    """In-memory zip of everything under root except checkpoints (and the data/cache folders)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6, strict_timestamps=False) as z:
        for base, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in ("data", "cache", "bin", "__pycache__", "src")]
            for f in files:
                if f.endswith(skip_ext):
                    continue
                full = os.path.join(base, f)
                z.write(full, os.path.relpath(full, root))
        for path, arc in extra_files:
            if os.path.exists(path):
                z.write(path, arc)
    return buf.getvalue()
