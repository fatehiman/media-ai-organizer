"""Write assets/app.ico (used by the exes) from media_organizer.icon.
Run:  python scripts/make_icon.py"""
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "src"))
from media_organizer.icon import make_icon  # noqa: E402

out = root / "assets" / "app.ico"
out.parent.mkdir(exist_ok=True)
make_icon(256).save(out, sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])
make_icon(256).save(root / "assets" / "app.png")
print("wrote", out)
