"""The app icon: a photo album with two pictures on its cover.

Drawn with Pillow, so the window icon needs no file, and
scripts/make_icon.py can write the .ico for the exe."""
from PIL import Image, ImageDraw

_S = 1024          # drawn big, then scaled down (smooth edges)


def _photo(size: int) -> Image.Image:
    """A picture: white frame, sky, sun and two hills."""
    w, h = size, int(size * 0.78)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=w // 14, fill=(255, 255, 255, 255))
    m = w // 12
    box = (m, m, w - m, h - m)
    inner = Image.new("RGBA", (box[2] - box[0], box[3] - box[1]), (0, 0, 0, 0))
    iw, ih = inner.size
    di = ImageDraw.Draw(inner)
    di.rectangle((0, 0, iw, ih), fill=(120, 195, 245, 255))                 # sky
    di.ellipse((iw * 0.62, ih * 0.12, iw * 0.88, ih * 0.12 + iw * 0.26),
               fill=(255, 205, 60, 255))                                     # sun
    di.polygon([(0, ih), (iw * 0.34, ih * 0.38), (iw * 0.68, ih)],
               fill=(70, 150, 95, 255))                                      # hills
    di.polygon([(iw * 0.4, ih), (iw * 0.72, ih * 0.52), (iw, ih)],
               fill=(45, 115, 75, 255))
    mask = Image.new("L", inner.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, iw - 1, ih - 1), radius=w // 30, fill=255)
    img.paste(inner, box[:2], mask)
    return img


def make_icon(size: int = 256) -> Image.Image:
    base = Image.new("RGBA", (_S, _S), (0, 0, 0, 0))
    d = ImageDraw.Draw(base)
    # Album cover and spine.
    d.rounded_rectangle((70, 90, 954, 934), radius=90, fill=(32, 92, 120, 255))
    d.rounded_rectangle((70, 90, 954, 934), radius=90, outline=(18, 60, 82, 255), width=14)
    d.rounded_rectangle((70, 90, 250, 934), radius=90, fill=(18, 60, 82, 255))
    d.rectangle((160, 90, 250, 934), fill=(18, 60, 82, 255))
    for y in (230, 512, 794):                       # binding rings
        d.ellipse((205, y - 38, 281, y + 38), fill=(235, 190, 70, 255))
    # Two pictures on the cover: one behind, tilted; one in front.
    back = _photo(470).rotate(10, expand=True, resample=Image.BICUBIC)
    base.alpha_composite(back, (440, 190))
    front = _photo(500).rotate(-6, expand=True, resample=Image.BICUBIC)
    base.alpha_composite(front, (330, 430))
    return base.resize((size, size), Image.LANCZOS)
