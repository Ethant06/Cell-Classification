"""
Create augmented views of one or more images for figures (training code unchanged).

Rotation/flip are fixed by --rotation-deg; random erasing uses fresh entropy each run (no fixed seed).

For each input image, writes the same five PNGs into its own output folder (see below).

Usage (repo root):
  python scripts/augment_one_image.py example_cell.png
  python scripts/augment_one_image.py example_cell.png data4/normal/normal137.png
  python scripts/augment_one_image.py normal137.png --out my_exports   # all images under my_exports/<name>_augmented/
"""

from __future__ import annotations

import argparse
import secrets
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF


def rotate_crop_to_square(pil: Image.Image, angle: float, size: int = 128, fill: int = 0) -> Image.Image:
    r = TF.rotate(pil, angle, expand=True, fill=fill)
    w, h = r.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    r = r.crop((left, top, left + side, top + side))
    return r.resize((size, size), Image.Resampling.BILINEAR)


def denorm_to_pil(t: torch.Tensor) -> Image.Image:
    x = t.clone().squeeze(0)
    x = x * 0.5 + 0.5
    x = x.clamp(0, 1)
    x = (x * 255).byte().cpu().numpy()
    return Image.fromarray(x, mode="L")


def random_erasing_center_focused(
    x: torch.Tensor,
    *,
    scale: tuple[float, float],
    ratio: tuple[float, float],
    value: float | str,
    center_fraction: float,
) -> torch.Tensor:
    """
    Like torchvision RandomErasing (area + aspect sampling) but the patch *center* is sampled
    uniformly inside the middle ``center_fraction`` of height and width, so boxes avoid edges.
    x: (C, H, W) in [0, 1].
    """
    C, H, W = x.shape
    device, dtype = x.device, x.dtype
    area = H * W

    def u(a: float, b: float) -> float:
        t = torch.empty((), device=device, dtype=torch.float32)
        t.uniform_(a, b)
        return float(t.item())

    target_area = u(scale[0], scale[1]) * area
    aspect = u(ratio[0], ratio[1])
    h = int(round((target_area * aspect) ** 0.5))
    w = int(round((target_area / aspect) ** 0.5))
    h = min(max(h, 1), H)
    w = min(max(w, 1), W)

    half = center_fraction / 2.0
    y0, y1 = H * half, H * (1.0 - half)
    x0, x1 = W * half, W * (1.0 - half)
    cy_lo = max(h / 2.0, y0)
    cy_hi = min(H - h / 2.0, y1)
    cx_lo = max(w / 2.0, x0)
    cx_hi = min(W - w / 2.0, x1)
    if cy_lo > cy_hi or cx_lo > cx_hi:
        cy_lo, cy_hi = h / 2.0, H - h / 2.0
        cx_lo, cx_hi = w / 2.0, W - w / 2.0

    cy = u(cy_lo, cy_hi)
    cx = u(cx_lo, cx_hi)
    top = max(0, min(int(round(cy - h / 2.0)), H - h))
    left = max(0, min(int(round(cx - w / 2.0)), W - w))

    out = x.clone()
    if isinstance(value, str) and value == "random":
        out[:, top : top + h, left : left + w] = torch.rand(
            C, h, w, device=device, dtype=dtype
        )
    else:
        out[:, top : top + h, left : left + w] = float(value)
    return out


def output_dir_for(src: Path, out_parent: Path | None, n_inputs: int) -> Path:
    """Pick folder so multiple images never overwrite each other."""
    stem = src.stem
    if out_parent is None:
        if n_inputs == 1:
            return src.parent / "augmented_outputs"
        return src.parent / f"augmented_outputs_{stem}"
    out_parent = out_parent.expanduser().resolve()
    if n_inputs == 1:
        return out_parent
    return out_parent / f"{stem}_augmented"


def process_one(
    src: Path,
    out_dir: Path,
    *,
    rotation_deg: float,
    legacy_rotate: bool,
    fill: int,
    erase_scale: tuple[float, float],
    erase_ratio: tuple[float, float],
    erase_near_center: bool,
    erase_center_fraction: float,
    erase_fill: float | str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pil = Image.open(src).convert("L")
    pil = pil.resize((128, 128), Image.Resampling.BILINEAR)

    pil.save(out_dir / "01_original_128_gray.png")

    fill_c = max(0, min(255, fill))
    if legacy_rotate:
        rot = TF.rotate(pil, rotation_deg, fill=fill_c)
        combo = TF.hflip(TF.rotate(pil, rotation_deg, fill=fill_c))
    else:
        rot = rotate_crop_to_square(pil, rotation_deg, fill=fill_c)
        combo = TF.hflip(rotate_crop_to_square(pil, rotation_deg, fill=fill_c))

    rot.save(out_dir / f"02_rotated_{rotation_deg:g}deg.png")
    TF.hflip(pil).save(out_dir / "03_horizontal_flip.png")
    combo.save(out_dir / f"04_rot{rotation_deg:g}deg_then_flip.png")

    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    x = to_tensor(combo)
    torch.manual_seed(secrets.randbelow(2**31))
    if erase_near_center:
        x = random_erasing_center_focused(
            x,
            scale=erase_scale,
            ratio=erase_ratio,
            value=erase_fill,
            center_fraction=erase_center_fraction,
        )
    else:
        erase = transforms.RandomErasing(
            p=1.0, scale=erase_scale, ratio=erase_ratio, value=erase_fill
        )
        x = erase(x.unsqueeze(0)).squeeze(0)
    x = normalize(x)
    denorm_to_pil(x).save(out_dir / "05_rot_flip_plus_random_erasing_reg.png")


def main() -> None:
    p = argparse.ArgumentParser(description="Export augmented variants of one or more images.")
    p.add_argument(
        "image_paths",
        type=Path,
        nargs="+",
        help="One or more input images (e.g. example_cell.png data4/normal/normal137.png)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output: for one image, this is the folder. For several images, each is written to OUT/<stem>_augmented/.",
    )
    p.add_argument("--rotation-deg", type=float, default=15.0)
    p.add_argument(
        "--erase-near-center",
        action="store_true",
        help="Place erase patch so its center lies in the image middle (helps visibility on black-bordered cells).",
    )
    p.add_argument(
        "--erase-center-fraction",
        type=float,
        default=0.55,
        metavar="F",
        help="With --erase-near-center: patch center is uniform in the central F of W and H (default 0.55).",
    )
    p.add_argument(
        "--erase-value",
        choices=("black", "gray", "white", "random"),
        default="gray",
        help="Erase fill before Normalize: black=0 (matches training; invisible on black bg), gray default for figures.",
    )
    p.add_argument(
        "--erase-scale",
        type=float,
        nargs=2,
        default=[0.05, 0.05],
        metavar=("MIN", "MAX"),
        help="RandomErasing erased area as fraction of image (torchvision default 0.02 0.1)",
    )
    p.add_argument(
        "--erase-ratio",
        type=float,
        nargs=2,
        default=[0.6, 0.6],
        metavar=("MIN", "MAX"),
        help="Aspect ratio (w/h) of erase box (torchvision default 0.1, 0.1)",
    )
    p.add_argument("--legacy-rotate", action="store_true")
    p.add_argument("--fill", type=int, default=0, metavar="0-255")
    args = p.parse_args()

    es = tuple(args.erase_scale)
    er = tuple(args.erase_ratio)
    if es[0] <= 0 or es[1] <= 0 or es[0] > es[1]:
        raise SystemExit("--erase-scale: need 0 < MIN and MIN <= MAX")
    if er[0] <= 0 or er[1] <= 0 or er[0] > er[1]:
        raise SystemExit("--erase-ratio: need 0 < MIN and MIN <= MAX (use 1 1 for a square box)")
    cf = args.erase_center_fraction
    if not (0.0 < cf <= 1.0):
        raise SystemExit("--erase-center-fraction: need 0 < F <= 1")

    _fill: dict[str, float | str] = {
        "black": 0.0,
        "gray": 0.5,
        "white": 1.0,
        "random": "random",
    }
    erase_fill = _fill[args.erase_value]

    paths = [x.expanduser().resolve() for x in args.image_paths]
    for src in paths:
        if not src.is_file():
            raise SystemExit(f"Not found: {src}")

    n = len(paths)
    for src in paths:
        od = output_dir_for(src, args.out, n)
        process_one(
            src,
            od,
            rotation_deg=args.rotation_deg,
            legacy_rotate=args.legacy_rotate,
            fill=args.fill,
            erase_scale=es,
            erase_ratio=er,
            erase_near_center=args.erase_near_center,
            erase_center_fraction=cf,
            erase_fill=erase_fill,
        )
        print(f"Wrote: {od}")


if __name__ == "__main__":
    main()
