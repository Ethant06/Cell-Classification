"""
Create augmented views of one or more images for figures (training code unchanged).

Rotation/flip are fixed by --rotation-deg; random erasing uses fresh entropy each run (no fixed seed).

For each input image, writes the same five PNGs into its own output folder (see below),
unless ``--erase-only`` is set (then: original 128×128 gray + one erase+normalize figure).

Usage (repo root):
  python scripts/augment_one_image.py example_cell.png
  python scripts/augment_one_image.py example_cell.png data4/normal/normal137.png
  python scripts/augment_one_image.py normal137.png --out my_exports   # all images under my_exports/<name>_augmented/
  python scripts/augment_one_image.py poster_visuals/normal137.png --out augmented_outputs_normal137 --erase-only --erase-figure-name erase_regularized_normal.png
  # Same erase box style as 05_rot_flip_plus_random_erasing_reg.png: default --erase-scale/--erase-ratio/--erase-value; add --erase-near-center for cell figures.
  # Same patch *geometry* as rotate_flip_erase on the aligned pair (diff the images), on an unrotated source:
  # python scripts/augment_one_image.py augmented_outputs_example_cell/original_euploid.png --out augmented_outputs_example_cell --erase-only --erase-figure-name erase_only_same_box_euploid.png --match-erase-box augmented_outputs_example_cell/rotate_flip_euploid.png augmented_outputs_example_cell/rotate_flip_erase_euploid.png
"""

from __future__ import annotations

import argparse
import secrets
from pathlib import Path

import numpy as np
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


def erase_bbox_from_image_pair(
    path_base: Path,
    path_erased: Path,
    *,
    diff_thresh: int = 14,
) -> tuple[int, int, int, int]:
    """
    Axis-aligned bbox of the erase patch: pixels that differ between ``path_base``
    (e.g. rotate+flip, no erase) and ``path_erased`` (same geometry + erase).
    Returns (top, left, h, w) in pixel coords, suitable for a 128×128 tensor.
    """
    a = np.asarray(Image.open(path_base).convert("L"), dtype=np.int16)
    b = np.asarray(Image.open(path_erased).convert("L"), dtype=np.int16)
    if a.shape != b.shape:
        raise ValueError(f"Image shape mismatch: {a.shape} vs {b.shape}")
    mask = np.abs(b - a) > diff_thresh
    if not np.any(mask):
        raise ValueError(
            f"No differing pixels above threshold {diff_thresh} — check paths or try a lower threshold."
        )
    ys, xs = np.where(mask)
    top, left = int(ys.min()), int(xs.min())
    h = int(ys.max() - ys.min() + 1)
    w = int(xs.max() - xs.min() + 1)
    return top, left, h, w


def apply_fixed_erase_rect(
    x: torch.Tensor,
    top: int,
    left: int,
    h: int,
    w: int,
    erase_fill: float | str,
) -> torch.Tensor:
    """Fill axis-aligned rectangle on tensor x (C, H, W) in [0, 1], like RandomErasing."""
    C, H, W = x.shape
    device, dtype = x.device, x.dtype
    top = max(0, min(top, H - 1))
    left = max(0, min(left, W - 1))
    h = max(1, min(h, H - top))
    w = max(1, min(w, W - left))
    out = x.clone()
    if isinstance(erase_fill, str) and erase_fill == "random":
        out[:, top : top + h, left : left + w] = torch.rand(
            C, h, w, device=device, dtype=dtype
        )
    else:
        out[:, top : top + h, left : left + w] = float(erase_fill)
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


def process_erase_only(
    src: Path,
    out_dir: Path,
    *,
    figure_name: str,
    erase_scale: tuple[float, float],
    erase_ratio: tuple[float, float],
    erase_near_center: bool,
    erase_center_fraction: float,
    erase_fill: float | str,
    fixed_rect: tuple[int, int, int, int] | None = None,
) -> None:
    """
    Same erase+normalize step as ``process_one`` (always-on erase, p=1.0), but on the
    resized original only—no rotation or horizontal flip. Uses the same box sampling
    as ``05_rot_flip_plus_random_erasing_reg.png`` when CLI flags match.

    If ``fixed_rect`` is (top, left, h, w), that rectangle is filled instead of random
    sampling (for matching a patch from ``--match-erase-box``).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pil = Image.open(src).convert("L")
    pil = pil.resize((128, 128), Image.Resampling.BILINEAR)

    pil.save(out_dir / "01_original_128_gray.png")

    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    x = to_tensor(pil)
    if fixed_rect is not None:
        t, l, h, w = fixed_rect
        x = apply_fixed_erase_rect(x, t, l, h, w, erase_fill)
    else:
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
    denorm_to_pil(x).save(out_dir / figure_name)


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
    p.add_argument(
        "--erase-only",
        action="store_true",
        help=(
            "Only write 01_original_128_gray.png and the erase+normalize image. "
            "Uses the same erase options as the full export (--erase-scale, --erase-ratio, "
            "--erase-value, --erase-near-center); ignores rotation/flip."
        ),
    )
    p.add_argument(
        "--erase-figure-name",
        type=str,
        default="erase_regularized.png",
        metavar="NAME.png",
        help="Filename for the erase-only result (used with --erase-only).",
    )
    p.add_argument(
        "--match-erase-box",
        nargs=2,
        type=Path,
        metavar=("BASE", "ERASED"),
        default=None,
        help=(
            "With --erase-only: BASE = rotate+flip image without erase, ERASED = same with erase "
            "(e.g. rotate_flip_euploid.png and rotate_flip_erase_euploid.png). "
            "Infers patch bbox from pixel differences and applies the same top,left,h,w on the unrotated input."
        ),
    )
    p.add_argument(
        "--match-erase-thresh",
        type=int,
        default=14,
        metavar="T",
        help="Graylevel difference threshold for --match-erase-box (default 14).",
    )
    args = p.parse_args()

    if args.match_erase_box and not args.erase_only:
        raise SystemExit("--match-erase-box requires --erase-only")

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
    fixed_rect: tuple[int, int, int, int] | None = None
    if args.match_erase_box:
        bp = args.match_erase_box[0].expanduser().resolve()
        ep = args.match_erase_box[1].expanduser().resolve()
        if not bp.is_file() or not ep.is_file():
            raise SystemExit(f"--match-erase-box: not found: {bp} or {ep}")
        fixed_rect = erase_bbox_from_image_pair(bp, ep, diff_thresh=args.match_erase_thresh)
        print(
            f"Erase patch from pair (top, left, h, w) = {fixed_rect} "
            f"(diff_thresh={args.match_erase_thresh})",
            flush=True,
        )

    if args.erase_only:
        for src in paths:
            od = output_dir_for(src, args.out, n)
            process_erase_only(
                src,
                od,
                figure_name=args.erase_figure_name,
                erase_scale=es,
                erase_ratio=er,
                erase_near_center=args.erase_near_center,
                erase_center_fraction=cf,
                erase_fill=erase_fill,
                fixed_rect=fixed_rect,
            )
            print(f"Wrote (erase-only): {od / args.erase_figure_name}")
        return

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
