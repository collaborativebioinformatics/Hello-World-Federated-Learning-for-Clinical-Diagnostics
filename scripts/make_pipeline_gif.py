"""Make docs/pipeline_live/pipeline.gif, the README's moving picture of the pipeline page.

Every frame is a still of docs/pipeline_live/index.html taken by headless Chrome through
the page's own ?scene=N&pause=MS parameters: the page opens on scene N, steps its clock
to MS milliseconds and freezes, so a frame depends on nothing but those two numbers and
the same command gives the same GIF. The first three scenes are sampled every 800 ms of
page time and shown for 250 ms a frame, about three times life speed. The fourth scene,
a doctor asking about three variants, is 40 s long and is sampled at the moments that
matter in each of its four stories: the question leaving, the reading from Oslo alone,
the question and the counts crossing between hospitals, the reading with all three
hospitals, and the model's line where there is one. The story timings come from
playStory() in the page's script.

Each still is cropped under the scene stepper, which drops the keyboard hint and the
footer, and the page's "paused" mark, which its still mode switches on, is painted over
in the background colour. A page hook for stills without that mark would remove the
painting; until then the mark sits at a fixed place right of the stepper.

Frames share one 256-colour palette, chosen by an octree over thumbnails of every frame so
that the few bright colours survive, and Pillow then writes each frame as the difference
from the one before; a frame identical to its predecessor only lengthens it.

Pillow is not a project dependency, as markdown is not for scripts/build_manuscript.py:

    uv run --with pillow python scripts/make_pipeline_gif.py
    uv run --with pillow python scripts/make_pipeline_gif.py --frames-dir some/folder   # also keep the PNG stills, for a look

Writes docs/pipeline_live/pipeline.gif and prints its size, frame count and length.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image  # supplied by `uv run --with pillow`

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_manuscript import find_browser  # noqa: E402  the same browser search as the manuscript build

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "pipeline_live" / "index.html"
OUT = ROOT / "docs" / "pipeline_live" / "pipeline.gif"

WINDOW = (1200, 770)  # a laptop window; the picture is 1070 by 600 units and fits with the caption and stepper under it
CROP = (0, 0, 1200, 715)  # under the stepper row, so the keyboard hint and the footer are left out
PAUSED_MARK = (1090, 684, 1200, 714)  # where the page writes "paused" when frozen, right of the stepper
HOLD, HOLD_LONG = 250, 600  # ms a frame; the long one for a reading, so it can be read
SCENE_LENGTH = {0: 7400, 1: 9200, 2: 12600}  # SCENES[].dur in the page, for the scenes sampled evenly
STEP, FIRST = 800, 400  # every 800 ms of page time, from 400 ms in
# Scene 4 plays four stories in a row, each through playStory(): 700 ms, the question to the doctor 1000,
# 600, the question to the other two hospitals 300 + 1400, their counts back 300 + 1400, the counts to the
# doctor 800, 800, the model's line 800 for a missense variant, then 2200: 10300 ms for a missense variant,
# 9500 for the deletion. The MYBPC3 deletion is played twice, as typed and with one spelling, 300 ms apart,
# and every other story is followed by 400 ms. The page steps its clock in 16.7 ms hops, so each story
# starts a little later than this sum, and the moments below sit in the middle of their phase.
STORY_STARTS = [0, 10700, 21400, 31200]  # DSP N1526K; TTR V142I; the MYBPC3 deletion as typed; with one spelling
STORY_MOMENTS = [(400, HOLD), (1300, HOLD), (2200, HOLD_LONG), (3300, HOLD), (4900, HOLD), (6300, HOLD), (7100, HOLD_LONG)]
MODEL_LINE = (9000, HOLD)  # the first two stories are missense variants, so the trained model adds its line
COLOURS = 256
WORKERS = 4


def plan() -> list[tuple[int, int, int]]:
    """(scene, pause in ms, hold in ms) for every frame, in order."""
    frames = []
    for scene, length in SCENE_LENGTH.items():
        frames += [(scene, pause, HOLD) for pause in range(FIRST, length, STEP)]
    for story, start in enumerate(STORY_STARTS):
        moments = STORY_MOMENTS + ([MODEL_LINE] if story < 2 else [])
        frames += [(3, start + offset, hold) for offset, hold in moments]
    return frames


def capture(browser: str, scene: int, pause: int, png: Path, profile: Path) -> None:
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    f"--window-size={WINDOW[0]},{WINDOW[1]}", f"--user-data-dir={profile}",
                    f"--screenshot={png}", f"{PAGE.as_uri()}?scene={scene}&pause={pause}"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)


def tidy(png: Path) -> Image.Image:
    """The still without the paused mark, cropped under the stepper."""
    image = Image.open(png).convert("RGB")
    image.load()
    x0, y0, x1, y1 = PAUSED_MARK
    image.paste(image.getpixel((x0 - 5, y0)), (x0, y0, x1, y1))
    return image.crop(CROP)


def main() -> int:
    parser = argparse.ArgumentParser(description="Make the pipeline GIF from stills of docs/pipeline_live/index.html")
    parser.add_argument("--frames-dir", type=Path, help="keep the PNG stills in this folder")
    parser.add_argument("--workers", type=int, default=WORKERS, help="Chrome processes at a time")
    args = parser.parse_args()

    if not PAGE.exists():
        print(f"{PAGE} is missing. Run scripts/export_pipeline_state.py first.", file=sys.stderr)
        return 1
    browser = find_browser()
    frames = plan()

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        stills = args.frames_dir or Path(folder) / "frames"
        stills.mkdir(parents=True, exist_ok=True)

        def one(item: tuple[int, tuple[int, int, int]]) -> Image.Image:
            index, (scene, pause, _) = item
            png = stills / f"frame_{index:03d}_scene{scene + 1}_{pause:05d}ms.png"
            capture(browser, scene, pause, png, Path(folder) / f"profile_{index}")
            return tidy(png)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            images = list(pool.map(one, enumerate(frames)))

    kept: list[Image.Image] = []
    holds: list[int] = []
    for image, (_, _, hold) in zip(images, frames):
        if kept and image.tobytes() == kept[-1].tobytes():
            holds[-1] += hold  # a still identical to the last one only lengthens it
            continue
        kept.append(image)
        holds.append(hold)

    # one palette for every frame, chosen on a strip of thumbnails
    width, height = kept[0].size
    thumb = (width // 4, height // 4)
    strip = Image.new("RGB", (thumb[0], thumb[1] * len(kept)))
    for row, image in enumerate(kept):
        strip.paste(image.resize(thumb), (0, row * thumb[1]))
    # octree rather than median cut: the page is mostly greys, and median cut merges the few bright
    # greens, oranges and blues into them, which dulls the badges; octree keeps them
    palette = strip.quantize(colors=COLOURS, method=Image.Quantize.FASTOCTREE)
    paletted = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in kept]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    paletted[0].save(OUT, save_all=True, append_images=paletted[1:], duration=holds, loop=0, disposal=1, optimize=False)
    size = OUT.stat().st_size
    print(f"wrote {OUT.relative_to(ROOT)}: {size / 1e6:.2f} MB, {len(kept)} frames "
          f"({len(frames) - len(kept)} identical stills folded in), {sum(holds) / 1000:.1f} s a loop, {width}x{height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
