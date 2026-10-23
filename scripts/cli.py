"""Generate one image. The prompt goes through the local upsampler first unless --no-pup is given.

  python scripts/cli.py "a red bicycle" [--model distilled] [--image a.png ...] [--boxes boxes.json] --out out.png
"""

import argparse
import json
from pathlib import Path

from PIL import Image

from flux3.generate import Models, Refused, generate
from flux3.pipeline import MAX_REFERENCES, MODELS, WEIGHTS, dit_path
from flux3.safety import exif


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--model", choices=MODELS, default="base", help="base: 50 steps, CFG 4; distilled: 16 steps, no CFG")
    parser.add_argument("--image", action="append", default=[], help=f"reference image; repeat for several (max {MAX_REFERENCES})")
    parser.add_argument("--boxes", help="JSON list of boxes, see README")
    parser.add_argument("--out", default="out.png")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance", type=float, help="CFG scale (base only)")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-pup", action="store_true", help="send the prompt as typed instead of upsampling it first; only for a prompt that already is an upsampler caption")
    parser.add_argument("--pup-memory", type=float, default=0.35, help="share of the GPU reserved for the upsampler")
    parser.add_argument("--dit", help=f"DiT checkpoint, instead of {WEIGHTS}/<model weights entry>")
    parser.add_argument("--safety", action="store_true", help="check the prompt, references and output; refuse if flagged")
    args = parser.parse_args()

    dit = args.dit or str(dit_path(args.model))
    models = Models({args.model: dit}, None if args.no_pup else str(WEIGHTS / "pup"), args.pup_memory)
    boxes = json.loads(Path(args.boxes).read_text()) if args.boxes else None
    images = [Image.open(path) for path in args.image]
    try:
        image, record = generate(models, args.model, args.prompt, images, boxes, args.width, args.height,
                                 args.steps, args.guidance, args.seed, pup=not args.no_pup, safety=args.safety)
    except Refused as error:
        raise SystemExit(f"Blocked by the safety check: {error}")
    print(f"prompt: {record['prompt']}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    image.save(args.out, exif=exif(args.model))
    record |= {"references": args.image, "dit": dit}
    Path(args.out).with_suffix(".json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
