#!/usr/bin/env python3
"""
Minimal film visual-language reader (outside TouchDesigner).

• Samples keyframes from a video
• Sends frames to OpenAI Vision (GPT‑4o / GPT‑4.1) with a strict schema
• Writes one JSON object per frame to analysis.jsonl
• Also derives simple indices and saves a CSV

Usage:
  export OPENAI_API_KEY=sk-...
  python analyze_video.py --video path/to/film.mp4 --every 12 --model gpt-4o-mini
"""
import os, io, json, base64, argparse, time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

# OpenAI SDK (>=1.0)
from openai import OpenAI

def b64_jpeg(rgb_arr, quality=90):
    im = Image.fromarray(rgb_arr)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def sample_frames(video_path, step=12, max_frames=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            ts = i / fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((ts, rgb))
            if max_frames and len(frames) >= max_frames:
                break
        i += 1
    cap.release()
    return frames

def call_openai_vision(client, model, schema_prompt_text, jpeg_b64, temperature=0.2):
    """
    Calls OpenAI VLM with an image (base64 JPEG) + strict schema prompt.
    Returns a dict parsed from JSON.
    """
    content = [
        {"type":"text","text":schema_prompt_text},
        {"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{jpeg_b64}"}},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"user","content":content}],
        temperature=temperature,
        response_format={ "type": "json_object" } # ask for strict JSON
    )
    txt = resp.choices[0].message.content
    try:
        return json.loads(txt)
    except Exception:
        return {"error":"bad_json","raw":txt}

def derive_indices(r):
    """Compute a few numeric indices for timeline plots."""
    st = r.get("style_indices", {})
    gest = r.get("gesturality", {})
    fig = r.get("figure_presence", {})

    surrealism = float(st.get("surrealism", 0))
    realism = float(st.get("realism", 0))
    gestural = float(gest.get("motion_implied",0) + gest.get("texture_motion",0) + gest.get("blur",0)) / 3.0
    presence = float(fig.get("humans",0) * fig.get("body_visibility",0) * fig.get("faces_visible",0))
    gaze_power = float(fig.get("gaze_to_camera",0)) * float(r.get("power_dynamics",{}).get("subject_power",0))
    return {
        "surrealism_index": max(0.0, min(1.0, (surrealism - realism + 1)/2)),  # map to 0..1
        "gestural_index": max(0.0, min(1.0, gestural)),
        "presence_index": max(0.0, min(1.0, presence)),
        "gaze_power_index": max(0.0, min(1.0, (gaze_power+1)/2)),
    }

def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--every", type=int, default=12, help="Sample every N frames")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional cap")
    parser.add_argument("--model", default=os.getenv("OPENAI_VLM_MODEL","gpt-4o-mini"))
    parser.add_argument("--outdir", default="out_vlm_read")
    parser.add_argument("--per_minute_limit", type=int, default=160, help="Throttle requests if needed")
    args = parser.parse_args()

    video_path = Path(args.video)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    analysis_path = outdir/"analysis.jsonl"
    csv_path = outdir/"indices.csv"
    schema_prompt_text = Path("schema_prompt.txt").read_text()

    api_key = os.getenv("OPENAI_API_KEY","")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in environment.")

    client = OpenAI(api_key=api_key)

    frames = sample_frames(video_path, step=args.every, max_frames=args.max_frames)
    print(f"Sampled {len(frames)} frames. Sending to {args.model}…")

    # Simple rate throttling
    per_minute = args.per_minute_limit
    interval = 60.0 / max(1, per_minute)
    last = 0.0

    with open(analysis_path, "w", encoding="utf-8") as fjson, open(csv_path,"w",encoding="utf-8") as fcsv:
        fcsv.write("t,surrealism_index,gestural_index,presence_index,gaze_power_index\n")
        for t, rgb in tqdm(frames):
            # throttle
            now = time.time()
            if now - last < interval:
                time.sleep(interval - (now - last))
            last = time.time()

            jpeg = b64_jpeg(rgb, quality=90)
            res = call_openai_vision(client, args.model, schema_prompt_text, jpeg)
            res["t"] = t

            # derive and persist
            idx = derive_indices(res if isinstance(res, dict) else {})
            res["_indices"] = idx

            fjson.write(json.dumps(res, ensure_ascii=False) + "\n")
            fcsv.write(f"{t:.3f},{idx['surrealism_index']:.4f},{idx['gestural_index']:.4f},{idx['presence_index']:.4f},{idx['gaze_power_index']:.4f}\n")

    print(f"Done.\nJSONL → {analysis_path}\nCSV   → {csv_path}")

if __name__ == "__main__":
    main()
