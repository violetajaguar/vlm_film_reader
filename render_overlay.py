#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, csv, json, textwrap
from pathlib import Path
import cv2, numpy as np

LABELS = [("Surrealismo","surrealism"),
          ("Gestual","gestural"),
          ("Presencia","presence"),
          ("PoderMirada","gaze")]

# -------- IO --------
def load_indices_csv(path):
    ts, surr, gest, pres, gaze = [], [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row["t"]); ts.append(t)
            surr.append(float(row["surrealism_index"]))
            gest.append(float(row["gestural_index"]))
            pres.append(float(row["presence_index"]))
            gaze.append(float(row["gaze_power_index"]))
    return {"t":np.array(ts,np.float32),
            "surrealism":np.array(surr,np.float32),
            "gestural":np.array(gest,np.float32),
            "presence":np.array(pres,np.float32),
            "gaze":np.array(gaze,np.float32)}

def load_captions_jsonl(path):
    if not path: return None, None
    times, caps = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                times.append(float(r.get("t", 0.0)))
                caps.append(r.get("caption_one_sentence",""))
            except: pass
    return np.array(times,np.float32), caps

# -------- helpers --------
def nearest_caption(t, cap_times, caps):
    if cap_times is None or len(cap_times)==0: return ""
    idx = int(np.clip(np.searchsorted(cap_times, t), 0, len(cap_times)-1))
    if idx>0 and abs(cap_times[idx-1]-t) < abs(cap_times[idx]-t): idx -= 1
    return caps[idx]

def interp(ts, vs, t):
    if t <= ts[0]: return float(vs[0])
    if t >= ts[-1]: return float(vs[-1])
    i = np.searchsorted(ts, t); t0,t1 = ts[i-1], ts[i]; v0,v1 = vs[i-1], vs[i]
    a = 0 if t1==t0 else (t - t0)/(t1 - t0); return float(v0*(1-a)+v1*a)

def put_text_w_shadow(img, txt, org, scale, color=(255,255,255), thick=1):
    x,y = org
    cv2.putText(img, txt, (x+1,y+1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick+2, cv2.LINE_AA)
    cv2.putText(img, txt, (x,y),     cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick,   cv2.LINE_AA)

def draw_sparkline(canvas, ts, vs, t_play):
    h,w = canvas.shape[:2]
    canvas[:] = (30,30,30)
    cv2.line(canvas,(0,h//2),(w-1,h//2),(55,55,55),1,cv2.LINE_AA)
    xs = np.linspace(0,1,w); pts=[]
    t0,t1 = float(ts[0]), float(ts[-1])
    for x in xs:
        tt = t0 + x*(t1-t0); v = interp(ts,vs,tt)
        y = int((1.0 - v) * (h-1)); pts.append([len(pts), y])
    pts = np.array(pts,np.int32)
    cv2.polylines(canvas,[pts],False,(200,200,200),1,cv2.LINE_AA)
    xph = int(np.clip((t_play-t0)/(t1-t0+1e-9),0,1)*(w-1))
    cv2.line(canvas,(xph,0),(xph,h-1),(240,240,240),1,cv2.LINE_AA)

# -------- panel --------
def draw_panel_on_frame(frame, t, idx, cap_text, *, topleft, alpha, scale, font_scale,
                        caption_band=0.30, caption_lines=4, caption_chars=52):
    """
    caption_band: % of panel height reserved for caption (0..1)
    caption_lines: max wrapped lines
    caption_chars: approx chars/line at scale=1 (auto scales)
    """
    h,w = frame.shape[:2]
    # wider by default: 90% of video width
    base_w, base_h = int(0.9*w), 260
    panel_w, panel_h = int(base_w*scale), int(base_h*scale)
    x0 = 20; y0 = 20 if topleft else h - panel_h - 20

    # keep panel on-screen
    panel_w = min(panel_w, w - x0 - 10)
    panel_h = min(panel_h, h - 20)

    overlay = frame.copy()
    cv2.rectangle(overlay,(x0,y0),(x0+panel_w,y0+panel_h),(20,20,20),-1)
    cv2.rectangle(overlay,(x0,y0),(x0+panel_w,y0+panel_h),(80,80,80),1)

    # layout
    cap_h = max(int(panel_h*caption_band), int(70*scale))
    top_h = panel_h - cap_h - 8

    # sparklines
    spark_h = max(22,int(28*scale))
    label_s = 0.44*scale*font_scale
    for i,(lab,key) in enumerate(LABELS):
        y_s = y0 + 8 + i*(spark_h+6)
        if y_s + spark_h > y0 + top_h: break
        sub = overlay[y_s:y_s+spark_h, x0+10:x0+panel_w-10]
        draw_sparkline(sub, idx["t"], idx[key], t)
        put_text_w_shadow(overlay, lab, (x0+12, y_s+spark_h-6), max(0.36, label_s))

    # bars
    bar_top = y0 + min(top_h- (len(LABELS)*(spark_h+6)) - 8, top_h-80)
    bar_top = max(bar_top, y0 + 8 + len(LABELS)*(spark_h+6) + 8)
    for i,(lab,key) in enumerate(LABELS):
        val = interp(idx["t"], idx[key], t)
        bar_x, bar_y = x0+10, bar_top + i*int(28*scale)
        bar_w, bar_h = panel_w-20, max(16, int(20*scale))
        if bar_y+bar_h < y0+top_h-4:
            cv2.rectangle(overlay,(bar_x,bar_y),(bar_x+bar_w,bar_y+bar_h),(55,55,55),-1)
            cv2.rectangle(overlay,(bar_x,bar_y),
                          (bar_x+int(bar_w*max(0,min(1,val))),bar_y+bar_h),
                          (205,205,205),-1)
            put_text_w_shadow(overlay, f"{val:.2f}", (bar_x+bar_w-80, bar_y+bar_h-4),
                              max(0.9, 0.9*scale*font_scale))

    # caption band
    cb_y0 = y0 + panel_h - cap_h
    cv2.rectangle(overlay,(x0,cb_y0),(x0+panel_w,y0+panel_h),(15,15,15),-1)
    cv2.line(overlay,(x0,cb_y0),(x0+panel_w,cb_y0),(90,90,90),1,cv2.LINE_AA)

    if cap_text:
        cap = cap_text.strip()
        chars = int(caption_chars*scale)  # base chars scaled
        lines = textwrap.wrap(cap, width=max(18, chars))[:caption_lines]
        line_h = max(24, int(26*scale))
        y = cb_y0 + line_h + 8
        for ln in lines:
            put_text_w_shadow(overlay, ln, (x0+12, y), max(0.7, 0.7*scale*font_scale))
            y += line_h

    # alpha blend only panel
    roi = frame[y0:y0+panel_h, x0:x0+panel_w].copy()
    roi_overlay = overlay[y0:y0+panel_h, x0:x0+panel_w]
    cv2.addWeighted(roi_overlay, float(alpha), roi, 1.0-float(alpha), 0.0, dst=roi)
    frame[y0:y0+panel_h, x0:x0+panel_w] = roi

# -------- main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--indices", required=True)
    ap.add_argument("--jsonl", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--bottom", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--scale", type=float, default=1.6)
    ap.add_argument("--font-scale", type=float, default=1.0, help="global text multiplier")
    ap.add_argument("--caption-band", type=float, default=0.30, help="portion of panel for caption (0..1)")
    ap.add_argument("--caption-lines", type=int, default=4)
    ap.add_argument("--caption-chars", type=int, default=52)
    args = ap.parse_args()

    idx = load_indices_csv(args.indices)
    cap_times = caps = None
    if args.jsonl and Path(args.jsonl).exists():
        cap_times, caps = load_captions_jsonl(args.jsonl)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"No se puede abrir: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = args.out or (str(Path(args.video).with_name(Path(args.video).stem + "_annotated.mp4")))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        t = i / fps
        caption = nearest_caption(t, cap_times, caps) if cap_times is not None else None
        draw_panel_on_frame(frame, t, idx, caption,
                            topleft=(not args.bottom),
                            alpha=args.alpha,
                            scale=args.scale,
                            font_scale=args.font_scale,
                            caption_band=args.caption_band,
                            caption_lines=args.caption_lines,
                            caption_chars=args.caption_chars)
        writer.write(frame); i += 1

    cap.release(); writer.release()
    print("Wrote:", out_path)

if __name__ == "__main__":
    main()

