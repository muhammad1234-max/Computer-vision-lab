# Task 3: Real-Time Echocardiogram Video Analysis

## Overview
Ultrasound (echocardiogram) footage of a beating heart is notoriously noisy, murky,
and low-contrast. This task builds a real-time OpenCV pipeline that reads each frame
from an .mp4 file, applies a full enhancement pipeline, and displays the raw feed
alongside the corrected feed live.

## How to Run the Video Loop

### Option A: Interactive (recommended — requires a display)
```bash
cd Task_3_Echo_Analysis
jupyter notebook realtime_echo.ipynb
```
Run the **"Real-Time Processing Loop"** cell. A window will appear showing
the raw ultrasound on the left and the enhanced stream on the right.
**Press Q to quit.**

### Option B: Static Preview (no display needed)
Run the **"Static Frame-by-Frame Preview"** cell instead.
It processes 4 evenly spaced frames from the video and shows them via matplotlib,
saving the result to `output/static_preview_frames.png`.

## Per-Frame Enhancement Pipeline

Each frame goes through these transformations in sequence:

| Step | Operation | Why |
|------|-----------|-----|
| 1 | Grayscale conversion | Ultrasound is single-channel; reduces noise in color channels |
| 2 | Histogram Equalization | Combats the murky, low-contrast look typical of ultrasound |
| 3 | COLORMAP_JET | Maps intensity to color spectrum; highlights blood flow intensities |
| 4 | Color Balance | Gray-world correction neutralizes artificial cast from the colormap |
| 5 | Logarithmic Transform | Reveals the darkest regions (heart chamber interiors) |
| 6 | Power-Law Gamma (γ=0.6) | Suppresses blinding white backscatter noise from the ultrasound probe |

## Monitoring Array Layout
```
+-----------------------+-----------------------+
|                       |                       |
|    RAW ULTRASOUND     |  ENHANCED PIPELINE    |
|   (original feed)     |   (all steps applied) |
|                       |                       |
+-----------------------+-----------------------+
```

## Dataset
- Source: Stanford EchoNet-Dynamic Dataset (Kaggle)
- Link: https://www.kaggle.com/datasets/manojkumarcs28/echonet-dynamic-by-stanford-university
- Place one short ultrasound clip as `data/sample_echo.mp4`

## Outputs (saved to `output/`)
- `frame_sample_NNN.png`              – Screenshots of the monitoring array (every 30 frames)
- `pipeline_screenshot_example.png`   – First monitoring array screenshot
- `static_preview_frames.png`         – Static 4-frame preview (no display needed)
