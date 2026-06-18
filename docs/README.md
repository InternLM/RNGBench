# Project Homepage

Source for the RNG-Bench project page, served via GitHub Pages.

## Enable GitHub Pages

Settings → **Pages** → set source to `Branch: main / docs/`. The page will be served at
`https://<org>.github.io/<repo>/`.

## Files

```
docs/
├── index.html             # Single-page site
└── static/
    ├── css/main.css       # All styling (no JS dependencies)
    └── images/            # All figures referenced by index.html
        ├── two_games.png  # Hero — Matching Pairs + 3D Maze side-by-side
        ├── memory_gap.png # External-memory ablation bar chart
        ├── scale_sweep.png# Grid-size sweep
        ├── case_dual.png  # Gemini vs GPT duel trajectories
        ├── case_single.png# (unused — single-player matched-pair curves)
        └── gameplay/      # Per-round Matching Pairs frames
            ├── mp_start.jpg
            ├── mp_flip1.jpg
            ├── mp_both.jpg
            └── mp_truth.jpg
```

Figures were exported with:

```bash
pdftocairo -png -r 200 -singlefile <paper.pdf> docs/static/images/<name>
```

## Update before going public

Search `index.html` for `href="#"` (rendered as the greyed-out disabled buttons) and replace with live URLs:

| Button       | Where to put the link             |
|--------------|-----------------------------------|
| Paper        | Camera-ready PDF                  |
| arXiv        | `https://arxiv.org/abs/...`       |
| Code         | GitHub repo URL                   |
| Dataset      | HuggingFace dataset URL           |
| SFT Checkpoint | HuggingFace model URL           |

Once a button has a real URL, remove the `is-disabled` class so it becomes clickable.

Also replace `Anonymous Authors` and the BibTeX block when de-anonymized.
