# Predictive-Maintenance-PyTorch — Interactive Documentation Site

Generated documentation site for https://github.com/shravanvinayhegde/Predictive-Maintenance-PyTorch

## What this is

A static Astro site reviewing `NASA_pytorch.ipynb` — a PyTorch pipeline predicting Remaining Useful
Life (RUL) for turbofan engines on NASA's C-MAPSS FD001 dataset. Every code snippet, hyperparameter,
and claim on this site is pulled directly from that notebook (and cross-checked against the repo's
README) — nothing is fabricated. The Results page intentionally shows "Not reported in repository"
rather than invented numbers, since the notebook ships unexecuted.

## Local development

```bash
npm install
npm run dev       # http://localhost:4321/Predictive-Maintenance-PyTorch/
npm run build     # outputs to dist/
npm run preview   # preview the production build
```

## Deploying to GitHub Pages

This is pre-configured for GitHub Pages at the repo's default Pages URL
(`https://shravanvinayhegde.github.io/Predictive-Maintenance-PyTorch/`), via `astro.config.mjs`
(`site` + `base`).

This project ships with `.github/workflows/deploy-docs.yml`, which builds this Astro project and
deploys it to GitHub Pages automatically on every push to `main` that touches `docs-site/`. To use it:

1. Place this entire folder in your repo as `docs-site/` (so the workflow's `path: docs-site` matches).
2. Commit and push `docs-site/` (including `.github/workflows/deploy-docs.yml` — move that file to the
   repo's top-level `.github/workflows/` directory, since GitHub only reads workflows from there).
3. In your repo on GitHub: Settings → Pages → Build and deployment → Source → **GitHub Actions**.
4. Push to `main`. Check the Actions tab for the run; once it's green, the site is live.

**Manual alternative (no Actions):** run `npm run build` locally, then push the contents of `dist/` to
a `gh-pages` branch (or a `/docs` folder on `main`) and point GitHub Pages at that instead.

## Structure

- `src/data/content.ts` — single source of truth for all facts (hyperparameters, code snippets,
  model specs, pipeline stages, file tree). Sourced from the notebook's actual cells.
- `src/pages/` — one route per section (home, map, pipeline, architecture, training, results, structure)
- `src/components/` — Sidebar, HUD strip, CommandPalette (Cmd/Ctrl+K), CodeBlock (syntax highlighting
  + line numbers, no external highlighter dependency)
- `public/search-index.json` — static index powering the command palette
