# Static Dashboard Deployment to GitHub Pages

Your Streamlit dashboard has been converted to **static HTML** for automatic deployment to GitHub Pages.

## How It Works

1. **Generate Static Dashboard**
   ```bash
   make dashboard
   ```
   Outputs: `dist/index.html` with embedded CSS, charts, and reports

2. **Include Test & Coverage Reports**
   - Test results from `reports/test-results/`
   - Coverage reports from `reports/coverage/html/`
   - Automatically bundled into `dist/`

3. **Deploy to GitHub Pages**
   - Automatic on every push to `Abdelrahman` branch
   - GitHub Actions builds → deploys `dist/` → live at GitHub Pages URL

## GitHub Pages URL

Once deployed, your dashboard is live at:
```
https://{github-username}.github.io/Chess-Elo-Predictor/
```

## Features

✅ **Dark theme** — Modern black and gold design  
✅ **Responsive** — Works on mobile, tablet, desktop  
✅ **Fast** — Pure HTML/CSS, no server required  
✅ **Linked Reports** — Test results and coverage accessible  
✅ **Auto-updates** — Redeploys on every push  

## Local Testing

Generate and preview dashboard locally:

```bash
# Generate static HTML
make dashboard

# Open in browser
# Windows:
start dist\index.html

# Mac:
open dist/index.html

# Linux:
xdg-open dist/index.html
```

## CI/CD Flow

```
Push to GitHub
    ↓
GitHub Actions:
  1. Run tests + coverage
  2. Generate test reports
  3. Generate static dashboard
  4. Copy reports to dist/
  5. Deploy dist/ to GitHub Pages
    ↓
Live Dashboard: https://...
```

## Dashboard Sections

### 1. Pipeline Status
- Shows completion status
- Last updated timestamp

### 2. Dataset Overview
- Game count (from merged_games.csv if available)
- Feature count
- Dataset composition

### 3. Model Performance
- Linked model reports (if generated)
- Performance metrics per model
- Track A vs Track B comparison

### 4. Integration Tests
- 28 integration tests coverage
- Links to:
  - Coverage report: `coverage/html/index.html`
  - Test results: `test-results/test-report.html`

### 5. Resources
- GitHub repository link
- Documentation links
- Pipeline phase count

## Customization

Edit `scripts/generate_static_dashboard.py` to:
- Change colors/theme (CSS in `generate_html()`)
- Add new sections
- Load additional data files
- Customize metrics displayed

## Troubleshooting

**Dashboard not updating?**
- Check GitHub Actions logs
- Verify `dist/` is being generated
- Confirm GitHub Pages is enabled in repo settings

**Missing reports?**
- Run `make test-report` first
- Ensure `reports/` directory exists
- Check file permissions

**Want to add interactivity?**
- Consider Plotly charts (embedded in HTML)
- Or deploy to Streamlit Cloud instead
- Or use GitHub Pages with Jekyll

---

## Previous Attempts

- ❌ npm stlite: Not available on npm registry
- ❌ pip stlite-cli: Not available on PyPI
- ✅ Static HTML: Works perfectly with GitHub Pages!

