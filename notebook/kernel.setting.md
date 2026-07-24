# Jupyter Notebook Kernel Setup Guide (uv)

This guide configures Jupyter to use the project's `uv`-managed virtual environment as a kernel.

## Prerequisites

- `uv` installed ([uv documentation](https://docs.astral.sh/uv/))
- The project's `pyproject.toml` and `uv.lock` in the repository root
- Jupyter installed (globally or in a base environment)

## Quick Setup (macOS / Linux)

### Step 1: Sync project dependencies

From the repository root (`/Users/yeon/work/uob/uob-final-dissertation-allianz-internship-2026/`):

```bash
uv sync
```

This installs all dependencies defined in `pyproject.toml` into the project's `.venv` directory.

### Step 2: Install ipykernel into the uv environment

```bash
uv pip install ipykernel
```

### Step 3: Register the kernel with Jupyter

```bash
uv run python3 -m ipykernel install --user --name sfp-detection --display-name "Python 3.11 (sfp-detection)"
```

**Note:** Change `sfp-detection` to any kernel name you prefer. The `--display-name` is what appears in the Jupyter UI.

### Step 4: Verify the kernel is registered

```bash
jupyter kernelspec list
```

You should see `sfp-detection` in the output list.

---

## Quick Setup (Windows)

### Step 1: Sync project dependencies

Open **PowerShell** or **Command Prompt** and navigate to the repository root:

```powershell
cd C:\path\to\uob-final-dissertation-allianz-internship-2026
uv sync
```

This installs all dependencies defined in `pyproject.toml` into the project's `.venv` directory.

### Step 2: Install ipykernel into the uv environment

```powershell
uv pip install ipykernel
```

### Step 3: Register the kernel with Jupyter

```powershell
uv run python -m ipykernel install --user --name sfp-detection --display-name "Python 3.11 (sfp-detection)"
```

**Note:** On Windows, use `python` instead of `python3`. Change `sfp-detection` to any kernel name you prefer.

### Step 4: Verify the kernel is registered

```powershell
jupyter kernelspec list
```

You should see `sfp-detection` in the output list.

## Using the Kernel in Jupyter

### Jupyter Lab or Notebook

1. Start Jupyter:
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```

2. Open or create a `.ipynb` file.

3. Click the **Kernel** button (top right) → **Select another kernel** → choose `Python 3.11 (sfp-detection)`.

4. Verify the kernel is active:
   - Cell output shows `Python 3.11.x [...]` at startup
   - `import sys; print(sys.executable)` points to `.venv` in your project root

### VS Code Jupyter Extension

1. Open a `.ipynb` file in VS Code.
2. Click the **kernel picker** (top right).
3. Select **Python 3.11 (sfp-detection)**.
4. Verify by running a cell with `import sys; print(sys.executable)`.

## Troubleshooting

### Kernel not appearing in Jupyter

**Check kernel installation (macOS/Linux):**
```bash
jupyter kernelspec list
```

**Check kernel installation (Windows):**
```powershell
jupyter kernelspec list
```

If missing, re-run Step 3:

**macOS/Linux:**
```bash
uv run python3 -m ipykernel install --user --name sfp-detection --display-name "Python 3.11 (sfp-detection)"
```

**Windows:**
```powershell
uv run python -m ipykernel install --user --name sfp-detection --display-name "Python 3.11 (sfp-detection)"
```

### "ModuleNotFoundError" when running cells

**Ensure dependencies are synced (all platforms):**
```bash
uv sync
```

If still broken, try reinstalling:

**macOS/Linux:**
```bash
uv pip install --force-reinstall ipykernel
uv run python3 -m ipykernel install --user --name sfp-detection --force --display-name "Python 3.11 (sfp-detection)"
```

**Windows:**
```powershell
uv pip install --force-reinstall ipykernel
uv run python -m ipykernel install --user --name sfp-detection --force --display-name "Python 3.11 (sfp-detection)"
```

### Using uv run within notebooks (alternative)

If kernel registration fails, you can run notebooks via:

**macOS/Linux:**
```bash
uv run jupyter lab
```

**Windows:**
```powershell
uv run jupyter lab
```

This ensures Jupyter itself runs within the uv environment, automatically using the correct kernel.

## Common Workflow

### macOS / Linux

After cloning the repository on a new machine:

```bash
# 1. Navigate to project root
cd /path/to/uob-final-dissertation-allianz-internship-2026

# 2. Sync dependencies
uv sync

# 3. Install and register kernel
uv pip install ipykernel
uv run python3 -m ipykernel install --user --name sfp-detection --display-name "Python 3.11 (sfp-detection)"

# 4. Start Jupyter
jupyter lab

# 5. Open a notebook and select "Python 3.11 (sfp-detection)" kernel
```

### Windows (PowerShell)

After cloning the repository on a new machine:

```powershell
# 1. Navigate to project root
cd C:\path\to\uob-final-dissertation-allianz-internship-2026

# 2. Sync dependencies
uv sync

# 3. Install and register kernel
uv pip install ipykernel
uv run python -m ipykernel install --user --name sfp-detection --display-name "Python 3.11 (sfp-detection)"

# 4. Start Jupyter
jupyter lab

# 5. Open a notebook and select "Python 3.11 (sfp-detection)" kernel
```

## Removing the kernel

If you need to unregister the kernel:

**macOS/Linux:**
```bash
jupyter kernelspec remove sfp-detection
```

**Windows:**
```powershell
jupyter kernelspec remove sfp-detection
```

## Additional Notes

- **Environment activation:** Unlike conda, `uv` does not require activating the environment explicitly. The kernel automatically uses the `.venv` directory.
- **Reproducibility:** Always commit `pyproject.toml` and `uv.lock` to version control. Running `uv sync` on a new machine ensures identical dependencies.
- **Fallback (conda):** If uv setup fails, the conda environment `sfp-detection` (see main README) can be used, but `uv` is the standard for this project.
