# Panel Migration Guide

## Overview

This document describes the Panel-based implementation of the ALS beamline script generation widgets, which replicates the functionality of the original ipywidgets-based implementation.

## Files Created

### 1. Panel_BaseWidgets.py

Contains Panel equivalents of the base widget classes:

- **ALS_ScriptGenWidget**: Top-level script generator base class
- **ALS_ExperimentWidget**: Experiment-level widget base class
- **ALS_MeasurementWidget**: Measurement-level widget base class
- **Utility functions**: `clean_script()`, `pad_digits()`

### 2. Panel_PXR_Widgets.py

Contains PXR-specific implementations:

- **PXR_ScriptGen**: Main script generator for PXR experiments
- **PXR_Experiment**: Individual sample/experiment widget
- **PXR_Scan**: Individual measurement scan widget

### 3. panel.ipynb

Demonstration notebook showing:

- Basic Panel widgets and capabilities
- Complete PXR script generator implementation
- Usage examples and comparisons

## Architecture

### Three-Level Hierarchy

```
PXR_ScriptGen (Top Level)
├── Save/Load JSON configuration
├── Add/Remove/Duplicate samples
│
├── PXR_Experiment (Sample 1)
│   ├── Motor positions (X, Y, Z)
│   ├── Optional offsets
│   ├── Advanced options (Accordion)
│   │
│   ├── PXR_Scan (Measurement 1)
│   │   ├── Energy range parameters
│   │   ├── Dynamic motor table
│   │   └── Add/Remove steps/motors
│   │
│   └── PXR_Scan (Measurement 2)
│       └── ...
│
└── PXR_Experiment (Sample 2)
    └── ...
```

## Key Differences from ipywidgets Version

### Widget Replacements

| ipywidgets            | Panel Equivalent          |
| --------------------- | ------------------------- |
| `widgets.Text()`      | `pn.widgets.TextInput()`  |
| `widgets.FloatText()` | `pn.widgets.FloatInput()` |
| `widgets.Button()`    | `pn.widgets.Button()`     |
| `widgets.Dropdown()`  | `pn.widgets.Select()`     |
| `widgets.Checkbox()`  | `pn.widgets.Checkbox()`   |
| `widgets.VBox()`      | `pn.Column()`             |
| `widgets.HBox()`      | `pn.Row()`                |
| `widgets.Tab()`       | `pn.Tabs()`               |
| `widgets.Accordion()` | `pn.Accordion()`          |
| `widgets.HTML()`      | `pn.pane.Markdown()`      |

### Layout Management

**ipywidgets**:

```python
widgets.VBox([
    widgets.HBox([w1, w2]),
    w3
])
```

**Panel**:

```python
pn.Column(
    pn.Row(w1, w2),
    w3
)
```

### Event Handling

**ipywidgets**:

```python
button = widgets.Button(description="Click")
button.on_click(callback_function)
```

**Panel**:

```python
button = pn.widgets.Button(name="Click")
button.on_click(callback_function)
```

### Tabs

**ipywidgets**:

```python
tab = widgets.Tab(children=[widget1, widget2])
tab.set_title(0, "Title 1")
```

**Panel**:

```python
tab = pn.Tabs(
    ("Title 1", widget1),
    ("Title 2", widget2)
)
```

## Usage

### In Jupyter Notebook

```python
import panel as pn
from Panel_PXR_Widgets import PXR_ScriptGen

# Initialize Panel
pn.extension('tabulator')

# Create script generator
generator = PXR_ScriptGen(path="/path/to/save")

# Display
generator.GUI
```

### As Standalone Web App

```bash
# Serve the notebook
panel serve panel.ipynb --show

# Or create a standalone script
panel serve my_app.py --show
```

### Programmatic Access

```python
# Access configuration
config = generator.save_as_dict()

# Get first experiment
exp = generator.exp_0001

# Modify parameters
exp.XPosition.value = 10.0
exp.name_of_sample.value = "MySample"

# Get first scan
scan = exp.scan_0001
scan.energy_start.value = 270.0

# Save to JSON
generator.save_name.value = "my_config"
generator.save_json()

# Load from JSON
generator.load_json()
```

## Features Implemented

### ✅ Complete Feature Parity

- [x] Three-level widget hierarchy
- [x] Dynamic table editing (add/remove rows and columns)
- [x] Nested tabs for samples and measurements
- [x] Accordion menus for parameter organization
- [x] JSON save/load configuration
- [x] Script export to CSV/TXT
- [x] Parameter validation
- [x] Widget state management
- [x] Add/duplicate/delete functionality at all levels

### ⚠️ Modified Features

- **File Browser**: Uses text input instead of PyQt6 dialogs
  - Simpler implementation
  - Can be enhanced with Panel's file input widgets if needed

### 🚀 Enhanced Features

- **Better Table Widget**: Panel's Tabulator is more powerful
- **Responsive Layouts**: Auto-adjusts to screen size
- **Modern Styling**: Cleaner default appearance
- **Web Deployment**: Can run as standalone web app
- **Reactive Updates**: Easier with `@pn.depends` decorator

## Migration Checklist

If migrating from ipywidgets to Panel:

1. **Replace imports**:

   - `from BaseWidgets import ...` → `from Panel_BaseWidgets import ...`
   - `from PXR_Widgets import ...` → `from Panel_PXR_Widgets import ...`

2. **Initialize Panel extension**:

   ```python
   pn.extension('tabulator')
   ```

3. **Update widget access**:

   - Tabs: `tab.selected_index` → `tab.active`
   - Widget children: `widget.children` → `widget.objects`

4. **File operations**:

   - Replace PyQt6 file dialogs with text input or custom implementation

5. **Display**:
   - ipywidgets automatically displays in Jupyter
   - Panel requires explicit display or `.servable()`

## Testing

### Verify Core Functionality

```python
# Test 1: Create generator
gen = PXR_ScriptGen()
assert hasattr(gen, 'GUI')

# Test 2: Add experiment
gen.new_tab(gen.layout, PXR_Experiment, PXR_Experiment.ALS_NAME)
assert len(gen.layout) >= 1

# Test 3: Access experiment
exp = gen.exp_0001
assert exp is not None

# Test 4: Modify parameters
exp.XPosition.value = 5.0
assert exp.XPosition.value == 5.0

# Test 5: Save configuration
config = gen.save_as_dict()
assert 'exp_0001' in config
```

## Deployment Options

### 1. Jupyter Notebook

- Run cells directly
- Interactive widgets in notebook

### 2. JupyterLab

- Better extension support
- Can dock panels

### 3. Standalone Web App

```bash
panel serve panel.ipynb --show --port 5006
```

### 4. Embedded in Web Server

```python
import panel as pn
from Panel_PXR_Widgets import PXR_ScriptGen

pn.extension('tabulator')

def create_app():
    gen = PXR_ScriptGen()
    return gen.GUI

pn.serve({'/': create_app}, port=5006, show=True)
```

### 5. Docker Deployment

```dockerfile
FROM python:3.11

RUN pip install panel pandas numpy

COPY Panel_BaseWidgets.py .
COPY Panel_PXR_Widgets.py .
COPY app.py .

CMD ["panel", "serve", "app.py", "--address", "0.0.0.0", "--port", "5006"]
```

## Performance Considerations

- **Startup**: Panel apps may take slightly longer to initialize
- **Reactivity**: Panel's reactive framework is more efficient for complex updates
- **Memory**: Similar memory footprint to ipywidgets
- **Rendering**: Panel uses Bokeh for rendering, which handles large tables well

## Troubleshooting

### Common Issues

1. **Tabs not updating**:

   ```python
   # Use clear() and append()
   layout.clear()
   layout.append(("Title", widget))
   ```

2. **Widgets not displaying**:

   ```python
   # Ensure Panel extension is initialized
   pn.extension('tabulator')
   ```

3. **JSON load fails**:

   - Check file path is correct
   - Verify JSON structure matches expected format

4. **Table editing not working**:
   - Ensure `editors` parameter is set on Tabulator
   - Check that columns are not set to read-only (`None`)

## Future Enhancements

Potential improvements:

1. **File Browser**: Implement native Panel file selector
2. **Visualization**: Add live plots of scan trajectories
3. **Validation**: Real-time parameter validation with visual feedback
4. **Templates**: Pre-configured experiment templates
5. **Export Formats**: Support for additional export formats
6. **Database Integration**: Save configurations to database
7. **User Authentication**: Multi-user support for web deployment
8. **Undo/Redo**: History tracking for parameter changes

## Support

For questions or issues:

- Panel documentation: https://panel.holoviz.org
- Panel GitHub: https://github.com/holoviz/panel
- HoloViz Discourse: https://discourse.holoviz.org

## License

Same license as original ipywidgets implementation.
