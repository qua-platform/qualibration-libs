# Code Cleanup TODO List

## Priority 1: Remove Dead Code & Code Smells (Immediate Impact)

- [x] **Remove unused overlay abstraction** (✅ 812 lines removed!)
  - [x] Delete overlays.py (507 lines)
  - [x] Delete overlay_backends.py (305 lines)
  - [x] Remove references from **init**.py (none found)
- [x] **Replace all hardcoded values** (✅ Completed)
  - [x] Fix matplotlib tight_layout magic numbers (0.03, 0.95)
  - [x] Replace hardcoded coordinate strings ("power", "detuning", "qubit")
  - [x] Use UnitConverter for all conversions (1e-3, 1e9)
- [x] **Remove duplicate/unused imports** (✅ 15 imports removed)
- [x] **Eliminate commented-out code** (✅ None found)

- test Run the full test suite to check for regressions:
  `bash cd /Users/shanto/Docs/internship/quantum-elements/qua-libs/qualibration_graphs/superconducting p3 test_02a_unified_plotting.py & p3 test_02b_unified_plotting.py & p3 test_02c_unified_plotting.py & p3 test_04b_unified_plotting.py & wait `
- if the tests pass, then add the \*.py changes and commit the changes with commit message "<work type> detailed summary of changes"

## Priority 2: Consolidate Duplicate Code (Major Size Reduction)

Use parallel subagents to work on distinct files at the same time.

- [ ] **Merge duplicate overlay methods**
  - [ ] Consolidate \_add_overlays variants into base class
  - [ ] Remove engine-specific overlay implementations
- [ ] **Extract common data extraction patterns**
  - [ ] Use DataExtractor consistently
  - [ ] Remove manual qubit extraction code
- [ ] **Unify subplot/grid creation**

  - [ ] Create single grid setup method in base class
  - [ ] Remove duplicate grid logic from engines

- test Run the full test suite to check for regressions:
  `bash cd /Users/shanto/Docs/internship/quantum-elements/qua-libs/qualibration_graphs/superconducting p3 test_02a_unified_plotting.py & p3 test_02b_unified_plotting.py & p3 test_02c_unified_plotting.py & p3 test_04b_unified_plotting.py & wait `
- if the tests pass, then add the \*.py changes and commit the changes with commit message "<work type> detailed summary of changes"

## Priority 3: Decompose Long Methods (Maintainability)

Use parallel subagents to work on distinct files at the same time.

- [ ] **plotly_engine.py methods**
  - [ ] Break down \_add_heatmap_trace (76 lines)
  - [ ] Split create_spectroscopy_figure (62 lines)
  - [ ] Refactor \_create_generic_figure (57 lines)
- [ ] **matplotlib_engine.py methods**

  - [ ] Decompose \_add_spectroscopy_traces (49 lines)
  - [ ] Split \_add_generic_trace (49 lines)

- test Run the full test suite to check for regressions:
  `bash cd /Users/shanto/Docs/internship/quantum-elements/qua-libs/qualibration_graphs/superconducting p3 test_02a_unified_plotting.py & p3 test_02b_unified_plotting.py & p3 test_02c_unified_plotting.py & p3 test_04b_unified_plotting.py & wait `
- if the tests pass, then add the \*.py changes and commit the changes with commit message "<work type> detailed summary of changes"

## Priority 4: Maximize Abstraction Usage

Use parallel subagents to work on distinct files at the same time.

- [ ] **Use ExperimentDetector everywhere**
  - [ ] Replace all \_is_flux_spectroscopy checks
  - [ ] Replace all \_is_power_rabi checks
- [ ] **Leverage existing utilities**

  - [ ] Use DataValidator for all validation
  - [ ] Apply RobustStatistics consistently

- test Run the full test suite to check for regressions:
  `bash cd /Users/shanto/Docs/internship/quantum-elements/qua-libs/qualibration_graphs/superconducting p3 test_02a_unified_plotting.py & p3 test_02b_unified_plotting.py & p3 test_02c_unified_plotting.py & p3 test_04b_unified_plotting.py & wait `
- if the tests pass, then add the \*.py changes and commit the changes with commit message "<work type> detailed summary of changes"

## Estimated Impact

- Dead code removal: ✅ 812 lines already removed (exceeded estimate!)
- Duplication removal: ~300-400 lines (pending)
- Method decomposition: Code stays same size but more maintainable
- **Total reduction: ~1100-1200 lines (30-35% of module)**
