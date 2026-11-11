"""Compatibility import: all inference now uses the checked master runner."""
from run_all_ecg import main, load_ischemia_models, predict_ischemia, render_comparison_audit

if __name__ == '__main__':
    raise SystemExit(main())
