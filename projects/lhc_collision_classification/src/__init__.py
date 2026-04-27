"""LHC proton-proton collision image classification.

Supervised classification of simulated 13 TeV proton-proton collision
events from the CERN Open Data project. The three classes are:

    QCD:   Quantum Chromodynamics multi-jet background.
    TTbar: Top-antitop pair production.
    WJets: W boson + jets.

Submodules:
    data_loader:    Walk a class-foldered image dataset and stream files.
    preprocessing:  Resize / normalise images and extract HOG descriptors.
    models:         Build classical ML pipelines (RF, SVM, XGBoost) and a
                    light Keras MLP head for HOG features.
    evaluation:     Cross-validated benchmark + comparison table helpers.
    train:          End-to-end CLI entry point.
"""
