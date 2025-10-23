# ML CI/CD (Streamlit, GridSearchCV, GitHub Actions, Docker)
## 📋 Details of the Work

This project is a **production-style template** for predicting **Employee Churn** with a complete, reproducible **ML + MLOps** workflow. It demonstrates how to go from a CSV dataset to a trained, tuned model and a deployable **Streamlit** app using **CI/CD with GitHub Actions** and **Docker** (pushed to **GitHub Container Registry**).

### 🎯 Objectives
- Build a **config-driven** ML pipeline for churn prediction
- Handle **mixed data** (numeric + categorical) with a robust preprocessing pipeline
- Perform **hyperparameter tuning** using **GridSearchCV** with **K-fold cross-validation**
- Save and serve the **best estimator** via a Streamlit app
- Automate **testing (CI)** and **train → build → publish (CD)** using GitHub Actions
- Containerize the app and publish to **GHCR** for easy, portable deployment
- Use a **branching strategy** (`murad` → PR → `main`) to reflect real-world workflows

### 🧠 Dataset
The target is **`Churn`** (0 = stay, 1 = leave).  
Example features (editable in `configs/config.yaml`):
- **Numeric:** `Age`, `Tenure`, `Salary`, `Performance Rating`, `Projects Completed`, `Training Hours`, `Promotions`, `Overtime Hours`, `Satisfaction Level`, `Average Monthly Hours Worked`, `Absenteeism`, `Distance from Home`, `Manager Feedback Score`
- **Categorical:** `Gender`, `Education Level`, `Marital Status`, `Job Role`, `Department`, `Work Location`, `Work-Life Balance`

### 🧩 Methodology
- **Preprocessing:** `ColumnTransformer` → `StandardScaler` (numeric) + `OneHotEncoder` (categorical)
- **Model:** `LogisticRegression`
- **Tuning:** `GridSearchCV` over hyperparameters (e.g., `C`, `solver`)
- **Validation:** `K-fold` cross-validation (configurable via `cv_folds`)
- **Selection:** Persist **best model** to `artifacts/best_model.pkl`

### 🏗️ System Pipeline
1. **Config** (`configs/config.yaml`) defines features, CV, scoring, and grid.
2. **Training** (`src/train.py`) loads data, builds pipeline, runs GridSearchCV, evaluates, saves best model.
3. **App** (`app/app.py`) loads the saved model and serves predictions via Streamlit.
4. **CI** (`.github/workflows/ci.yml`) runs tests on every push/PR.
5. **CD** (`.github/workflows/cd.yml`) trains on push to branches (e.g., `murad`, `main`), builds Docker, pushes to **GHCR**.

```text
CSV → config.yaml → (preprocess + GridSearchCV + CV) → best_model.pkl
                                      │
                                      └─> Streamlit UI (interactive inference)
CI: pytest on push/PR  |  CD: train → docker build → push :latest to GHCR

