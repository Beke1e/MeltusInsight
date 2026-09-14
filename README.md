# Meltus Insight

Meltus Insight is a Flask-based diabetes screening and patient-management application developed in the University of Gondar research context.

## Features

- Secure login, registration, sessions, password hashing, and role-based access
- Patient registration with automatically generated Patient IDs and registration dates
- Patient CRUD operations with view, edit, search, and administrator-protected delete actions
- Diabetes prediction using the supplied trained TensorFlow/Keras model
- Patient-linked prediction records with create, view, edit, delete, and export actions
- Prediction filtering by Patient ID, outcome, and date range
- Prediction-only CSV exports containing model inputs, outcome, Patient ID, and timestamps
- Responsive dashboard, sidebar navigation, header actions, and mobile layout
- About Us page with project and development-team information

## Project Structure

```text
app.py                 Flask application
train_model.py         Model training script
model.h5               Trained Keras model
scaler.pkl             Feature scaling artifact
static/style.css       Shared application stylesheet
templates/             Jinja2 application templates
diabetes.db            Local SQLite database, created at runtime
```

## Requirements

Use Python 3.10 or a compatible Python environment. Install the application dependencies:

```powershell
python -m pip install flask numpy pandas scikit-learn tensorflow-cpu openpyxl
```

The Excel import feature requires `openpyxl` for `.xlsx` files.

## Run Locally

From the project directory:

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:5000/login
```

Create an account at `/register`. The first registered account receives administrator access.

## Configuration

Set a strong secret key before deployment:

```powershell
$env:SECRET_KEY = "your-long-random-secret-key"
```

The application creates a local `.secret_key` file when `SECRET_KEY` is not provided. This file is ignored by Git.

For production, run behind a production WSGI server and HTTPS. Do not use Flask's development server for public deployment.

## Prediction Workflow

1. Register or sign in.
2. Create a patient record from **Patients**.
3. Open **New prediction** and select the Patient ID.
4. Enter the nine encoded model inputs.
5. Save the prediction and review the result page.
6. Use **History** to view, edit, delete, filter, or export prediction records.

Prediction exports intentionally contain only prediction-related values and do not export general patient records such as name, phone, residence, or education.

## Model Information

The application uses the supplied diabetes mellitus model trained with:

- Multilayer perceptron dense layers
- Dropout regularization
- Attention-inspired feature learning architecture
- Adam optimizer with learning rate `0.01`
- Batch size `32`
- `100` training epochs
- Binary outcome evaluation using accuracy

The dataset context is Debark General Hospital. The described preprocessing includes missing-value handling, outlier removal, categorical encoding, normalization, class-balancing augmentation, feature selection, and dimensionality reduction.

## Development Team

**Bekele Mulat**  
University of Gondar  
Email: [bekele.mulat@uog.edu.et](mailto:bekele.mulat@uog.edu.et)  
Phone: 0936328953

**Ejargew Abay**  
University of Gondar  
Email: [Ejargew.Abay@uog.edu.et](mailto:Ejargew.Abay@uog.edu.et)  
Phone: 0939698582

## Data and Security Notes

- `diabetes.db`, `.secret_key`, runtime CSV files, virtual environments, and Python caches are excluded from Git.
- Patient and prediction data are stored locally in SQLite.
- Configure a strong `SECRET_KEY` and use HTTPS in deployment.
- Review institutional data-governance requirements before using real patient information.
