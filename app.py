import streamlit as st
import numpy as np
import tensorflow as tf

# -----------------------------
# Load the trained Keras model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_milling_model.keras")

model = load_model()

st.title("🛠️ Tool Wear Prediction App")
st.write("Enter process parameter values to get predictions from the trained model.")

# -----------------------------
# Define feature ranges (10 features only)
# From your dataset summary (min–max values)
# -----------------------------
feature_ranges = {
    "run": (1, 100),                       # min=1, max=19
    "smcDC": (0.1318359, 9.995117),       # min=0.1318359, max=9.995117
    "vib_spindle": (0.2099609, 0.3759766),# min=0.2099609, max=0.3759766
    "case": (1, 50),                      # min=1, max=16
    "AE_table": (0.012207, 0.355835),     # min=0.012207, max=0.355835
    "DOC": (0.75, 1.5),                   # min=0.75, max=1.5
    "AE_spindle": (0.007324, 0.4498291),  # min=0.007324, max=0.4498291
    "vib_table": (0.002441, 1.860352),    # min=0.002441, max=1.860352
    "feed": (0.25, 0.5),                  # min=0.25, max=0.5
    "smcAC": (-4.80957, 4.487305),        # min=-4.80957, max=4.487305
}

# -----------------------------
# Collect user inputs + detect outliers
# -----------------------------
inputs = []
outliers = []
missing_values = []

for feature, (fmin, fmax) in feature_ranges.items():
    if feature == "run":
        # Integer input for "run" feature
        value = st.number_input(
            f"{feature} (Range: {int(fmin)} to {int(fmax)})",
            value=None,
            placeholder=f"Enter integer between {int(fmin)} and {int(fmax)}",
            step=1,  # Integer steps
            format="%d"  # Display as integer
        )
        # Convert to integer if provided
        if value is not None:
            value = int(value)
    else:
        # Float input for all other features - without step to avoid display issues
        value = st.number_input(
            f"{feature} (Range: {fmin:.6f} to {fmax:.6f})",
            value=None,
            placeholder=f"Enter value between {fmin:.6f} and {fmax:.6f}",
            # step=0.01  # Removed to prevent display rounding issues
        )
    
    # Check if value is provided
    if value is None:
        missing_values.append(feature)
    else:
        # Outlier check only if value is provided
        if value < fmin or value > fmax:
            outliers.append((feature, value, fmin, fmax))
        
        inputs.append(float(value))  # Ensure all values are float for the model

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    # Check if all values are provided
    if missing_values:
        st.error(f"❌ Please provide values for: {', '.join(missing_values)}")
    else:
        # Convert to numpy array
        input_data = np.array([inputs], dtype=np.float32)
        
        if outliers:
            for feat, val, fmin, fmax in outliers:
                st.warning(
                    f"⚠️ {feat} = {val} is **outside the expected range** "
                    f"({fmin} to {fmax}). This feature may be an **outlier** "
                    f"and should be examined or monitored again."
                )
        
        # Run prediction
        prediction = model.predict(input_data)
        st.success(f"✅ Predicted Tool Wear: {prediction[0][0]:.4f} units")
        st.info("ℹ️ Note: Predictions with outlier inputs may be unreliable.")
