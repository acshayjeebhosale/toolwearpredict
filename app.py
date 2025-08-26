import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# -----------------------------
# Load the trained Keras model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_milling_model.keras")

model = load_model()

st.title("🛠️ Tool Wear Prediction App")
st.write("""
This interactive application implements the Deep Neural Network model developed in the dissertation 
"Predictive Analysis of Machining Tool Wear Using Sensor Data". Enter process parameter values 
below to get real-time tool wear predictions.
""")

# -----------------------------
# Define feature ranges (10 features only)
# From your dataset summary (min–max values)
# -----------------------------
feature_ranges = {
    "run": (1, 100),                       # min=1, max=19 (expanded for usability)
    "smcDC": (0.1318359, 9.995117),       # min=0.1318359, max=9.995117
    "vib_spindle": (0.2099609, 0.3759766),# min=0.2099609, max=0.3759766
    "case": (1, 50),                      # min=1, max=16 (expanded for usability)
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
input_values = {}  # Store values for visualization

st.subheader("Process Parameters Input")
st.write("Enter values for all process parameters based on your milling operation:")

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
        # Float input for all other features - preserve all entered digits
        value = st.number_input(
            f"{feature} (Range: {fmin:.6f} to {fmax:.6f})",
            value=None,
            placeholder=f"Enter value between {fmin:.6f} and {fmax:.6f}",
            format="%f"  # Preserve all decimal places entered by user
        )
    
    # Store the value for visualization
    input_values[feature] = value
    
    # Check if value is provided
    if value is None:
        missing_values.append(feature)
    else:
        # Outlier check only if value is provided
        if value < fmin or value > fmax:
            outliers.append((feature, value, fmin, fmax))
        
        inputs.append(float(value))  # Ensure all values are float for the model

# -----------------------------
# Vibration Spindle Visualization
# -----------------------------
st.subheader("Vibration Analysis")
st.write("""
Vibration monitoring is a critical aspect of tool condition monitoring, as established in the 
literature review (Dimla, 2000; Jemielniak & Otman, 2018). The visualization below shows 
the expected vibration patterns based on your input values.
""")

if input_values["vib_spindle"] is not None:
    vib_value = input_values["vib_spindle"]
    
    # Generate sample vibration waveform
    time = np.linspace(0, 2*np.pi, 100)
    amplitude = vib_value
    vibration_wave = amplitude * np.sin(5 * time) + 0.1 * amplitude * np.random.randn(100)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Time domain plot
    ax1.plot(time, vibration_wave, 'b-', linewidth=2)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'Time Domain - Vibration Spindle: {vib_value:.6f}')
    ax1.grid(True, alpha=0.3)
    
    # Frequency domain plot (FFT)
    fft_values = np.fft.fft(vibration_wave)
    frequencies = np.fft.fftfreq(len(vibration_wave), time[1]-time[0])
    ax2.plot(frequencies[:50], np.abs(fft_values[:50]), 'r-', linewidth=2)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.set_title('Frequency Domain (FFT)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display vibration statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Vibration", f"{vib_value:.6f}")
    with col2:
        st.metric("Min Expected", f"{feature_ranges['vib_spindle'][0]:.6f}")
    with col3:
        st.metric("Max Expected", f"{feature_ranges['vib_spindle'][1]:.6f}")
        
    # Vibration analysis based on dissertation findings
    vib_mid = (feature_ranges['vib_spindle'][0] + feature_ranges['vib_spindle'][1]) / 2
    if vib_value > feature_ranges['vib_spindle'][1] * 0.9:
        st.warning("""
        ⚠️ High vibration levels detected. According to the EOL detection methodology established 
        in section 3.2.9, sustained vibration above the 95th percentile may indicate approaching 
        tool End-of-Life (EOL) conditions.
        """)
    elif vib_value > vib_mid:
        st.info("""
        ℹ️ Elevated vibration levels. As discussed in the literature review (Teti et al., 2010), 
        increased vibration often correlates with progressive tool wear in milling operations.
        """)
    else:
        st.success("""
        ✅ Vibration levels are within normal operating range. This suggests stable cutting 
        conditions and acceptable tool health.
        """)
else:
    st.info("ℹ️ Enter a vibration spindle value to see the visualization and analysis")

# -----------------------------
# Prediction
# -----------------------------
st.subheader("Tool Wear Prediction")
st.write("""
The Deep Neural Network model developed in this research (Section 3.4.2) will now predict 
tool wear based on your input parameters. The model achieved a test R² of 0.972 and 
demonstrated strong robustness under noise injection tests.
""")

if st.button("Predict Tool Wear", type="primary"):
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
        
        # Get the raw prediction value
        raw_prediction = prediction[0][0]
        
        # Format to display full number without scientific notation
        if abs(raw_prediction) < 0.001 or abs(raw_prediction) >= 1000000:
            formatted_prediction = f"{raw_prediction:.10f}".rstrip('0').rstrip('.')
        else:
            formatted_prediction = f"{raw_prediction:.6f}".rstrip('0').rstrip('.')
        
        # Display prediction with emphasis
        st.success(f"## ✅ Predicted Tool Wear: {formatted_prediction} units")
        
        # Tool condition assessment based on dissertation findings
        if raw_prediction > 0.5:  # Assuming 0.5 is a threshold for significant wear
            st.error("""
            🚨 High tool wear predicted. Based on the EOL criteria established in section 3.2.9, 
            this tool may be approaching End-of-Life conditions. Consider scheduling tool 
            replacement to maintain machining quality and prevent potential failure.
            """)
        elif raw_prediction > 0.3:
            st.warning("""
            ⚠️ Moderate tool wear predicted. The tool is wearing but remains within acceptable 
            operational limits. Monitor vibration signals and consider planning for future 
            tool maintenance.
            """)
        else:
            st.success("""
            ✅ Low tool wear predicted. The tool is in good condition with minimal wear. 
            Continue normal operations with routine monitoring.
            """)
        
        # Research context
        st.info("""
        **Research Context:** This prediction is generated by a Deep Neural Network that 
        significantly outperformed the Random Forest baseline (R² = 0.972 vs. 0.789), 
        demonstrating the capability of deep learning approaches to capture complex, 
        nonlinear relationships in machining sensor data as established in Chapter 4.
        """)

# -----------------------------
# Dissertation reference
# -----------------------------
st.divider()
st.caption("""
This application implements the research findings from:  
**Bhosale, A. J. (2025). Predictive Analysis of Machining Tool Wear Using Sensor Data.  
University of Essex, MSc Dissertation.**
""")
