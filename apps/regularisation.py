import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# Create a dataset in 1d
def f(x):
    return x * np.sin(x)

# Create a dataset
x_plot = np.linspace(0, 10, 100)
x = np.linspace(0, 10, 30)
y = f(x) + np.random.randn(30) * 1.5

# Create a streamlit app
st.title("Regression Model")
st.write(
    "This app shows the fit of a regression model with varying degrees and varying penalty."
)

# The sidebar contains the sliders and the regression type dropdown
with st.sidebar:
    # Create a slider for degree
    degree = st.slider("Degree", 1, 15, 4)

    # Create a slider for alpha (common parameter for both Ridge and Lasso)
    alpha = st.slider("Alpha", 0.0, 10.0, 1.0)

    # Create a dropdown for regression type
    regression_type = st.selectbox("Regression Type", ["Ridge", "Lasso"])

# Create a model based on selected regression type
if regression_type == "Ridge":
    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=alpha))
else:
    model = make_pipeline(PolynomialFeatures(degree), Lasso(alpha=alpha))

# Fit the model
model.fit(x[:, np.newaxis], y)

# Predict
y_plot = model.predict(x_plot[:, np.newaxis])

# Plot the data
fig, ax = plt.subplots()

ax.scatter(x, y, label="Data")
ax.plot(x_plot, y_plot, label="Predicted", color="r")
ax.plot(x_plot, f(x_plot), label="True", color="k", linestyle="--")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_ylim(-10, 20)
fig.legend()
st.pyplot(fig)

# Print the model: y = a + b*x + c*x^2 + ...
st.write("The model is:")
intercept = model.steps[1][1].intercept_
coefficients = model.steps[1][1].coef_
st.write(
    f"y = {intercept:.2f}"
    + "".join([f" + {c:.2f}x^{i}" for i, c in enumerate(coefficients, start=1)])
)

# Line between the model and the plot
st.write("---")

st.write("Magnitude of largest coefficient:")
st.write(f"{np.max(np.abs(coefficients)):.2f}")
