import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(0)
# Create a dataset in 1d
def f(x):
    return x * np.sin(x) + np.cos(4*x)

n_points = 50

# Create a dataset
x_plot = np.linspace(0, 10, 100)
x = np.linspace(0, 10, n_points)
y = f(x) + np.random.randn(n_points) * 1.0
m_y = np.mean(y)
s_y = np.std(y)
y = (y - m_y) / s_y

f_true = (f(x_plot) - m_y) / s_y
# Create a streamlit app
st.title("Regression Model")
st.write(
    "This app shows the fit of a regression model with varying degrees and varying penalty."
)

# The sidebar contains the sliders and the regression type dropdown
with st.sidebar:
    # Create a slider for degree
    degree = st.slider("Degree", 1, 15, 1)

    # Create a slider for alpha (common parameter for both Ridge and Lasso)
    alpha = st.slider("Alpha", 0.0, 1000.0, 0.0)

    # Create a dropdown for regression type
    regression_type = st.selectbox("Regression Type", ["Ridge", "Lasso"])

# Create a model based on selected regression type
if regression_type == "Ridge":
    model = make_pipeline(PolynomialFeatures(degree, include_bias=False), Ridge(alpha=alpha))
else:
    model = make_pipeline(PolynomialFeatures(degree, include_bias=False), Lasso(alpha=alpha))

# Fit the model
model.fit(x[:, np.newaxis], y)

# Predict
y_plot = model.predict(x_plot[:, np.newaxis])

# Plot the data
fig, ax = plt.subplots()

ax.scatter(x, y, label="Data")
ax.plot(x_plot, y_plot, label="Predicted", color="r")
ax.plot(x_plot, f_true, label="True", color="k", linestyle="--")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_ylim(-3, 3)
fig.legend()
st.pyplot(fig)

# Print the model: y = a + b*x + c*x^2 + ...
st.write("The model is:")
intercept = model.steps[1][1].intercept_
coefficients = model.steps[1][1].coef_
latex_eq = (
    f"y = {intercept:.10f}"
    + "".join([f" + {'-' if c < 0 else ''} {abs(c):.10f}x^{{{i}}}" for i, c in enumerate(coefficients, start=1)])
)

latex_eq = latex_eq.replace("x^{1}", "x")
# replace + - with -
latex_eq = latex_eq.replace(" + -", " - ")

st.latex(latex_eq)


# Line between the model and the plot
st.write("---")
max_coef = np.max(np.abs(coefficients))
max_coef_intercept = np.max([np.abs(intercept), max_coef])
st.write(f"Magnitude of largest coefficient: {max_coef_intercept:.4f}")

st.write(f"L1 norm of coefficients (+ intercept): {np.sum(np.abs(coefficients)) + np.abs(intercept):.4f}")

st.write(f"L2 norm of coefficients (+ intercept): {np.sqrt(np.sum(coefficients**2) + intercept**2):.4f}")