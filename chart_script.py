import plotly.graph_objects as go
import plotly.io as pio

# Data from the provided JSON
models = ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "K-Nearest Neighbors", "Naive Bayes", "AdaBoost"]
f1_scores = [0.85, 0.78, 0.89, 0.87, 0.81, 0.83, 0.86]

# Abbreviate model names to fit 15 character limit
model_abbrev = ["Logistic", "Decision Tree", "Random Forest", "SVM", "KNN", "Naive Bayes", "AdaBoost"]

# Brand colors in order
colors = ['#1FB8CD', '#DB4545', '#2E8B57', '#5D878F', '#D2BA4C', '#B4413C', '#964325']

# Create bar chart
fig = go.Figure()

# Add bars with different colors
for i, (model, abbrev, score) in enumerate(zip(models, model_abbrev, f1_scores)):
    fig.add_trace(go.Bar(
        x=[abbrev],
        y=[score],
        marker_color=colors[i],
        name=model,
        text=[f'{score:.2f}'],
        textposition='outside',
        cliponaxis=False
    ))

# Update layout
fig.update_layout(
    title="Binary Classification Performance",
    xaxis_title="ML Models",
    yaxis_title="F1-Score",
    yaxis=dict(range=[0, 1]),
    showlegend=False  # Remove legend since model names are on x-axis
)

# Save the chart
fig.write_image("model_performance_chart.png")