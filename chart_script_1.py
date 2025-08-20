import plotly.graph_objects as go
import json

# Load the data
data = {
    "models": ["Random Forest", "Logistic Regression", "SVM", "Naive Bayes"], 
    "auc_scores": [0.94, 0.91, 0.92, 0.88], 
    "fpr_rf": [0, 0.05, 0.1, 0.2, 0.3, 1], 
    "tpr_rf": [0, 0.4, 0.7, 0.85, 0.92, 1], 
    "fpr_lr": [0, 0.08, 0.15, 0.25, 0.35, 1], 
    "tpr_lr": [0, 0.35, 0.65, 0.8, 0.88, 1], 
    "fpr_svm": [0, 0.06, 0.12, 0.22, 0.32, 1], 
    "tpr_svm": [0, 0.38, 0.68, 0.82, 0.9, 1], 
    "fpr_nb": [0, 0.1, 0.2, 0.3, 0.4, 1], 
    "tpr_nb": [0, 0.3, 0.6, 0.75, 0.85, 1]
}

# Brand colors in order
colors = ['#1FB8CD', '#DB4545', '#2E8B57', '#5D878F']

# Create figure
fig = go.Figure()

# Add ROC curves for each model
model_data = [
    ('Random Forest', data['fpr_rf'], data['tpr_rf'], data['auc_scores'][0]),
    ('Logistic Reg', data['fpr_lr'], data['tpr_lr'], data['auc_scores'][1]), 
    ('SVM', data['fpr_svm'], data['tpr_svm'], data['auc_scores'][2]),
    ('Naive Bayes', data['fpr_nb'], data['tpr_nb'], data['auc_scores'][3])
]

for i, (model, fpr, tpr, auc) in enumerate(model_data):
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'{model} (AUC={auc})',
        line=dict(color=colors[i], width=3),
        cliponaxis=False
    ))

# Add diagonal line for random classifier
fig.add_trace(go.Scatter(
    x=[0, 1],
    y=[0, 1],
    mode='lines',
    name='Random (AUC=0.5)',
    line=dict(color='gray', width=2, dash='dash'),
    cliponaxis=False
))

# Update layout
fig.update_layout(
    title='ROC Model Comparison',
    xaxis_title='False Pos Rate',
    yaxis_title='True Pos Rate',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

# Update axes
fig.update_xaxes(range=[0, 1], showgrid=True)
fig.update_yaxes(range=[0, 1], showgrid=True)

# Save the chart
fig.write_image('roc_curves_comparison.png')