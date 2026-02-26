import plotly.graph_objects as go
import plotly.subplots as sp
import pandas as pd
from plotly.colors import qualitative
import numpy as np

def plot_cumulative_reward_interactive_enhanced(reward_log, time_steps, action_log, event_df):
    """
    Enhanced interactive cumulative reward line chart with action annotations,
    significant events highlighting, and detailed hover information.
    """
    cumulative = []
    total = 0
    for r in reward_log:
        total += r
        cumulative.append(total)
    
    # Identify significant reward changes (could be important decision points)
    significant_points = []
    reward_changes = [0] + [reward_log[i] - reward_log[i-1] for i in range(1, len(reward_log))]
    threshold = np.std(reward_changes) * 1.5
    for i, change in enumerate(reward_changes):
        if abs(change) > threshold:
            significant_points.append(i)
    
    # Create the main figure
    fig = sp.make_subplots(specs=[[{"secondary_y": True}]])
    
    # Main cumulative reward line
    fig.add_trace(go.Scatter(
        x=time_steps,
        y=cumulative,
        mode='lines+markers',
        line=dict(color='#1f77b4', width=2),
        marker=dict(
            size=8,
            color=['#FF4500' if i in significant_points else '#1f77b4' for i in range(len(time_steps))],
            line=dict(width=2, color='DarkSlateGrey')
        ),
        name='Cumulative Reward',
        text=[
            f"<b>Step {t}</b><br>" + 
            f"Step Reward: {r:.2f}<br>" + 
            f"Cumulative: {cr:.2f}<br>" + 
            f"Action: {act}<br>" + 
            f"{'<b>SIGNIFICANT CHANGE!</b>' if i in significant_points else ''}"
            for i, (t, r, cr, act) in enumerate(zip(time_steps, reward_log, cumulative, action_log))
        ],
        hovertemplate="%{text}<extra></extra>"
    ))
    
    # Step rewards as a bar chart
    fig.add_trace(go.Bar(
        x=time_steps,
        y=reward_log,
        name='Step Reward',
        marker_color='rgba(58, 71, 80, 0.6)',
        opacity=0.7,
        yaxis='y2'
    ))
    
    # Add annotations for significant points
    for point in significant_points:
        fig.add_annotation(
            x=time_steps[point],
            y=cumulative[point],
            text=f"Action: {action_log[point]}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40
        )
    
    # Update layout
    fig.update_layout(
        title={
            'text': "Agent Reward Analysis Over Time",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        xaxis=dict(
            title="Time Step",
            tickfont=dict(size=12),
            gridcolor='lightgray',
        ),
        yaxis=dict(
            title="Cumulative Reward",
            tickfont=dict(size=12),
            gridcolor='lightgray',
        ),
        yaxis2=dict(
            title=dict(
                text="Step Reward",
                font=dict(color="rgba(58, 71, 80, 1)")  # Correct structure
                ),
            tickfont=dict(color="rgba(58, 71, 80, 1)"),
            anchor="x",
            overlaying="y",
            side="right"
            ),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=60, r=60, t=80, b=60),
        plot_bgcolor='rgba(240, 240, 240, 0.5)'
    )
    
    # Add manual controls
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                buttons=[
                    dict(
                        args=[{"visible": [True, True]}],
                        label="Show All",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [True, False]}],
                        label="Cumulative Only",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [False, True]}],
                        label="Step Rewards Only",
                        method="update"
                    )
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top"
            ),
        ]
    )


    # Add a range slider for time steps
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="linear"
        )
    )
    
    return fig

def plot_reward_decomposition_interactive_enhanced(time_steps, reward_dict, action_log, event_df=None):
    """
    Enhanced interactive stacked area chart for reward decomposition with 
    improved hover details, filtering, and decision point analysis.
    """
    # Create DataFrame from reward dictionary
    df = pd.DataFrame(reward_dict, index=time_steps)
    df = df.loc[:, (df != 0).any(axis=0)]
    
    # Create figure
    fig = go.Figure()
    
    # Calculate percentage contribution
    df_percentage = df.copy().astype(float)
    for i in range(len(df)):
        row_sum = df.iloc[i].sum()
        if row_sum > 0:
            df_percentage.iloc[i] = (df.iloc[i] / row_sum) * 100
    
    # Color scheme
    vibrant_colors = qualitative.Bold
    
    # Add traces for each reward component
    for i, channel in enumerate(df.columns):
        # Create hover texts for non-zero values
        hover_texts = []
        
        for idx, t in enumerate(time_steps):
            if idx < len(df):
                raw_value = df[channel].iloc[idx]
                
                if raw_value > 0:
                    pct_value = df_percentage[channel].iloc[idx]
                    cumulative_val = df[channel].iloc[:idx+1].sum()
                    current_action = action_log[idx] if idx < len(action_log) else "unknown"
                    
                    # Note: No time step here, it will appear once at the top in unified mode
                    hover_texts.append(
                        f"<b>Component:</b> {channel}<br>" +
                        f"<b>Value:</b> {raw_value:.2f} ({pct_value:.1f}%)<br>" +
                        f"<b>Action:</b> {current_action}<br>" +
                        f"<b>Cumulative:</b> {cumulative_val:.2f}<br><br>"
                    )
                else:
                    # For zero values, use None to skip them in hover display
                    hover_texts.append(None)
        
        # Add trace with prepared hover texts
        fig.add_trace(go.Scatter(
            x=time_steps,
            y=df[channel],          
            mode='lines',
            name=channel,
            stackgroup='one',
            line=dict(width=2, color=vibrant_colors[i % len(vibrant_colors)]),
            opacity=0.8,
            hoverinfo='text' if any(df[channel] > 0) else 'skip',
            hovertext=hover_texts
        ))
    
    # Add total reward line - AFTER component traces
    fig.add_trace(go.Scatter(
        x=time_steps,
        y=df.sum(axis=1),
        mode='lines',
        name='Total Reward',
        line=dict(color='black', width=3, dash='dot'),
        visible='legendonly'  # Hidden by default, can be toggled
    ))
    
    # Add vertical lines for significant actions
    # Identify action changes
    action_changes = [i for i in range(1, len(action_log)) if action_log[i] != action_log[i-1]]
    for idx in action_changes:
        if idx < len(time_steps):
            fig.add_shape(
                type="line",
                x0=time_steps[idx], y0=0,
                x1=time_steps[idx], y1=df.sum(axis=1).max() * 1.1,
                line=dict(color="rgba(0,0,0,0.3)", width=1, dash="dash"),
            )
            fig.add_annotation(
                x=time_steps[idx], 
                y=df.sum(axis=1).max() * 1.05,
                text=f"→ {action_log[idx]}",
                showarrow=False,
                textangle=-90,
                font=dict(size=10)
            )
    
    # Update layout with more controls and details
    fig.update_layout(
        title={
            'text': "Reward Component Analysis - Decision Making Insights",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        xaxis_title="Time Step",
        yaxis_title="Reward Component Value",
        hovermode='x unified',  # This shows time step once at the top
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial",
            bordercolor="lightgray"
        ),
        legend=dict(
            title="Reward Components",
            groupclick="toggleitem"
        ),
        margin=dict(l=60, r=60, t=100, b=60),
        plot_bgcolor='rgba(240, 240, 240, 0.5)'
    )
    
    # Add annotation explaining hover behavior
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.05,
        text="Red points indicate significant reward changes that may explain agent decisions",
        showarrow=False,
        font=dict(size=10),
        align="left"
    )
    
    return fig





def create_decision_analysis_dashboard(reward_log, time_steps, action_log, reward_dict, event_df=None):
    """
    Creates a comprehensive dashboard with synchronized visualizations for decision analysis.
    """
    from plotly.subplots import make_subplots
    
    # Create figures
    fig1 = plot_cumulative_reward_interactive_enhanced(reward_log, time_steps, action_log, event_df)
    fig2 = plot_reward_decomposition_interactive_enhanced(time_steps, reward_dict, action_log, event_df)
    
    # Combine into a single dashboard
    dashboard = make_subplots(
        rows=2, 
        cols=1,
        subplot_titles=("Agent Reward Progression", "Reward Component Analysis"),
        vertical_spacing=0.2,
        specs=[[{"type": "scatter"}], [{"type": "scatter"}]]
    )
    
    # Add traces from fig1
    for trace in fig1.data:
        dashboard.add_trace(trace, row=1, col=1)
    
    # Add traces from fig2
    for trace in fig2.data:
        dashboard.add_trace(trace, row=2, col=1)
    
    # Update layout
    dashboard.update_layout(
        height=1000,
        title={
            'text': "Agent Decision Making Analysis",
            'y':0.98,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        hovermode='x unified',
    )
    
    # Add a note about significant points
    dashboard.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.01,
        text="",
        showarrow=False,
        font=dict(size=10),
        align="left"
    )
    
    return dashboard
