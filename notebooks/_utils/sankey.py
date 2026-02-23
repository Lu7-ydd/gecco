import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc

def plot_anndata_sankey(
    adata,
    sort_label,
    left_label,
    right_label,
    title='Sankey Plot',
    width=500,
    height=600,
    font_size=12,
    save_path=None
):
    """
    Create a Sankey plot from anndata.obs with 3 labels.
    
    Groups and sorts left nodes by sort_label, then simulates snap behavior
    to reorder right nodes by their flow-weighted y-coordinates.
    
    Parameters
    ----------
    adata : AnnData
        Input AnnData object.
    sort_label : str
        Name of the sorting label column in adata.obs. Values in this label
        determine the grouping. Each unique value in sort_label will have
        one or more corresponding values in left_label.
    left_label : str
        Name of the left label column in adata.obs (shown on the left side).
    right_label : str
        Name of the right label column in adata.obs (shown on the right side).
    title : str
        Title of the plot.
    width : int
        Width of the plot.
    height : int
        Height of the plot.
    font_size : int
        Font size for the plot.
    save_path : str, optional
        If provided, save debug info to this path.
    
    Returns
    -------
    fig : go.Figure
        The Plotly figure object.
    left_order : list
        Left side labels ordered from top to bottom (as shown in figure).
    right_order : list
        Right side labels ordered from top to bottom (as shown in figure).
    
    Notes
    -----
    1. Left nodes are ordered by sort_label:
       - Unique sort values are sorted by number of distinct left values (descending)
       - Within each sort group, left values are sorted alphabetically
    2. Right nodes are ordered by simulating snap behavior:
       - Calculate y-coordinate as weighted average of connected left nodes
       - Sort right nodes by y-coordinate (descending, top to bottom)
    3. Node labels are hidden on the plot (use hover to see labels)
    """
    
    # Extract the three label columns
    sort_data = np.array(adata.obs[sort_label]).astype(str)
    left_data = np.array(adata.obs[left_label]).astype(str)
    right_data = np.array(adata.obs[right_label]).astype(str)
    
    if len(sort_data) != len(left_data) or len(sort_data) != len(right_data):
        raise ValueError("All three label columns must have the same length!")
    
    # Create a DataFrame for easier manipulation
    df = pd.DataFrame({
        'sort': sort_data,
        'left': left_data,
        'right': right_data
    })
    
    # ===== STEP 1: Order left nodes by sort_label (like original plot_anndata_sankey) =====
    
    # sort: descending by number of unique left values
    sort_unique_counts = df.groupby('sort')['left'].nunique().reset_index(name='count')
    sort_unique_counts = sort_unique_counts.sort_values(
        ['count', 'sort'], 
        ascending=[False, True]
    )
    sort_order = sort_unique_counts['sort'].tolist()

    # Create mapping from sort value to its rank (for ordering)
    sort_rank_map = {val: i for i, val in enumerate(sort_order)}
    
    # Get unique left values, ordered by their sort_label rank
    sort_left_pairs = df[['sort', 'left']].drop_duplicates()
    sort_left_pairs['sort_rank'] = sort_left_pairs['sort'].map(sort_rank_map)
    sort_left_pairs = sort_left_pairs.sort_values(
        ['sort_rank', 'left'],
        ascending=[True, True]
    )
    l_unique = sort_left_pairs['left'].tolist()  # Left order (sorted by sort_label)
    
    # Right nodes: initially sorted alphabetically
    r_unique_orig = sorted(list(set(right_data)))
    
    n_left = len(l_unique)
    n_right = len(r_unique_orig)
    
    # ===== STEP 2: Calculate y-coordinates (simulate snap behavior) =====
    
    # Create initial index mapping (before snap reordering)
    l_map_init = {label: i for i, label in enumerate(l_unique)}
    r_map_init = {label: i for i, label in enumerate(r_unique_orig)}
    
    # Count flows
    flow_df = df.groupby(['left', 'right']).size().reset_index(name='count')
    
    # Calculate y-coordinates for left nodes (evenly spaced, y=1 at top)
    y_coords = {}
    for i in range(n_left):
        if n_left > 1:
            y_coords[i] = 1 - i / (n_left - 1)
        else:
            y_coords[i] = 0.5
    
    # Calculate y-coordinates for right nodes (weighted average of connected left nodes)
    right_y_weights = {}
    for idx, row in flow_df.iterrows():
        left_idx = l_map_init[row['left']]
        right_idx = r_map_init[row['right']]
        weight = row['count']
        
        if right_idx not in right_y_weights:
            right_y_weights[right_idx] = {'sum_y': 0, 'sum_weight': 0}
        right_y_weights[right_idx]['sum_y'] += y_coords[left_idx] * weight
        right_y_weights[right_idx]['sum_weight'] += weight
    
    for i in range(n_right):
        if i in right_y_weights and right_y_weights[i]['sum_weight'] > 0:
            y_coords[i + n_left] = right_y_weights[i]['sum_y'] / right_y_weights[i]['sum_weight']
        else:
            if n_right > 1:
                y_coords[i + n_left] = 1 - i / (n_right - 1)
            else:
                y_coords[i + n_left] = 0.5
    
    # ===== STEP 3: Simulate snap - reorder right nodes by y-coordinate =====
    
    # Left nodes keep their sort_label order
    left_sorted = list(range(n_left))  # No reordering for left
    
    # Right nodes: sort by y-coordinate (descending, top to bottom)
    right_sorted = sorted(range(n_right), key=lambda i: -y_coords.get(i + n_left, 0))
    
    # ===== STEP 4: Create new index mapping (after snap-like reordering) =====
    
    l_map_new = {l_unique[i]: i for i in range(n_left)}  # Left order unchanged
    r_map_new = {r_unique_orig[old_i]: n_left + new_i for new_i, old_i in enumerate(right_sorted)}
    
    # ===== STEP 5: Prepare labels =====
    
    # Left order (top to bottom) - same as sort_label order
    left_order = [f"{l_unique[i]} ({left_label})" for i in left_sorted]
    
    # Right order (top to bottom) - reordered by y-coordinate
    right_order = [f"{r_unique_orig[i]} ({right_label})" for i in right_sorted]
    
    # Combined real_labels (for hover)
    real_labels = left_order + right_order
    
    # Display labels (empty strings to hide on plot)
    display_labels = [""] * len(real_labels)
    
    # ===== STEP 6: Update flows with new mapping =====
    
    sources_new = [l_map_new[l] for l in flow_df['left']]
    targets_new = [r_map_new[r] for r in flow_df['right']]
    
    # ===== STEP 7: Color settings =====
    
    palette = pc.qualitative.Plotly * (n_left // len(pc.qualitative.Plotly) + 1)
    node_colors = []
    
    # Left colors (colorful)
    for i in range(n_left):
        node_colors.append(palette[i])
    # Right colors (gray)
    for i in range(n_right):
        node_colors.append("darkgray")

    link_color_static = "rgba(180, 180, 180, 0.3)"
    
    # ===== STEP 8: Create the plot =====
    # Use arrangement='snap' with explicit label order
    # Plotly respects the order of node.label array
    
    fig = go.Figure(data=[go.Sankey(
        arrangement='snap',
        node=dict(
            pad=5,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=display_labels,  # Hidden on plot
            customdata=real_labels,  # Shown on hover
            hovertemplate='%{customdata}<br>Count: %{value}<extra></extra>',
            color=node_colors
        ),
        link=dict(
            source=sources_new,
            target=targets_new,
            value=flow_df['count'].tolist(),
            color=link_color_static
        )
    )])

    fig.update_layout(
        title_text=title,
        font_size=font_size,
        width=width,
        height=height,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    # ===== STEP 9: Display and debug =====
    fig.show()
    
    
    # Save debug info if save_path provided
    if save_path:
        import json as json_mod
        
        json_path = save_path.replace('.html', '_debug.json') if save_path.endswith('.html') else save_path + '_debug.json'
        
        debug_info = {
            'left_order_top_to_bottom': left_order,
            'right_order_top_to_bottom': right_order,
            'l_unique': l_unique,
            'r_unique_original': r_unique_orig,
            'right_sorted_indices': right_sorted,
            'sort_order': sort_order,
            'sources': sources_new,
            'targets': targets_new,
            'values': flow_df['count'].tolist(),
            'node_colors': node_colors,
            'y_coordinates': {str(k): v for k, v in y_coords.items()},
            'flows': flow_df.to_dict('records')
        }
        
        with open(json_path, 'w') as f:
            json_mod.dump(debug_info, f, indent=2)
    