import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

# --- TSP Engine with High Precision ---
class WarehouseTSPPro:
    def __init__(self):
        self.rack_width = 0.51
        self.aisle_gap = 0.95
        self.total_rows_y = 10.0
        self.depot = {'x': 7.0, 'y': 0.0}
        
        # Calculate Aisle Centers (Indices 0 to 5)
        # Aisle 0 is between C-01 and C-02
        self.aisle_centers = []
        curr_x = 0.0
        for i in range(6):
            if i == 0 or i == 5: 
                curr_x += self.rack_width
            else: 
                curr_x += self.rack_width * 2
            self.aisle_centers.append(round(curr_x + self.aisle_gap / 2, 1))
            curr_x += self.aisle_gap

    def get_dist(self, p1, p2):
        """Calculates distance while strictly obeying warehouse rack constraints."""
        if abs(p1['x'] - p2['x']) < 0.01:
            return abs(p1['y'] - p2['y'])
        else:
            # Must cross via top (y=10) or bottom (y=0)
            dist_bottom = p1['y'] + abs(p1['x'] - p2['x']) + p2['y']
            dist_top = (10.0 - p1['y']) + abs(p1['x'] - p2['x']) + (10.0 - p2['y'])
            return min(dist_bottom, dist_top)

    def get_path_points(self, p1, p2):
        """Generates waypoints to prevent diagonal movement across racks."""
        if abs(p1['x'] - p2['x']) < 0.01:
            return [p1, p2]
        else:
            dist_bottom = p1['y'] + abs(p1['x'] - p2['x']) + p2['y']
            dist_top = (10.0 - p1['y']) + abs(p1['x'] - p2['x']) + (10.0 - p2['y'])
            y_cross = 0.0 if dist_bottom <= dist_top else 10.0
            return [p1, {'x': p1['x'], 'y': y_cross}, {'x': p2['x'], 'y': y_cross}, p2]

    def solve_tsp(self, picks):
        nodes = [self.depot] + picks
        n = len(nodes)
        
        # Distance Matrix
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist_matrix[i, j] = self.get_dist(nodes[i], nodes[j])

        # 1. Greedy Initialization (Nearest Neighbor)
        tour = [0]
        unvisited = list(range(1, n))
        while unvisited:
            curr = tour[-1]
            next_node = min(unvisited, key=lambda x: dist_matrix[curr, x])
            tour.append(next_node)
            unvisited.remove(next_node)
        tour.append(0)

        # 2. 2-opt Refinement for Maximum Precision
        improved = True
        while improved:
            improved = False
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    # Distance if we swap edges
                    old_dist = dist_matrix[tour[i-1], tour[i]] + dist_matrix[tour[j], tour[j+1]]
                    new_dist = dist_matrix[tour[i-1], tour[j]] + dist_matrix[tour[i], tour[j+1]]
                    if new_dist < old_dist:
                        tour[i:j+1] = reversed(tour[i:j+1])
                        improved = True
        
        # Final Path Generation
        full_path = []
        total_dist = 0
        for k in range(len(tour)-1):
            p1, p2 = nodes[tour[k]], nodes[tour[k+1]]
            full_path += self.get_path_points(p1, p2)
            total_dist += dist_matrix[tour[k], tour[k+1]]
            
        return full_path, total_dist, tour, nodes

# --- Visualizer Function ---
def draw_tsp_map(engine, path, picks):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Rack Dimensions
    rack_bounds = [0.0, 0.51, 1.46, 2.48, 3.43, 4.45, 5.40, 6.42, 7.37, 7.88]
    for i in range(0, len(rack_bounds), 2):
        ax.add_patch(patches.Rectangle((rack_bounds[i], 0), rack_bounds[i+1]-rack_bounds[i], 10, 
                                        facecolor='#f5f6fa', edgecolor='#dcdde1', linewidth=1))
    
    # Path Arrows
    for i in range(len(path)-1):
        p1, p2 = path[i], path[i+1]
        ax.annotate("", xy=(p2['x'], p2['y']), xytext=(p1['x'], p1['y']),
                    arrowprops=dict(arrowstyle="->", color="#2980b9", lw=1.5, alpha=0.8))
    
    # Pick Points - Corrected for Attribute Error
    for p in picks:
        ax.plot(p['x'], p['y'], 'ro', markersize=8, markeredgecolor='white', markeredgewidth=1, zorder=10)
    
    # Depot
    ax.plot(engine.depot['x'], engine.depot['y'], 'gs', markersize=14, label='Start/End (Depot)')
    
    ax.set_title("Optimal Picking Sequence (TSP Algorithm)", fontsize=16, fontweight='bold')
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.5, 11)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    return fig

# --- Streamlit Layout ---
st.set_page_config(page_title="Warehouse TSP Routing", layout="wide")

# Section: Image and Title
st.title("Warehouse Picking Route Optimization")

img = Image.open('Screenshot 2026-01-29 at 19.01.36.png')
st.image(img, caption='Warehouse Layout Reference', use_container_width=True)


engine = WarehouseTSPPro()

# Sidebar: Manual Entry Only
with st.sidebar:
    st.header("Pick Points Configuration")
    st.write("Enter locations as: Aisle, Y")
    st.write("(Aisle 0 = First aisle between C-01 and C-02)")
    
    default_points = "0, 1.5\n1, 4.0\n2, 8.5\n3, 2.0\n4, 6.5\n5, 1.0"
    user_input = st.text_area("Input picking locations (one per line):", value=default_points, height=300)

# Parsing Inputs
final_picks = []
lines = user_input.split('\n')
for line in lines:
    if ',' in line:
        try:
            a_idx, y_coord = map(float, line.split(','))
            a_idx = int(a_idx)
            if 0 <= a_idx <= 5:
                final_picks.append({
                    'aisle_idx': a_idx,
                    'x': engine.aisle_centers[a_idx],
                    'y': round(y_coord, 1)
                })
        except ValueError:
            pass

# Main Content Logic
if final_picks:
    full_path, total_distance, tour_order, nodes = engine.solve_tsp(final_picks)
    
    col_map, col_data = st.columns([3, 1])
    
    with col_map:
        # Display the TSP Visualization
        st.pyplot(draw_tsp_map(engine, full_path, final_picks))
    
    with col_data:
        st.subheader("Performance Summary")
        st.write(f"Total distance: **{total_distance:.2f} meters**")
        st.write(f"Number of points: **{len(final_picks)}**")
        
        st.subheader("Picking Order")
        picking_sequence = []
        for i, idx in enumerate(tour_order):
            location_name = "Depot" if idx == 0 else f"Point {idx}"
            picking_sequence.append({
                "Step": i,
                "Location": location_name,
                "Aisle": nodes[idx].get('aisle_idx', 'Home'),
                "Y-Position": nodes[idx]['y']
            })
        st.table(picking_sequence)

else:
    st.warning("Please enter valid pick points in the sidebar to generate the route.")