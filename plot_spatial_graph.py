import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
from libpysal.weights import Queen

# Load shapefile
shapefile_path = r"C:\PythonProjects\seoul_submunicipalities.shp"
gdf = gpd.read_file(shapefile_path)
gdf = gdf[gdf.is_valid].copy()
gdf = gdf.reset_index(drop=True)

# Compute Queen adjacency
w = Queen.from_dataframe(gdf)

# Create Graph
G = nx.Graph()

# Add nodes with positions based on centroids
for i, row in gdf.iterrows():
    centroid = row.geometry.centroid
    G.add_node(i, pos=(centroid.x, centroid.y), cluster=row.get("cluster", -1))

# Add edges between neighbors
for i, neighbors in w.neighbors.items():
    for j in neighbors:
        if not G.has_edge(i, j):
            G.add_edge(i, j)

# Plot graph
pos = nx.get_node_attributes(G, 'pos')
clusters = nx.get_node_attributes(G, 'cluster')
colors = [clusters[n] for n in G.nodes]

plt.figure(figsize=(12, 12))
nx.draw(G, pos, node_color=colors, cmap='tab10', node_size=50, edge_color="gray", with_labels=False)
plt.title("Spatial Adjacency Graph of Seoul Neighborhoods (Colored by Cluster)", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.show()
