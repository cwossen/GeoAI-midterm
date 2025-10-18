import geopandas as gpd
import libpysal
import networkx as nx
import matplotlib.pyplot as plt
import os

# ✅ 1. Load shapefile
shapefile_path = r"C:\PythonProjects\seoul_submunicipalities.shp"

if not os.path.exists(shapefile_path):
    print(f"❌ Shapefile not found at: {shapefile_path}")
    exit()

gdf = gpd.read_file(shapefile_path)

# ✅ 2. Ensure the geometry is valid
gdf = gdf[gdf.is_valid]

# ✅ 3. Build spatial weights using Queen contiguity
w = libpysal.weights.Queen.from_dataframe(gdf)

# ✅ 4. Convert to NetworkX graph
G = w.to_networkx()

# ✅ 5. Basic graph stats
print(f"📊 Number of nodes: {G.number_of_nodes()}")
print(f"🔗 Number of edges: {G.number_of_edges()}")

# ✅ 6. Optional: visualize the adjacency graph (basic layout)
plt.figure(figsize=(8, 8))
pos = nx.spring_layout(G, seed=42)  # layout for visualization
nx.draw(G, pos, node_size=30, node_color='blue', edge_color='gray', with_labels=False)
plt.title("Seoul Neighborhood Adjacency Graph")
plt.show()

# ✅ 7. Attach adjacency info back to GeoDataFrame (optional)
gdf["neighbors"] = gdf.index.map(lambda idx: w.neighbors[idx])
print("\n📌 Sample of neighborhood adjacency:")
print(gdf[["neighbors"]].head())
