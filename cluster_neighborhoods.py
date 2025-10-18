import geopandas as gpd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

# ✅ Path to your shapefile
shapefile_path = r"C:\PythonProjects\seoul_submunicipalities.shp"
gdf = gpd.read_file(shapefile_path)

# ✅ Compute features again (or load from previous step if saved)
gdf = gdf[gdf.is_valid].copy()
gdf["area"] = gdf.geometry.area
gdf["centroid_x"] = gdf.geometry.centroid.x
gdf["centroid_y"] = gdf.geometry.centroid.y

# ✅ Normalize (manually to avoid re-importing scaler)
gdf["area_norm"] = (gdf["area"] - gdf["area"].mean()) / gdf["area"].std()
gdf["x_norm"] = (gdf["centroid_x"] - gdf["centroid_x"].mean()) / gdf["centroid_x"].std()
gdf["y_norm"] = (gdf["centroid_y"] - gdf["centroid_y"].mean()) / gdf["centroid_y"].std()

# ✅ Select features for clustering
features = gdf[["area_norm", "x_norm", "y_norm"]]

# ✅ Run k-means clustering
k = 5  # try different values later (e.g., 3 to 8)
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
gdf["cluster"] = kmeans.fit_predict(features)

# ✅ Plot the clusters
gdf.plot(column="cluster", cmap="tab10", legend=True, figsize=(10, 10), edgecolor="black")
plt.title(f"Clustering of Seoul Neighborhoods (k = {k})")
plt.axis("off")
plt.show()

# ✅ Optional: save shapefile with cluster labels
output_path = r"C:\PythonProjects\seoul_clusters.shp"
gdf.to_file(output_path)
print(f"✅ Clustered shapefile saved to {output_path}")
