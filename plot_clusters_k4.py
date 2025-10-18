import geopandas as gpd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ✅ Load shapefile
shapefile_path = r"C:\PythonProjects\seoul_submunicipalities.shp"
gdf = gpd.read_file(shapefile_path)
gdf = gdf[gdf.is_valid].copy()

# ✅ Feature engineering
gdf["area"] = gdf.geometry.area
gdf["centroid_x"] = gdf.geometry.centroid.x
gdf["centroid_y"] = gdf.geometry.centroid.y

# ✅ Normalize features
gdf["area_norm"] = (gdf["area"] - gdf["area"].mean()) / gdf["area"].std()
gdf["x_norm"] = (gdf["centroid_x"] - gdf["centroid_x"].mean()) / gdf["centroid_x"].std()
gdf["y_norm"] = (gdf["centroid_y"] - gdf["centroid_y"].mean()) / gdf["centroid_y"].std()

features = gdf[["area_norm", "x_norm", "y_norm"]]

# ✅ KMeans clustering (k = 4)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
gdf["cluster"] = kmeans.fit_predict(features)

# ✅ Plotting
fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(column="cluster", cmap="tab10", legend=True, edgecolor="black", ax=ax)
plt.title("Seoul Neighborhoods Clustered (k = 4)", fontsize=14)
plt.axis("off")

# ✅ Annotate each neighborhood with its cluster label
for idx, row in gdf.iterrows():
    centroid = row["geometry"].centroid
    ax.text(
        centroid.x,
        centroid.y,
        str(row["cluster"]),
        fontsize=7,
        ha="center",
        va="center",
        color="white",
        weight="bold"
    )

plt.tight_layout()
plt.show()

# ✅ Save shapefile for QGIS
gdf.to_file("seoul_clusters_k4.shp")
print("✅ Clustered shapefile saved as seoul_clusters_k4.shp")
