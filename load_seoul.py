import geopandas as gpd
import matplotlib.pyplot as plt
import os

# ✅ 1. Define full path to your shapefile
shapefile_path = r"C:\PythonProjects\seoul_submunicipalities.shp"

# ✅ 2. Check if file exists before loading
if not os.path.exists(shapefile_path):
    print(f"❌ File not found: {shapefile_path}")
    exit()

# ✅ 3. Load shapefile
gdf = gpd.read_file(shapefile_path)

# ✅ 4. Inspect data
print("✅ First 5 rows:")
print(gdf.head())

print("\n📌 Columns:")
print(gdf.columns)

print("\n📦 Geometry types:")
print(gdf.geometry.geom_type.value_counts())

print("\n📍 Number of neighborhoods:", len(gdf))

# ✅ 5. Plot the neighborhoods
gdf.plot(edgecolor='black', figsize=(10, 10))
plt.title("🗺️ Seoul Submunicipalities (Dong Level)")
plt.show()

# ✅ 6. Add area column (in current CRS units)
gdf['area'] = gdf.geometry.area
print("\n📏 First 5 area values:")
print(gdf[['area']].head())
