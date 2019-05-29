import arcpy
# Set local variables
demASCII = "c:/Users/langzx/dem.asc"
dem8Raster = "c:/Users/langzx/dem8raster"
rasterType = "INTEGER"
arcpy.ASCIIToRaster_conversion(demASCII, dem8Raster, rasterType)

# Execute ASCIIToRaster

reliefASCII = "c:/Users/langzx/relief.asc"
relief8Raster = "c:/Users/langzx/relief8raster"
rasterType = "INTEGER"
arcpy.ASCIIToRaster_conversion(reliefASCII, relief8Raster, rasterType)

lidarASCII = "c:/Users/langzx/lidar.asc"
lidarRaster = "c:/Users/langzx/lidarraster"
rasterType = "INTEGER"
arcpy.ASCIIToRaster_conversion(lidarASCII, lidarRaster, rasterType)

##Chapter 8 flood

terrainASCII = "c:/Users/langzx/terrain.asc"
terrainRaster = "c:/Users/langzx/terrainraster"
rasterType = "INTEGER"
arcpy.ASCIIToRaster_conversion(terrainASCII, terrainRaster, rasterType)

floodASCII = "c:/Users/langzx/flood.asc"
floodRaster = "c:/Users/langzx/floodraster"
rasterType = "INTEGER"
arcpy.ASCIIToRaster_conversion(floodASCII, floodRaster, rasterType)

flood90ASCII = "c:/Users/langzx/flood90.asc"
flood90Raster = "c:/Users/langzx/flood90raster"
rasterType = "INTEGER"
arcpy.ASCIIToRaster_conversion(flood90ASCII, flood90Raster, rasterType)

pathASCII = "c:/Users/langzx/path.asc"
pathRaster = "c:/Users/langzx/pathraster"
rasterType = "INTEGER"
arcpy.ASCIIToRaster_conversion(pathASCII, pathRaster, rasterType)
