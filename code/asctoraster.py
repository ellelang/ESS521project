import arcpy
# Set local variables
demASCII = "c:/Users/langzx/dem.asc"
demRaster = "c:/Users/langzx/demraster"
rasterType = "INTEGER"
arcpy.ASCIIToRaster_conversion(demASCII, demRaster, rasterType)

# Execute ASCIIToRaster

reliefASCII = "c:/Users/langzx/relief.asc"
reliefRaster = "c:/Users/langzx/reliefraster"
rasterType = "INTEGER"
arcpy.ASCIIToRaster_conversion(reliefASCII, reliefRaster, rasterType)

lidarASCII = "c:/Users/langzx/lidar.asc"
lidarRaster = "c:/Users/langzx/lidarraster"
rasterType = "INTEGER"
arcpy.ASCIIToRaster_conversion(lidarASCII, lidarRaster, rasterType)
