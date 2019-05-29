import numpy as np
from pathlib import Path
data_folder = Path('C:/Users/langzx/Desktop/github/ESS521project/data/Assignment5')
import gdal
import ogr
from linecache import getline

myArray = np.loadtxt(data_folder/"myGrid.asc",skiprows=6)
myArray

header = "ncols {}\n".format(myArray.shape[1])
header += "nrows {}\n".format(myArray.shape[0])
header += "xllcorner 277750.0\n"
header += "yllcorner 6122250.0\n"
header += "cellsize 1.0\n"
header += "NODATA_value -9999"
np.savetxt(data_folder/"myGrid.asc", myArray, header=header, fmt="%1.2f")
f = open(data_folder/"myGrid.asc", "w")



# File name of ASCII digital elevation model
source = "dem.asc"
# File name of the slope grid
slopegrid = "slope.asc"
# File name of the aspect grid
aspectgrid = "aspect.asc"
# Output file name for shaded relief
shadegrid = "relief.asc"

# Shaded elevation parameters
# Sun direction
azimuth = 315.0
# Sun angle
altitude = 45.0
# Elevation exagerationexaggeration
z = 1.0
# Resolution
scale = 1.0
# No data value for output
NODATA = -9999
# Needed for numpy conversions
deg2rad = 3.141592653589793 / 180.0
rad2deg = 180.0 / 3.141592653589793
# Parse the header using a loop and
# the built-in linecache module
hdr = [getline(source, i) for i in range(1, 7)]
hdr
#float(hdr[0].split(" ")[-1].strip())

values = [float(h.split(" ")[-1].strip()) for h in hdr]
cols, rows, lx, ly, cell, nd = values
xres = cell
yres = cell * -1

# Load the dem into a numpy array
arr = np.loadtxt(data_folder/source, skiprows=6)
arr[11,20]
# Exclude 2 pixels around the edges which are usually NODATA.
# Also set up structure for 3x3 windows to process the slope
# throughout the grid
arr.shape[0]
arr.shape[1]
window = []
for row in range(3):
    for col in range(3):
        window.append(arr[row:(row + arr.shape[0] - 2),
                          col:(col + arr.shape[1] - 2)])
window

# Process each 3x3 window in both the x and y directions
x = ((z * window[0] + z * window[3] + z * window[3] + z *
window[6]) -
(z * window[2] + z * window[5] + z * window[5] + z *
window[8])) / (8.0 * xres * scale)

y = ((z * window[6] + z * window[7] + z * window[7] + z *
window[8]) -
(z * window[0] + z * window[1] + z * window[1] + z *
window[2])) /(8.0 * yres * scale)   

x
    
# Calculate slope
slope = 90.0 - np.arctan(np.sqrt(x * x + y * y)) * rad2deg
# Calculate aspect
aspect = np.arctan2(x, y)
# Calculate the shaded relief
shaded = np.sin(altitude * deg2rad) * np.sin(slope * deg2rad) + \
np.cos(altitude * deg2rad) * np.cos(slope * deg2rad) * \
np.cos((azimuth - 90.0) * deg2rad - aspect)
# Scale values from 0-1 to 0-255
shaded = shaded * 255

# Rebuild the new header
header = "ncols {}\n".format(shaded.shape[1])
header += "nrows {}\n".format(shaded.shape[0])
header += "xllcorner {}\n".format(lx + (cell * (cols -
shaded.shape[1])))
header += "yllcorner {}\n".format(ly + (cell * (rows -
shaded.shape[0])))
header += "cellsize {}\n".format(cell)
header += "NODATA_value {}\n".format(NODATA)
# Set no-data values
byheader = bytes(header, "UTF-8")
byheader
header
headercode = header.encode(encoding='UTF-8')
headercode
for pane in window:
    slope[pane == nd] = NODATA
    aspect[pane == nd] = NODATA
    shaded[pane == nd] = NODATA

slope

##deal with the hearder code utc-8
np.savetxt(data_folder/slopegrid, slope, header = header, fmt="%4i")
np.savetxt(data_folder/aspectgrid,aspect, header = header, fmt="%4i")
np.savetxt(data_folder/shadegrid, shaded, header = header, fmt="%4i")

###############

import gdal
import ogr
# Elevation DEM
source = "dem.asc"
# Output shapefile
target = "contour"
ogr_driver = ogr.GetDriverByName("ESRI Shapefile")
ogr_ds = ogr_driver.CreateDataSource(target + ".shp")
ogr_lyr = ogr_ds.CreateLayer(target,geom_type=ogr.wkbLineString25D)
field_defn = ogr.FieldDefn("ID", ogr.OFTInteger)
ogr_lyr.CreateField(field_defn)
field_defn = ogr.FieldDefn("ELEV", ogr.OFTReal)
ogr_lyr.CreateField(field_defn)
# gdal.ContourGenerate() arguments
# Band srcBand,
# double contourInterval,
# double contourBase,
# double[] fixedLevelCount,
# int useNoData,
# double noDataValue,
# Layer dstLayer,
# int idField,
# int elevField
ds = gdal.Open(source)
# EPGS:3157
gdal.ContourGenerate(ds.GetRasterBand(1), 400, 10, [], 0, 0, ogr_lyr,0, 1)
ogr_ds = None
import shapefile
import pngcanvas
# Open the contours
r = shapefile.Reader("contour.shp")
# Setup the world to pixels conversion
xdist = r.bbox[2] - r.bbox[0]
ydist = r.bbox[3] - r.bbox[1]
iwidth = 800
iheight = 600
xratio = iwidth/xdist
yratio = iheight/ydist
contours = []
# Loop through all shapes
for shape in r.shapes():
    for i in range(len(shape.parts)):
        pixels = []
        pt = None
        if i < len(shape.parts) - 1:
            pt = shape.points[shape.parts[i]:shape.parts[i+1]]
        else:
            pt = shape.points[shape.parts[i]:]
        for x, y in pt:
            px = int(iwidth - ((r.bbox[2] - x) * xratio))
            py = int((r.bbox[3] - y) * yratio)
            pixels.append([px, py])
        contours.append(pixels)
        
# Set up the output canvas
canvas = pngcanvas.PNGCanvas(iwidth, iheight)
# PNGCanvas accepts rgba byte arrays for colors
red = [0xff, 0, 0, 0xff]
canvas.color = red
# Loop through the polygons and draw them
for c in contours: 
    canvas.polyline(c)
# Save the image
with open("contours.png", "wb") as f:
    f.write(canvas.dump())

################
from laspy.file import File
import numpy as np
# Source LAS file
source = "lidar.las"
# Output ASCII DEM file
target = "lidar.asc"
# Grid cell size (data units)
cell = 1.0
# No data value for output DEM
NODATA = 0
# Open LIDAR LAS file
las = File(source, mode="r")
# xyz min and max
min = las.header.min
max = las.header.max
# Get the x axis distance im meters
xdist = max[0] - min[0]
# Get the y axis distance in meters
ydist = max[1] - min[1]
# Number of columns for our grid
cols = int(xdist) / cell
# Number of rows for our grid
rows = int(ydist) / cell

cols += 1
rows += 1
cols = int(cols)
rows = int(rows)
# Track how many elevation
# values we aggregate

count = np.zeros((rows, cols)).astype(np.float32)
# Aggregate elevation values

zsum = np.zeros((rows, cols)).astype(np.float32)
# Y resolution is negative
ycell = -1 * cell
# Project x, y values to grid
projx = (las.x - min[0]) / cell
projy = (las.y - min[1]) / ycell
# Cast to integers and clip for use as index
ix = projx.astype(np.int32)
iy = projy.astype(np.int32)
# Loop through x, y, z arrays, add to grid shape,
# and aggregate values for averaging
for x, y, z in np.nditer([ix, iy, las.z]):
    count[y, x] += 1
    zsum[y, x] += z
# Change 0 values to 1 to avoid numpy warnings,
# and NaN values in array
nonzero = np.where(count > 0, count, 1)
# Average our z values
zavg = zsum / nonzero
# Interpolate 0 values in array to avoid any
# holes in the grid
mean = np.ones((rows, cols)) * np.mean(zavg)
left = np.roll(zavg, -1, 1)
lavg = np.where(left > 0, left, mean)
right = np.roll(zavg, 1, 1)
ravg = np.where(right > 0, right, mean)
interpolate = (lavg + ravg) / 2
fill = np.where(zavg > 0, zavg, interpolate)
    
# Create our ASCII DEM header
header = "ncols {}\n".format(fill.shape[1])
header += "nrows {}\n".format(fill.shape[0])
header += "xllcorner {}\n".format(min[0])
header += "yllcorner {}\n".format(min[1])
header += "cellsize {}\n".format(cell)
header += "NODATA_value {}\n".format(NODATA)
with open(target, "wb") as f:
    f.write(bytes(header, 'UTF-8'))
    np.savetxt(f, fill, fmt="%1.2f")



try:
    import Image
    import ImageOps
except:
    from PIL import Image, ImageOps

# Source gridded LIDAR DEM file
source = "lidar.asc"
# Output image file
target = "lidar.bmp"
# Load the ASCII DEM into a numpy array
arr = np.loadtxt(source, skiprows=6)
# Convert array to numpy image
im = Image.fromarray(arr).convert("RGB")
# Enhance the image:
# equalize and increase contrast
im = ImageOps.equalize(im)
im = ImageOps.autocontrast(im)
# Save the image
im.save(target)

import colorsys
source = "lidar.asc"
# Output image file
target = "lidar.bmp"
# Load the ASCII DEM into a numpy array
arr = np.loadtxt(source, skiprows=6)
# Convert the numpy array to a PIL image.
# Use black and white mode so we can stack
# three bands for the color image.
im = Image.fromarray(arr).convert('L')
# Enhance the image
im = ImageOps.equalize(im)
im = ImageOps.autocontrast(im)

# Begin building our color ramp
palette = []
# Hue, Saturation, Value
# color space starting with yellow.
h = .67
s = 1
v = 1
# We'll step through colors from:
# blue-green-yellow-orange-red.
# Blue=low elevation, Red=high-elevation
step = h / 256.0
# Build the palette
for i in range(256):
    rp, gp, bp = colorsys.hsv_to_rgb(h, s, v)
    r = int(rp * 255)
    g = int(gp * 255)
    b = int(bp * 255)
    palette.extend([r, g, b])
    h -= step
# Apply the palette to the image
im.putpalette(palette)
# Save the image
im.save(target)
