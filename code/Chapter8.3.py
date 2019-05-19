import numpy as np
from linecache import getline

def floodFill(c, r, mask):
    """
    Crawls a mask array containing
    only 1 and 0 values from the
    starting point (c=column,
    r=row - a.k.a. x, y) and returns
    an array with all 1 values
    connected to the starting cell.
    This algorithm performs a 4-way
    check non-recursively.
    """
    # cells already filled
    filled = set()
    # cells to fill
    fill = set()
    fill.add((c, r))
    width = mask.shape[1]-1
    height = mask.shape[0]-1
    # Our output inundation array
    flood = np.zeros_like(mask, dtype=np.int8)
    # Loop through and modify the cells which
    # need to be checked.
    while fill:
        # Grab a cell
        x, y = fill.pop()
        if y == height or x == width or x < 0 or y < 0:
        # Don't fill
            continue
        if mask[y][x] == 1:
        # Do fill
            flood[y][x] = 1
            filled.add((x, y))
            # Check neighbors for 1 values
            west = (x-1, y)
            east = (x+1, y)
            north = (x, y-1)
            south = (x, y+1)
            if west not in filled:
                fill.add(west)
            if east not in filled:
                fill.add(east)
            if north not in filled:
                fill.add(north)
            if south not in filled:
                fill.add(south)
    return flood


source = "terrain.asc"
target = "flood90.asc"
print("Opening image...")
img = np.loadtxt(source, skiprows=6)
print("Image opened")
# Mask elevations lower than 70 meters.
wet = np.where(img < 90, 1, 0)
print("Image masked")
# Parse the header using a loop and
# the built-in linecache module
hdr = [getline(source, i) for i in range(1, 7)]
values = [float(h.split(" ")[-1].strip()) for h in hdr]
cols, rows, lx, ly, cell, nd = values
xres = cell
yres = cell * -1
# Starting point for the
# flood inundation in pixel coordinates
sx = 2582

sy = 2057
print("Beginning flood fill")
fld = floodFill(sx, sy, wet)
print("Finished flood fill")
header = ""

for i in range(6):
    header += hdr[i]
print("Saving grid")
# Open the output file, add the hdr, save the array
with open(target, "wb") as f:
    f.write(bytes(header, 'UTF-8'))
    np.savetxt(f, fld, fmt="%1i")
print("Done!")

##Creating a color hillshade

relief = "relief.asc"
dem = "dem.asc"
target = "hillshade.tif"
# Load the relief as the background image
bg = gd.numpy.loadtxt(relief, skiprows=6)
# Load the DEM into a numpy array as the foreground image
fg = gd.numpy.loadtxt(dem, skiprows=6)[:-2, :-2]
# Create a blank 3-band image to colorize the DEM
rgb = gd.numpy.zeros((3, len(fg), len(fg[0])), gd.numpy.uint8)
# Class list with DEM upper elevation range values.
classes = [356, 649, 942, 1235, 1528,1821, 2114, 2300, 2700]
# Color look-up table (lut)
# The lut must match the number of classes.
# Specified as R, G, B tuples
lut = [[63, 159, 152], [96, 235, 155], [100, 246, 174],\
       [248, 251, 155], [246, 190, 39], [242, 155, 39],\
       [165, 84, 26], [236, 119, 83], [203, 203, 203]]
# Starting elevation value of the first class
start = 1
# Process all classes.
for i in range(len(classes)):
    mask = gd.numpy.logical_and(start <= fg,\
    fg <= classes[i])
for j in range(len(lut[i])):
    rgb[j] = gd.numpy.choose(mask, (rgb[j], lut[i][j]))
start = classes[i]+1

# Convert the shaded relief to a PIL image
im1 = Image.fromarray(bg).convert('RGB')
# Convert the colorized DEM to a PIL image.
# We must transpose it from the Numpy row, col order
# to the PIL col, row order (width, height).
im2 = Image.fromarray(rgb.transpose(1, 2, 0)).convert('RGB')
# Blend the two images with a 40% alpha
hillshade = Image.blend(im1, im2, .4)
# Save the hillshade
hillshade.save(target)
