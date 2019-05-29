import networkx as nx
import math
from itertools import tee
import shapefile
import os

savedir = "."

def haversine(n0, n1):
    x1, y1 = n0
    x2, y2 = n1
    x_dist = math.radians(x1 - x2)
    y_dist = math.radians(y1 - y2)
    y1_rad = math.radians(y1)
    y2_rad = math.radians(y2)
    a = math.sin(y_dist/2)**2 + math.sin(x_dist/2)**2 \
    * math.cos(y1_rad) * math.cos(y2_rad)
    c = 2 * math.asin(math.sqrt(a))
    distance = c * 6371
    return distance

def pairwise(iterable):
    """Return an iterable in tuples of two
    s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

shp = "road_network.shp"


G = nx.DiGraph()
r = shapefile.Reader(shp)

for s in r.shapes():
    for p1, p2 in pairwise(s.points):
        G.add_edge(tuple(p1), tuple(p2))

sg = list(nx.connected_component_subgraphs(G.to_undirected()))[0]


r = shapefile.Reader("start_end")
start = r.shape(0).points[0]
end = r.shape(1).points[0]

for n0, n1 in sg.edges:
    dist = haversine(n0, n1)
    sg.edges[n0][n1]["dist"] = dist
    

nn_start = None
nn_end = None
start_delta = float("inf")
end_delta = float("inf")

for n in sg.nodes():
    s_dist = haversine(start, n)
    e_dist = haversine(end, n)
    if s_dist < start_delta:
        nn_start = n
        start_delta = s_dist
    if e_dist < end_delta:
        nn_end = n
        end_delta = e_dist
        
path = nx.shortest_path(sg, source=nn_start, target=nn_end, weight="dist")
w = shapefile.Writer(shapefile.POLYLINE)
w.field("NAME", "C", 40)
w.line(parts=[[list(p) for p in path]])
w.record("route")
w.save(os.path.join(savedir, "route"))
