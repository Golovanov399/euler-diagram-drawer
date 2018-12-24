#!/usr/bin/python3

import sys

class Graph:
	def __init__(self, labels):
		self.labels = labels
		self.n = len(labels)
		# for our purposes it's convenitent to store it as an incidence matrix
		self.inc = [[False] * len(labels) for x in labels]

	def add_edge(self, u, v):
		# undirected
		self.inc[u][v] = True
		self.inc[v][u] = True

	def get_edges(self):
		for i in range(self.n):
			for j in range(i):
				if self.inc[i][j]:
					yield (j, i)

	def are_connected(self, u, v):
		st = [u]
		used = {u}
		while st != []:
			u = st.pop()
			if u == v:
				return True
			for i in range(self.n):
				if self.inc[u][i] and i not in used:
					used.add(i)
					st.append(i)
		return False

	def add_vertices(self, cnt, label):
		self.labels += [label] * cnt
		for i in range(self.n):
			self.inc[i] += [False] * cnt
		self.n += cnt
		self.inc += [[False] * self.n for i in range(cnt)]

def spring(g):
	# spring embedder
	# https://arxiv.org/pdf/1201.3011.pdf, algorithm 1

	import random
	from math import log, sqrt
	n = g.n
	maxc = 1000
	xy = [(random.randint(-maxc, maxc), random.randint(-maxc, maxc)) for i in range(n)]

	# try to force that 0 label should be on the outer face
	min_idx = 0
	for i in range(1, n):
		if xy[i][0] < xy[min_idx][0]:
			min_idx = i
	if min_idx > 0:
		xy[min_idx], xy[0] = xy[0], xy[min_idx]

	c1, c2, c3, c4, M = 2, maxc / 2, 1, 0.1, 10000
	for i in range(M):
		forces = []
		for j in range(n):
			x, y = 0, 0
			for k in range(n):
				if k == j:
					continue
				cur_x, cur_y = xy[k][0] - xy[j][0], xy[k][1] - xy[j][1]
				d = sqrt(cur_x ** 2 + cur_y ** 2)
				scale = c1 * log(d / c2) if g.inc[j][k] else -c3 / sqrt(d)
				x += cur_x / d * scale
				y += cur_y / d * scale
			forces.append((x, y))
		for j in range(n):
			xy[j] = (xy[j][0] + 4 * forces[j][0],
					 xy[j][1] + 4 * forces[j][1])
	return xy

const_sz = (500, 500)
const_bb = (400, 400)

def normalize(positions, sz, bb):
	mx, Mx, my, My = (None,) * 4
	for p in positions:
		mx = p[0] if not mx else min(mx, p[0])
		Mx = p[0] if not Mx else max(Mx, p[0])
		my = p[1] if not my else min(my, p[1])
		My = p[1] if not My else max(My, p[1])
	k = max((Mx - mx) / bb[0], (My - my) / bb[1], 1)
	actual = ((Mx - mx) / k, (My - my) / k)
	off = ((sz[0] - actual[0]) / 2,
		   (sz[1] - actual[1]) / 2)
	for i in range(len(positions)):
		p = positions[i]
		positions[i] = (off[0] + (p[0] - mx) / k, off[1] + (p[1] - my) / k)
	return positions

def generate_pic(positions, graph, filename, norm=True):
	from PIL import Image, ImageDraw
	sz = const_sz
	bb = const_bb
	im = Image.new("RGB", sz, (255, 255, 255))
	draw = ImageDraw.Draw(im)
	if norm:
		positions = normalize(positions, const_sz, const_bb)
	for i in range(len(positions)):
		for j in range(i):
			if graph.inc[i][j]:
				draw.line(list(map(int, [positions[i][0],
										 positions[i][1],
										 positions[j][0],
										 positions[j][1]])),
						  fill="black", width=2)
	for i in range(len(positions)):
		p = positions[i]
		draw.ellipse(list(map(int, [p[0] - 3,
									p[1] - 3,
									p[0] + 3,
									p[1] + 3])),
					 fill="red",
					 outline="black")
	for i in range(len(positions)):
		p = positions[i]
		draw.text(list(map(int, [p[0] + 3,
								 p[1] + 3])),
				  str(graph.labels[i]),
				  fill="blue")
	im.save(filename)

def sign(x):
	eps = 1e-5
	return -1 if x < -eps else 1 if x > eps else 0

def popcount(x):
	return bin(x).count('1')

class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def __add__(self, p):
		return Point(self.x + p.x, self.y + p.y)

	def __sub__(self, p):
		return Point(self.x - p.x, self.y - p.y)

	def __mul__(self, p):	# cross product
		return self.x * p.y - self.y * p.x

	def __mod__(self, p):	# dot product
		return self.x * p.x + self.y * p.y

	def half(self):	# 0 if y > 0 or (y == 0 and x > 0), 1 otherwise
		if sign(self.y) != 0:
			return sign(self.y) < 0
		else:
			return sign(self.x) < 0

	def __repr__(self):
		return "(%.5f, %.5f)" % (self.x, self.y)

def intersects(a, b, c, d):
	# checks whether [a, b] intersects [c, d], but not sure about endpoints
	if not isinstance(a, Point):
		a = Point(*a)
		b = Point(*b)
		c = Point(*c)
		d = Point(*d)
	if sign((c - a) * (b - a)) == 0:
		if sign((d - a) * (b - a)) == 0:
			return not(sign((c - a) % (d - a)) == sign((c - b) % (d - b)) == sign((a - c) % (b - c)) == sign((a - d) % (b - d)) == 1)
		else:
			return sign((a - c) % (b - c)) == -1
	if sign((d - a) * (b - a)) == 0:
		if sign((c - a) * (b - a)) == 0:
			return not(sign((c - a) % (d - a)) == sign((c - b) % (d - b)) == sign((a - c) % (b - c)) == sign((a - d) % (b - d)) == 1)
		else:
			return sign((a - d) % (b - d)) == -1
	if sign((a - c) * (d - c)) == 0:
		if sign((b - c) * (d - c)) == 0:
			return not(sign((a - c) % (b - c)) == sign((a - d) % (b - d)) == sign((c - a) % (d - a)) == sign((c - b) % (d - b)) == 1)
		else:
			return sign((c - a) % (d - a)) == -1
	if sign((b - c) * (d - c)) == 0:
		if sign((a - c) * (d - c)) == 0:
			return not(sign((a - c) % (b - c)) == sign((a - d) % (b - d)) == sign((c - a) % (d - a)) == sign((c - b) % (d - b)) == 1)
		else:
			return sign((c - b) % (d - b)) == -1
	return sign((c - a) * (b - a)) == sign((b - a) * (d - a)) and sign((a - c) * (d - c)) == sign((d - c) * (b - c))

def get_faces(graph, positions):
	# obtain faces from a graph planar representation
	sorted_neighbors = [[] for i in range(graph.n)]
	for i in range(graph.n):
		for j in range(i):
			if graph.inc[i][j]:
				sorted_neighbors[i].append(j)
				sorted_neighbors[j].append(i)
	def comp_build(ip):
		p = Point(*positions[ip])
		def comp(ix, iy):
			x = Point(*positions[ix])
			y = Point(*positions[iy])
			if (x - p).half() != (y - p).half():
				return (x - p).half() - (y - p).half()
			else:
				return -sign((x - p) * (y - p))
		return comp
	from functools import cmp_to_key
	for i in range(graph.n):
		sorted_neighbors[i].sort(key=cmp_to_key(comp_build(i)))
	
	faces = []
	used_edges = set()
	for i in range(graph.n):
		for j in sorted_neighbors[i]:
			if (i, j) in used_edges:
				continue
			face = [i]
			u, v = i, j
			used_edges.add((u, v))
			while v != i:
				u, v = v, u
				idx = sorted_neighbors[u].index(v) - 1
				v = sorted_neighbors[u][idx]
				face.append(u)
				used_edges.add((u, v))
			faces.append(face)
	return faces

def can_draw_edge(edges, positions, e):
	for ed in edges:
		if len(set(ed) | set(e)) < 4:
			continue
		if intersects(*map(positions.__getitem__, ed + e)):
			print(ed, e, file=sys.stderr)
			return False
	for i in range(len(positions)):
		if i in e:
			continue
		if sign((Point(*positions[i]) - Point(*positions[e[0]])) * (Point(*positions[i]) - Point(*positions[e[1]]))) == 0 and \
		   sign((Point(*positions[i]) - Point(*positions[e[0]])) % (Point(*positions[i]) - Point(*positions[e[1]]))) < 0:
			return False
	return True

def generate_diagram(graph, trs, tr_edges, positions, filename):
	from PIL import Image, ImageDraw
	sz = (500, 500)
	bb = (400, 400)
	im = Image.new("RGB", sz, (255, 255, 255))
	draw = ImageDraw.Draw(im, "RGBA")

	colors = (
		(240, 163, 255),
		(0, 117, 220),
		(153, 63, 0),
		(76, 0, 92),
		(25, 25, 25),
		(0, 92, 49),
		(43, 206, 72),
		(255, 204, 153),
		(128, 128, 128),
		(148, 255, 181),
		(143, 124, 0),
		(157, 204, 0),
		(194, 0, 136),
		(0, 51, 128),
		(255, 164, 5),
		(255, 168, 187),
		(66, 102, 0),
		(255, 0, 16),
		(94, 241, 242),
		(0, 153, 143),
		(224, 255, 102),
		(116, 10, 255),
		(153, 0, 0),
		(255, 255, 128),
		(255, 255, 0),
		(255, 80, 5),
	) # from https://en.wikipedia.org/wiki/Help:Distinguishable_colors

	total_mask = 0
	for m in graph.labels:
		total_mask |= m
	colors_cnt = popcount(total_mask)
	imgs = [Image.new("RGBA", const_sz, (255, 255, 255, 0)) for i in range(colors_cnt)]
	draws = list(map(lambda x: ImageDraw.Draw(x), imgs))

	def get_by_mask(mask):
		return [x for x in range(colors_cnt) if ((mask & (1 << x)) > 0)]

	get_position = dict()
	for i, j in graph.get_edges():
		if (i, j) in tr_edges:
			for x in get_by_mask(graph.labels[i] & (~graph.labels[j])):
				get_position[(i, j, x)] = (2 * positions[i][0] + positions[j][0]) / 3, (2 * positions[i][1] + positions[j][1]) / 3
				get_position[(j, i, x)] = (2 * positions[i][0] + positions[j][0]) / 3, (2 * positions[i][1] + positions[j][1]) / 3
			for x in get_by_mask(graph.labels[j] & (~graph.labels[i])):
				get_position[(i, j, x)] = (positions[i][0] + 2 * positions[j][0]) / 3, (positions[i][1] + 2 * positions[j][1]) / 3
				get_position[(j, i, x)] = (positions[i][0] + 2 * positions[j][0]) / 3, (positions[i][1] + 2 * positions[j][1]) / 3
		else:
			for x in get_by_mask(graph.labels[i] ^ graph.labels[j]):
				get_position[(i, j, x)] = (positions[i][0] + positions[j][0]) / 2, (positions[i][1] + positions[j][1]) / 2
				get_position[(j, i, x)] = (positions[i][0] + positions[j][0]) / 2, (positions[i][1] + positions[j][1]) / 2

	for t in trs:
		i, j, k = sorted(t)
		ijk = (i, j, k)
		m = (graph.labels[i] | graph.labels[j] | graph.labels[k]) & (~(graph.labels[i] & graph.labels[j] & graph.labels[k]))
		ez = (graph.labels[i] | graph.labels[j] | graph.labels[k]) & (~m)
		for l in get_by_mask(ez):
			draws[l].polygon(list(map(int, [*(positions[i] + positions[j] + positions[k])])), fill=colors[l]+(255,))
		for l in get_by_mask(m):
			guys = []
			for x in ijk:
				y = graph.labels[x]
				if (y & (1 << l)) > 0:
					guys.append(x)
			without = [x for x in ijk if x not in guys]
			if len(guys) == 1:
				draws[l].polygon(list(map(int, [*(positions[guys[0]] + get_position[(guys[0], without[0], l)] + get_position[(guys[0], without[1], l)])])),
								 fill=colors[l]+(255,))
				draw.line(list(map(int, [*(get_position[(guys[0], without[0], l)] + get_position[(guys[0], without[1], l)])])), fill=colors[l]+(255,), width=3)
			else:
				draws[l].polygon(list(map(int, [*(positions[guys[0]] + positions[guys[1]] + get_position[(guys[1], without[0], l)] + get_position[(guys[0], without[0], l)])])),
								 fill=colors[l]+(255,))
				draw.line(list(map(int, [*(get_position[(guys[0], without[0], l)] + get_position[(guys[1], without[0], l)])])), fill=colors[l]+(255,), width=3)

	for i in range(colors_cnt):
		imgs[i].putalpha(sum(colors[i]) // 3)
		im.paste(imgs[i], (0, 0), imgs[i])
	im.save(filename, "PNG")

def create_diagram(args, label=""):
	letters = set()
	for x in args:
		letters |= set(x)
	if len(letters) == 0:
		print("Empty diagram!")
		return

	letters = sorted(list(letters))

	masks = [0]
	for x in args:
		mask = 0
		for c in x:
			mask |= 1 << letters.index(c)
		masks.append(mask)
	masks.sort()
	# all these sorts are just for debug and visual clarity purposes

	g = Graph(masks)
	for i in range(len(masks)):
		for j in range(i):
			tmp = masks[i] ^ masks[j]
			if (tmp & (tmp - 1)) == 0:
				g.add_edge(i, j)

	positions = spring(g) # https://arxiv.org/pdf/1201.3011.pdf, algorithm 1

	edges = list(g.get_edges())
	intersect_cnt = [0] * len(edges)
	for i in range(len(edges)):
		for j in range(i):
			if len(set(edges[i]) | set(edges[j])) < 4:
				continue
			if intersects(*map(positions.__getitem__, edges[i] + edges[j])):
				intersect_cnt[i] += 1
				intersect_cnt[j] += 1
	indices = list(range(len(edges)))
	indices.sort(key=lambda x: intersect_cnt[x], reverse=True)
	removed = [False] * len(edges)
	for i in indices:
		if intersect_cnt[i] == 0:
			continue
		removed[i] = True
		for j in range(len(edges)):
			if len(set(edges[i]) | set(edges[j])) < 4 or removed[j]:
				continue
			if intersects(*map(positions.__getitem__, edges[i] + edges[j])):
				intersect_cnt[i] -= 1
				intersect_cnt[j] -= 1

	for i in range(len(removed)):
		if removed[i]:
			g.inc[edges[i][0]][edges[i][1]] = False
			g.inc[edges[i][1]][edges[i][0]] = False

	# now we achieve connectivity
	edges = list(g.get_edges())
	possible_edges = []
	for i in range(g.n):
		for j in range(i):
			if not g.are_connected(i, j):
				possible_edges.append((j, i))
	possible_edges.sort(key=lambda x: popcount(masks[x[0]] ^ masks[x[1]]))
	for e in possible_edges:
		if can_draw_edge(edges, positions, e):
			edges.append(e)
			g.add_edge(e[0], e[1])

	# gonna find a triangulation
	positions = normalize(positions, (500, 500), (400, 400))
	on_side = 4
	g.add_vertices(on_side * 4, 0)
	masks += [0] * on_side * 4
	for i in range(4):
		for j in range(on_side):
			x, y = 0, j * 500 / on_side
			for _ in range(i):
				x, y = y, 499 - x
			positions.append((x, y))
	edges = list(g.get_edges())
	tr_edges = dict()
	for i in range(g.n):
		for j in range(i):
			if not g.inc[i][j]:
				if can_draw_edge(edges, positions, (j, i)):
					g.inc[i][j] = g.inc[j][i] = True
					edges.append((j, i))
					tr_edges[(j, i)] = (masks[j] & (~masks[i]), masks[i] & (~masks[j]))

	
	triangles = list(filter(lambda x: len(x) == 3, get_faces(g, positions)))

	generate_pic(positions, g, "graph_%s.png" % label, norm=False) # just to look at it
	generate_diagram(g, triangles, tr_edges, positions, "diagram_%s.png" % label)

if __name__ == "__main__":
	args = sys.argv[1:]
	if args == []:
		s = "a"; create_diagram(s.split(), s.replace(' ', '_'))
		s = "a b"; create_diagram(s.split(), s.replace(' ', '_'))
		s = "a b ab"; create_diagram(s.split(), s.replace(' ', '_'))
		s = "a ab"; create_diagram(s.split(), s.replace(' ', '_'))
		s = "a b c"; create_diagram(s.split(), s.replace(' ', '_'))
		s = "a b c ac bc"; create_diagram(s.split(), s.replace(' ', '_'))
		s = "a b c ab bc ac abc"; create_diagram(s.split(), s.replace(' ', '_'))
		s = "a b c d ab ac ad bc bd cd abc abd acd bcd abcd"; create_diagram(s.split(), s.replace(' ', '_'))
	else:
		create_diagram(args, "input")
