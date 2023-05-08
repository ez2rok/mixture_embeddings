"""Modified from: https://github.com/RyanLiGod/hnsw-python/blob/master/hnsw.py"""

from heapq import heapify, heappop, heappush, heapreplace, nlargest, nsmallest
from math import log2
from operator import itemgetter
from random import random

import torch
import numpy as np
from tqdm import tqdm
from icecream import ic

# local files
from src.util.distance_functions.distance_matrix import DISTANCE_MATRIX
from src.util.distance_functions.distance_functions import DISTANCE_TORCH
from src.util.ml_and_math.loss_functions import AverageMeter


class HNSW:
    # self._graphs[level][i] contains a {j: dist} dictionary,
    # where j is a neighbor of i and dist is distance

    def l2_distance(self, a, b):
        return np.linalg.norm(a - b)

    def cosine_distance(self, a, b):
        try:
            denom = np.linalg.norm(a) * (np.linalg.norm(b))
            num = np.dot(a, b)
            return num / denom
        except ValueError:
            print(a)
            print(b)

    def _distance(self, x, y):
        result =  self.distance_func(x, [y])
        return result[0]

    def vectorized_distance_(self, x, ys):
        # pprint.pprint([self.distance_func(x, y) for y in ys])
        return [self.distance_func(x, y) for y in ys]
        

    def __init__(self, num_neighbors, distance_type, device, m=5, ef=200, m0=None, heuristic=True, vectorized=False):
        self.data = []
        if distance_type == "euclidean":
            # l2 distance
            # distance_func = self.l2_distance
            distance_func = DISTANCE_TORCH[distance_type]
        elif distance_type == "cosine":
            # cosine distance
            distance_func = self.cosine_distance
        elif distance_type == 'hyperbolic':
            distance_func = DISTANCE_TORCH[distance_type]
        else:
            raise TypeError('Please check your distance type!')

        self.distance_func = distance_func

        if vectorized:
            self.distance = self._distance
            self.vectorized_distance = distance_func
        else:
            self.distance = distance_func
            self.vectorized_distance = self.vectorized_distance_
            # self.distance = DISTANCE_TORCH[distance_type]

        self._m = m
        self._ef = ef
        self._m0 = 2 * m if m0 is None else m0
        self._level_mult = 1 / log2(m)
        self._graphs = []
        self._enter_point = None

        self._select = (
            self._select_heuristic if heuristic else self._select_naive)
        
        self.num_neighbors = num_neighbors
        self.device = device
        self.do_count = False
        self.count = 0
        self.num_comparisons = AverageMeter()

    def add(self, elem, ef=None):

        if ef is None:
            ef = self._ef

        distance = self.distance
        data = self.data
        graphs = self._graphs
        point = self._enter_point
        m = self._m

        # level at which the element will be inserted
        level = int(-log2(random()) * self._level_mult) + 1
        # print("level: %d" % level)

        # elem will be at data[idx]
        idx = len(data)
        data.append(elem)

        if point is not None:  # the HNSW is not empty, we have an entry point
            dist = distance(elem, data[point])
            # for all levels in which we dont have to insert elem,
            # we search for the closest neighbor
            for layer in reversed(graphs[level:]):
                point, dist = self._search_graph_ef1(elem, point, dist, layer)
            # at these levels we have to insert elem; ep is a heap of entry points.
            ep = [(-dist, point)]
            # pprint.pprint(ep)
            layer0 = graphs[0]
            for layer in reversed(graphs[:level]):
                level_m = m if layer is not layer0 else self._m0
                # navigate the graph and update ep with the closest
                # nodes we find
                ep = self._search_graph(elem, ep, layer, ef)
                # insert in g[idx] the best neighbors
                layer[idx] = layer_idx = {}
                self._select(layer_idx, ep, level_m, layer, heap=True)
                # assert len(layer_idx) <= level_m
                # insert backlinks to the new node
                for j, dist in layer_idx.items():
                    self._select(layer[j], (idx, dist), level_m, layer)
                    # assert len(g[j]) <= level_m
                # assert all(e in g for _, e in ep)
        for i in range(len(graphs), level):
            # for all new levels, we create an empty graph
            graphs.append({idx: {}})
            self._enter_point = idx

    def balanced_add(self, elem, ef=None):
        if ef is None:
            ef = self._ef

        distance = self.distance
        data = self.data
        graphs = self._graphs
        point = self._enter_point
        m = self._m
        m0 = self._m0

        idx = len(data)
        data.append(elem)

        if point is not None:
            dist = distance(elem, data[point])
            pd = [(point, dist)]
            # pprint.pprint(len(graphs))
            for layer in reversed(graphs[1:]):
                point, dist = self._search_graph_ef1(elem, point, dist, layer)
                pd.append((point, dist))
            for level, layer in enumerate(graphs):
                # print('\n')
                # pprint.pprint(layer)
                level_m = m0 if level == 0 else m
                candidates = self._search_graph(
                    elem, [(-dist, point)], layer, ef)
                layer[idx] = layer_idx = {}
                self._select(layer_idx, candidates, level_m, layer, heap=True)
                # add reverse edges
                for j, dist in layer_idx.items():
                    self._select(layer[j], [idx, dist], level_m, layer)
                    assert len(layer[j]) <= level_m
                if len(layer_idx) < level_m:
                    return
                if level < len(graphs) - 1:
                    if any(p in graphs[level + 1] for p in layer_idx): 
                        return
                point, dist = pd.pop()
        graphs.append({idx: {}})
        self._enter_point = idx

    def search(self, q, k=None, ef=None):
        """Find the k points closest to q."""

        distance = self.distance
        graphs = self._graphs
        point = self._enter_point

        if ef is None:
            ef = self._ef

        if point is None:
            raise ValueError("Empty graph")

        dist = distance(q, self.data[point])
        # look for the closest neighbor from the top to the 2nd level
        for layer in reversed(graphs[1:]):
            point, dist = self._search_graph_ef1(q, point, dist, layer)
        # look for ef neighbors in the bottom level
        ep = self._search_graph(q, [(-dist, point)], graphs[0], ef)

        if k is not None:
            ep = nlargest(k, ep)
        else:
            ep.sort(reverse=True)
            
        # return [(idx, -md) for md, idx in ep]
            
        nn_distances = []
        nn_idxs = []
        for md, idx in ep:
            nn_distances.append(-md)
            nn_idxs.append(idx)
        
        nn_idxs = torch.tensor(nn_idxs, device=self.device)
        nn_distances = torch.tensor(nn_distances, device=self.device)
        return nn_idxs, nn_distances

    def _search_graph_ef1(self, q, entry, dist, layer):
        """Equivalent to _search_graph when ef=1."""

        vectorized_distance = self.vectorized_distance
        data = self.data

        best = entry
        best_dist = dist
        candidates = [(dist, entry)]
        visited = set([entry])

        while candidates:
            dist, c = heappop(candidates)
            if dist > best_dist:
                break
            edges = [e for e in layer[c] if e not in visited]
            visited.update(edges)
            edge_data = [data[e] for e in edges]
            dists = vectorized_distance(q, edge_data)
            
            if self.do_count:
                self.count += len(edge_data)
            
            for e, dist in zip(edges, dists):
                if dist < best_dist:
                    best = e
                    best_dist = dist
                    heappush(candidates, (dist, e))
                    # break

        return best, best_dist

    def _search_graph(self, q, ep, layer, ef):

        vectorized_distance = self.vectorized_distance
        data = self.data

        candidates = [(-mdist, p) for mdist, p in ep]
        heapify(candidates)
        visited = set(p for _, p in ep)

        while candidates:
            dist, c = heappop(candidates)
            mref = ep[0][0]
            if dist > -mref:
                break
            # pprint.pprint(layer[c])
            edges = [e for e in layer[c] if e not in visited]
            visited.update(edges)
            edge_data = [data[e] for e in edges]
            dists = vectorized_distance(q, edge_data)
            
            if self.do_count:
                self.count += len(edge_data)
            
            for e, dist in zip(edges, dists):
                mdist = -dist
                if len(ep) < ef:
                    heappush(candidates, (dist, e))
                    heappush(ep, (mdist, e))
                    mref = ep[0][0]
                elif mdist > mref:
                    heappush(candidates, (dist, e))
                    heapreplace(ep, (mdist, e))
                    mref = ep[0][0]

        return ep

    def _select_naive(self, d, to_insert, m, layer, heap=False):

        if not heap:
            idx, dist = to_insert
            assert idx not in d
            if len(d) < m:
                d[idx] = dist
            else:
                max_idx, max_dist = max(d.items(), key=itemgetter(1))
                if dist < max_dist:
                    del d[max_idx]
                    d[idx] = dist
            return

        assert not any(idx in d for _, idx in to_insert)
        to_insert = nlargest(m, to_insert)  # smallest m distances
        unchecked = m - len(d)
        assert 0 <= unchecked <= m
        to_insert, checked_ins = to_insert[:unchecked], to_insert[unchecked:]
        to_check = len(checked_ins)
        if to_check > 0:
            checked_del = nlargest(to_check, d.items(), key=itemgetter(1))
        else:
            checked_del = []
        for md, idx in to_insert:
            d[idx] = -md
        zipped = zip(checked_ins, checked_del)
        for (md_new, idx_new), (idx_old, d_old) in zipped:
            if d_old <= -md_new:
                break
            del d[idx_old]
            d[idx_new] = -md_new
            assert len(d) == m

    def _select_heuristic(self, d, to_insert, m, g, heap=False):

        nb_dicts = [g[idx] for idx in d]

        def prioritize(idx, dist):
            return any(nd.get(idx, float('inf')) < dist for nd in nb_dicts), dist, idx

        if not heap:
            idx, dist = to_insert
            to_insert = [prioritize(idx, dist)]
        else:
            to_insert = nsmallest(m, (prioritize(idx, -mdist)
                                      for mdist, idx in to_insert))

        assert len(to_insert) > 0
        assert not any(idx in d for _, _, idx in to_insert)

        unchecked = m - len(d)
        assert 0 <= unchecked <= m
        to_insert, checked_ins = to_insert[:unchecked], to_insert[unchecked:]
        to_check = len(checked_ins)
        if to_check > 0:
            checked_del = nlargest(to_check, (prioritize(idx, dist)
                                              for idx, dist in d.items()))
        else:
            checked_del = []
        for _, dist, idx in to_insert:
            d[idx] = dist
        zipped = zip(checked_ins, checked_del)
        for (p_new, d_new, idx_new), (p_old, d_old, idx_old) in zipped:
            if (p_old, d_old) <= (p_new, d_new):
                break
            del d[idx_old]
            d[idx_new] = d_new
            assert len(d) == m

    def __getitem__(self, idx):

        for g in self._graphs:
            try:
                yield from g[idx].items()
            except KeyError:
                return
            
    def fit(self, X_ref):
        for x in tqdm(X_ref, desc='Building HNSW graph'):
            self.add(x)
            
    def kneighbors(self, X_qry):
        
        self.do_count = True
        nn_distances_list = []
        nn_idxs_list = []
        
        for q in tqdm(X_qry, desc='Querying'):
            self.count = 0
            nn_idxs, nn_distances = self.search(q, k=self.num_neighbors)
            self.num_comparisons.update(self.count, n=1)
            
            nn_idxs_list.append(nn_idxs)
            nn_distances_list.append(nn_distances)
        
        self.do_count = False
        nn_idxs = torch.vstack(nn_idxs_list)
        nn_distances = torch.vstack(nn_distances_list)
        return nn_distances, nn_idxs
    
    
    
    
    
    
    
# Modified from: https://zilliz.com/blog/hierarchical-navigable-small-worlds-HNSW

from bisect import insort
from heapq import heapify, heappop, heappush

import numpy as np

from ._base import _BaseIndex


class HNSW(_BaseIndex):

    def __init__(self, L=5, mL=0.62, efc=10, distance_type='euclidean'):
        self._L = L
        self._mL = mL
        self._efc = efc
        self._index = [[] for _ in range(L)]
        
        self.distance = DISTANCE_TORCH[distance_type]
        
        self.num_comparisons = AverageMeter()
        self.do_count = False
        self.count = 0

    def _search_layer(self, graph, entry, query, ef=1):

        # best = (np.linalg.norm(graph[entry][0] - query), entry)
        best = (self.distance(graph[entry][0], query), entry)
        self.count += 1 & self.do_count # update count

        nns = [best]
        visit = set(best)  # set of visited nodes
        candid = [best]  # candidate nodes to insert into nearest neighbors
        heapify(candid)

        # find top-k nearest neighbors
        while candid:
            cv = heappop(candid)

            if nns[-1][0] > cv[0]:
                break

            # loop through all nearest neighbors to the candidate vector
            for e in graph[cv[1]][1]:
                # d = np.linalg.norm(graph[e][0] - query)
                d = self.distance(graph[e][0], query)
                self.count += 1 & self.do_count # update count
                
                if (d, e) not in visit:
                    visit.add((d, e))

                    # push only "better" vectors into candidate heap
                    if d < nns[-1][0] or len(nns) < ef:
                        heappush(candid, (d, e))
                        insort(nns, (d, e))
                        if len(nns) > ef:
                            nns.pop()

        return nns

    def search(self, query, ef=1):

        # if the index is empty, return an empty list
        if not self._index[0]:
            return []

        best_v = 0  # set the initial best vertex to the entry point
        for graph in self._index:
            best_d, best_v = self._search_layer(graph, best_v, query, ef=1)[0]
            if graph[best_v][2]:
                best_v = graph[best_v][2]
            else:
                return self._search_layer(graph, best_v, query, ef=ef)

    def _get_insert_layer(self):
        # ml is a multiplicative factor used to normalize the distribution
        l = -int(np.log(np.random.random()) * self._mL)
        return min(l, self._L-1)

    def insert(self, vec, efc=10):

        # if the index is empty, insert the vector into all layers and return
        if not self._index[0]:
            i = None
            for graph in self._index[::-1]:
                graph.append((vec, [], i))
                i = 0
            return

        l = self._get_insert_layer()

        start_v = 0
        for n, graph in enumerate(self._index):

            # perform insertion for layers [l, L) only
            if n < l:
                _, start_v = self._search_layer(graph, start_v, vec, ef=1)[0]
            else:
                node = (vec, [], len(self._index[n+1]) if n < self._L-1 else None)
                nns = self._search_layer(graph, start_v, vec, ef=efc)
                for nn in nns:
                    node[1].append(nn[1])  # outbound connections to NNs
                    graph[nn[1]][1].append(len(graph))  # inbound connections to node
                graph.append(node)

            # set the starting vertex to the nearest neighbor in the next layer
            start_v = graph[start_v][2]
           
    def fit(self, X_ref):
        for x in tqdm(X_ref, desc='Building HNSW graph'):
            self.add(x) 
            
    def kneighbors(self, X_qry):
        
        self.do_count = True
        nn_distances_list = []
        nn_idxs_list = []
        
        for query in tqdm(X_qry, desc='Querying'):
            self.count = 0
            nns = self.search(query) # list of (distance, entries)
            self.num_comparisons.update(self.count, n=1)
            
            nn_idxs_list.append(nn_idxs)
            nn_distances_list.append(nn_distances)
        
        self.do_count = False
        nn_idxs = torch.vstack(nn_idxs_list)
        nn_distances = torch.vstack(nn_distances_list)
        return nn_distances, nn_idxs