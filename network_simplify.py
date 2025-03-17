from __future__ import annotations
"""Simplify, correct, and consolidate spatial graph nodes and edges."""

import networkx as nx
import tools.networkX_manipulate as ntx
import os, json

from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Iterator

edgelist_file = "Hybrid_Network.edgelist"

def _is_endpoint(
    G: nx.MultiDiGraph,
    node: int,
    node_attrs_include: Iterable[str] | None,
    edge_attrs_differ: Iterable[str] | None,
) -> bool:
    """
    A copied function from https://github.com/gboeing/osmnx/blob/main/osmnx/simplification.py

    Determine if a node is a true endpoint of an edge.

    Return True if the node is a "true" endpoint of an edge in the network,
    otherwise False. OpenStreetMap data includes many nodes that exist only as
    geometric vertices to allow ways to curve. `node` is a true edge endpoint
    if it satisfies at least 1 of the following 5 rules:

    1) It is its own neighbor (ie, it self-loops).

    2) Or, it has no incoming edges or no outgoing edges (ie, all its incident
    edges are inbound or all its incident edges are outbound).

    3) Or, it does not have exactly two neighbors and degree of 2 or 4.

    4) Or, if `node_attrs_include` is not None and it has one or more of the
    attributes in `node_attrs_include`.

    5) Or, if `edge_attrs_differ` is not None and its incident edges have
    different values than each other for any of the edge attributes in
    `edge_attrs_differ`.

    Parameters
    ----------
    G
        Input graph.
    node
        The ID of the node to check.
    node_attrs_include
        Node attribute names for relaxing the strictness of endpoint
        determination. A node is always an endpoint if it possesses one or
        more of the attributes in `node_attrs_include`.
    edge_attrs_differ
        Edge attribute names for relaxing the strictness of endpoint
        determination. A node is always an endpoint if its incident edges have
        different values than each other for any attribute in
        `edge_attrs_differ`.

    Returns
    -------
    endpoint
        True if node is an endpoint, otherwise False.
    """
    neighbors = set(list(G.predecessors(node)) + list(G.successors(node)))
    n = len(neighbors)
    d = G.degree(node)

    # RULE 1
    # if the node appears in its list of neighbors, it self-loops: this is
    # always an endpoint
    if node in neighbors:
        return True

    # RULE 2
    # if node has no incoming edges or no outgoing edges, it is an endpoint
    if G.out_degree(node) == 0 or G.in_degree(node) == 0:
        return True

    # RULE 3
    # else, if it does NOT have 2 neighbors AND either 2 or 4 directed edges,
    # it is an endpoint. either it has 1 or 3+ neighbors, in which case it is
    # a dead-end or an intersection of multiple streets or it has 2 neighbors
    # but 3 degree (indicating a change from oneway to twoway) or more than 4
    # degree (indicating a parallel edge) and thus is an endpoint
    if not ((n == 2) and (d in {2, 4})):  # noqa: PLR2004
        return True

    # RULE 4
    # non-strict mode: does it contain an attr denoting that it is an endpoint
    if node_attrs_include is not None and len(set(node_attrs_include) & G.nodes[node].keys()) > 0:
        return True

    # RULE 5
    # non-strict mode: do its incident edges have different attr values? for
    # each attribute to check, collect the attribute's values in all inbound
    # and outbound edges. if there is more than 1 unique value then this node
    # is an endpoint
    if edge_attrs_differ is not None:
        for attr in edge_attrs_differ:
            in_values = {v for _, _, v in G.in_edges(node, data=attr, keys=False)}
            out_values = {v for _, _, v in G.out_edges(node, data=attr, keys=False)}
            if len(in_values | out_values) > 1:
                return True

    # if none of the preceding rules passed, then it is not an endpoint
    return False

def _build_path(
    G: nx.MultiDiGraph,
    endpoint: int,
    endpoint_successor: int,
    endpoints: set[int],
) -> list[int]:
    """
    A copied function from https://github.com/gboeing/osmnx/blob/main/osmnx/simplification.py

    Build a path of nodes from one endpoint node to next endpoint node.

    Parameters
    ----------
    G
        Input graph.
    endpoint
        The endpoint node from which to start the path.
    endpoint_successor
        The successor of endpoint through which the path to the next endpoint
        will be built.
    endpoints
        The set of all nodes in the graph that are endpoints.

    Returns
    -------
    path
        The first and last items in the resulting path list are endpoint
        nodes, and all other items are interstitial nodes that can be removed
        subsequently.
    """
    # start building path from endpoint node through its successor
    path = [endpoint, endpoint_successor]

    # for each successor of the endpoint's successor
    for this_successor in G.successors(endpoint_successor):
        successor = this_successor
        if successor not in path:
            # if this successor is already in the path, ignore it, otherwise add
            # it to the path
            path.append(successor)
            while successor not in endpoints:
                # find successors (of current successor) not in path
                successors = [n for n in G.successors(successor) if n not in path]

                # 99%+ of the time there will be only 1 successor: add to path
                if len(successors) == 1:
                    successor = successors[0]
                    path.append(successor)

                # handle relatively rare cases or OSM digitization quirks
                elif len(successors) == 0:
                    if endpoint in G.successors(successor):
                        # we have come to the end of a self-looping edge, so
                        # add first node to end of path to close it and return
                        return [*path, endpoint]

                    # otherwise, this can happen due to OSM digitization error
                    # where a one-way street turns into a two-way here, but
                    # duplicate incoming one-way edges are present
                    msg = f"Unexpected simplify pattern handled near {successor}"
                    raise Warning(msg)
                    return path
                else:  # pragma: no cover
                    # if successor has >1 successors, then successor must have
                    # been an endpoint because you can go in 2 new directions.
                    # this should never occur in practice
                    msg = f"Impossible simplify pattern failed near {successor}."
                    raise ValueError(msg)

            # if this successor is an endpoint, we've completed the path
            return path

    # if endpoint_successor has no successors not already in the path, return
    # the current path: this is usually due to a digitization quirk on OSM
    return path

def _get_paths_to_simplify(
    G: nx.MultiDiGraph,
    node_attrs_include: Iterable[str] | None,
    edge_attrs_differ: Iterable[str] | None,
) -> Iterator[list[int]]:
    """
    A copied function from https://github.com/gboeing/osmnx/blob/main/osmnx/simplification.py

    Generate all the paths to be simplified between endpoint nodes.

    The path is ordered from the first endpoint, through the interstitial nodes,
    to the second endpoint.

    Parameters
    ----------
    G
        Input graph.
    node_attrs_include
        Node attribute names for relaxing the strictness of endpoint
        determination. A node is always an endpoint if it possesses one or
        more of the attributes in `node_attrs_include`.
    edge_attrs_differ
        Edge attribute names for relaxing the strictness of endpoint
        determination. A node is always an endpoint if its incident edges have
        different values than each other for any attribute in
        `edge_attrs_differ`.

    Yields
    ------
    path_to_simplify
    """
    # first identify all the nodes that are endpoints
    endpoints = {n for n in G.nodes if _is_endpoint(G, n, node_attrs_include, edge_attrs_differ)}
    msg = f"Identified {len(endpoints):,} edge endpoints"
    print(msg)

    # for each endpoint node, look at each of its successor nodes
    for endpoint in endpoints:
        for successor in G.successors(endpoint):
            if successor not in endpoints:
                # if endpoint node's successor is not an endpoint, build path
                # from the endpoint node, through the successor, and on to the
                # next endpoint node
                yield _build_path(G, endpoint, successor, endpoints)

def _remove_rings(
    G: nx.MultiDiGraph,
    node_attrs_include: Iterable[str] | None,
    edge_attrs_differ: Iterable[str] | None,
) -> nx.MultiDiGraph:
    """
    A copied function from https://github.com/gboeing/osmnx/blob/main/osmnx/simplification.py

    Remove all graph components that consist only of a single chordless cycle.

    This identifies all connected components in the graph that consist only of
    a single isolated self-contained ring, and removes them from the graph.

    Parameters
    ----------
    G
        Input graph.
    node_attrs_include
        Node attribute names for relaxing the strictness of endpoint
        determination. A node is always an endpoint if it possesses one or
        more of the attributes in `node_attrs_include`.
    edge_attrs_differ
        Edge attribute names for relaxing the strictness of endpoint
        determination. A node is always an endpoint if its incident edges have
        different values than each other for any attribute in
        `edge_attrs_differ`.

    Returns
    -------
    G
        Graph with all chordless cycle components removed.
    """
    to_remove = set()
    for wcc in nx.weakly_connected_components(G):
        if not any(_is_endpoint(G, n, node_attrs_include, edge_attrs_differ) for n in wcc):
            to_remove.update(wcc)
    G.remove_nodes_from(to_remove)
    return G

def simplify_graph(  # noqa: C901, PLR0912
    G: nx.MultiDiGraph,
    *,
    node_attrs_include: Iterable[str] | None = None,
    edge_attrs_differ: Iterable[str] | None = None,
    remove_rings: bool = True,
    edge_attr_aggs: dict[str, Any] | None = None,
) -> tuple [nx.MultiDiGraph, dict]:
    """
    A copied function from https://github.com/gboeing/osmnx/blob/main/osmnx/simplification.py

    Simplify a graph's topology by removing interstitial nodes.

    This simplifies the graph's topology by removing all nodes that are not
    intersections or dead-ends, by creating an edge directly between the end
    points that encapsulate them while retaining the full geometry of the
    original edges, saved as a new `geometry` attribute on the new edge.

    Note that only simplified edges receive a `geometry` attribute. Some of
    the resulting consolidated edges may comprise multiple OSM ways, and if
    so, their unique attribute values are stored as a list. Optionally, the
    simplified edges can receive a `merged_edges` attribute that contains a
    list of all the `(u, v)` node pairs that were merged together.

    Use the `node_attrs_include` or `edge_attrs_differ` parameters to relax
    simplification strictness. For example, `edge_attrs_differ=["osmid"]` will
    retain every node whose incident edges have different OSM IDs. This lets
    you keep nodes at elbow two-way intersections (but be aware that sometimes
    individual blocks have multiple OSM IDs within them too). You could also
    use this parameter to retain nodes where sidewalks or bike lanes begin/end
    in the middle of a block. Or for example, `node_attrs_include=["highway"]`
    will retain every node with a "highway" attribute (regardless of its
    value), even if it does not represent a street junction.

    Parameters
    ----------
    G
        Input graph.
    node_attrs_include
        Node attribute names for relaxing the strictness of endpoint
        determination. A node is always an endpoint if it possesses one or
        more of the attributes in `node_attrs_include`.
    edge_attrs_differ
        Edge attribute names for relaxing the strictness of endpoint
        determination. A node is always an endpoint if its incident edges have
        different values than each other for any attribute in
        `edge_attrs_differ`.
    remove_rings
        If True, remove any graph components that consist only of a single
        chordless cycle (i.e., an isolated self-contained ring).
    edge_attr_aggs
        Allows user to aggregate edge segment attributes when simplifying an
        edge. Keys are edge attribute names and values are aggregation
        functions to apply to these attributes when they exist for a set of
        edges being merged. Edge attributes not in `edge_attr_aggs` will
        contain the unique values across the merged edge segments. If None,
        defaults to `{"weight": sum}`.

    Returns
    -------
    Gs
        Topologically simplified graph, with a new `geometry` attribute on
        each simplified edge.
    merged_edges_dict
        a dict for merged edges
    """
    if G.graph.get("simplified"):  # pragma: no cover
        msg = "This graph has already been simplified, cannot simplify it again."
        raise ValueError(msg)

    msg = "Begin topologically simplifying the graph..."
    print(msg)

    # default edge segment attributes to aggregate upon simplification
    if edge_attr_aggs is None:
        edge_attr_aggs = {"weight": sum}

    # make a copy to not mutate original graph object caller passed in
    G = G.copy()
    initial_node_count = len(G)
    initial_edge_count = len(G.edges)
    all_nodes_to_remove = []
    all_edges_to_add = []
    merged_edges_dict = {}

    # generate each path that needs to be simplified
    for path in _get_paths_to_simplify(G, node_attrs_include, edge_attrs_differ):
        # add the interstitial edges we're removing to a list so we can retain
        # their spatial geometry
        merged_edges = []
        path_attributes: dict[str, Any] = {}
        for u, v in zip(path[:-1], path[1:]):
            # keep track of the edges that were merged
            if not merged_edges:
                merged_edges.extend((u, v))
            else:
                merged_edges.append(v)

            # there should rarely be multiple edges between interstitial nodes
            # usually happens if OSM has duplicate ways digitized for just one
            # street... we will keep only one of the edges (see below)
            edge_count = G.number_of_edges(u, v)
            if edge_count != 1:
                msg = f"Found {edge_count} edges between {u} and {v} when simplifying"
                print(msg)

            # get edge between these nodes: if multiple edges exist between
            # them (see above), we retain only one in the simplified graph
            # We can't assume that there exists an edge from u to v
            # with key=0, so we get a list of all edges from u to v
            # and just take the first one.
            edge_data = next(iter(G.get_edge_data(u, v).values()))
            for attr in edge_data:
                if attr in path_attributes:
                    # if this key already exists in the dict, append it to the
                    # value list
                    path_attributes[attr].append(edge_data[attr])
                else:
                    # if this key doesn't already exist, set the value to a list
                    # containing the one value
                    path_attributes[attr] = [edge_data[attr]]

        # consolidate the path's edge segments' attribute values
        for attr_name, attr_values in path_attributes.items():
            if attr_name in edge_attr_aggs:
                # if this attribute's values must be aggregated, do so now
                agg_func = edge_attr_aggs[attr_name]
                path_attributes[attr_name] = agg_func(attr_values)
            elif len(set(attr_values)) == 1:
                # if there's only 1 unique value, keep that single value
                path_attributes[attr_name] = attr_values[0]
            else:
                # otherwise, if there are multiple uniques, keep one of each
                path_attributes[attr_name] = list(set(attr_values))

        merged_edges_dict[f"{path[0]}{path[-1]}"] = ",".join(merged_edges)

        # add the nodes and edge to their lists for processing at the end
        all_nodes_to_remove.extend(path[1:-1])
        all_edges_to_add.append(
            {"origin": path[0], "destination": path[-1], "attr_dict": path_attributes},
        )

    # for each edge to add in the list we assembled, create a new edge between
    # the origin and destination
    for edge in all_edges_to_add:
        G.add_edge(edge["origin"], edge["destination"], **edge["attr_dict"])

    # finally remove all the interstitial nodes between the new edges
    G.remove_nodes_from(set(all_nodes_to_remove))

    if remove_rings:
        G = _remove_rings(G, node_attrs_include, edge_attrs_differ)

    # mark the graph as having been simplified
    G.graph["simplified"] = True

    msg = (
        f"Simplified graph: {initial_node_count:,} to {len(G):,} nodes, "
        f"{initial_edge_count:,} to {len(G.edges):,} edges"
    )
    print(msg)
    return G, merged_edges_dict

# Example usage:
if __name__ == "__main__":
    # Open a directed multigraph example.
    G = ntx.open_edgelist(edgelist_file, True)
    print("Number of edges loaded:", G.number_of_edges())

    G_simplified, merged_dict = simplify_graph(G)
    
    print("\nEdges After simplification:" , G_simplified.number_of_edges())
    result_path = f"{edgelist_file.split('.')[0]}_simplified.edgelist"
    nx.write_weighted_edgelist(G_simplified, result_path)
    result_path = f"{edgelist_file.split('.')[0]}_simplified.json"
    with open(result_path, 'w') as f:
        json.dump(merged_dict, f, ensure_ascii=False)