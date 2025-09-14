import random

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def order_edges(edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Orders the edges of a graph in a consistent manner."""

    for idx, edge in enumerate(edges):
        if edge[0] > edge[1]:
            edges[idx] = (edge[1], edge[0])

    return sorted(edges)

def star_graph_edges(num_nodes: int, num_edges: int) -> tuple[list[tuple[int, int]], int]:
    """Generates a star graph with a specified number of nodes and edges. The graph must
    have at least num_nodes - 1 edges. Additional edges are randomly created between leaf
    nodes.

    Parameters:
        num_nodes: The number of nodes in the graph.
        num_edges: The total number of edges in the graph. Must be at least num_nodes - 1.
    Returns:
        edges: A list of edges represented as tuples (node1, node2).
        hub_node: The index of the hub node.
    """

    if num_edges < num_nodes - 1:
        raise ValueError("Not enough edges to form a star graph")
    if num_edges > (num_nodes * (num_nodes - 1)) // 2:
        raise ValueError("Too many edges for the number of nodes")
    
    num_leaf_edges = num_edges - (num_nodes - 1)

    hub_node = random.randint(0, num_nodes - 1)
    # Create hub edges
    hub_edges = [(hub_node, i) for i in range(num_nodes) if i != hub_node]

    possible_leaf_edges = []
    for i in range(num_nodes - 1):
        for j in range(i + 1, num_nodes):
            if i != j and i != hub_node and j != hub_node:
                possible_leaf_edges.append((i, j))

    # Randomly select leaf edges
    leaf_edges = random.sample(possible_leaf_edges, num_leaf_edges)

    edges = hub_edges + leaf_edges
    edges = order_edges(edges)

    return edges, hub_node

def star_graph(
        num_nodes: int, 
        num_edges: int, 
        img_size: int = 256, 
        drift: float = 0, 
        **draw_kwargs
        ) -> np.ndarray:
    """Generates and draws a star graph with a specified number of nodes and edges. The graph must
    have at least num_nodes - 1 edges. Additional edges are randomly created between leaf
    nodes.

    Parameters:
        num_nodes: The number of nodes in the graph.
        num_edges: The total number of edges in the graph. Must be at least num_nodes - 1.
        img_size: The size of the output image in pixels.
        drift: The maximum random displacement applied to each node's position.
        draw_kwargs: Additional keyword arguments for the draw_graph function.
    Returns:
        img: A numpy array representing the drawn graph image.
    """

    edges, hub_node = star_graph_edges(num_nodes, num_edges)

    graph = nx.Graph(edges)

    pos = nx.shell_layout(
        graph,
        nlist=[ [hub_node], [i for i in range(num_nodes) if i != hub_node] ]
        )
    
    # Normalize between 0 and img_size
    for idx, p in pos.items():
        dx, dy = random.uniform(-drift, drift), random.uniform(-drift, drift)
        pos[idx] = (img_size*(p[0]+0.5) + dx, img_size*(p[1]+0.5) + dy)

    img = draw_graph(graph, pos, img_size, **draw_kwargs)

    return img, edges

def num_grid_graph_edges(num_rows: int, num_cols: int) -> int:
    """Calculates the number of edges in a full grid graph with the given number of rows and columns."""

    return 2*(num_rows-1)*(num_cols-1) + (num_rows-1) + (num_cols-1)

def grid_graph_edges(num_rows: int, num_cols: int, num_edges: int) -> list[tuple[int, int]]:
    """Generates a grid graph with a specified number of rows, columns, and edges. First, a full 
    grid graph is created, and then edges are randomly removed to reach the desired number of edges. 
    `num_edges` must be at most `num_grid_graph_edges(num_rows, num_cols)`.

    Parameters:
        num_rows: The number of rows in the grid.
        num_cols: The number of columns in the grid.
        num_edges: The total number of edges in the graph, after edge removal.
    Returns:
        edges: A list of edges represented as tuples (node1, node2).
    """

    num_edges_full_grid = num_grid_graph_edges(num_rows, num_cols)

    if num_edges > num_edges_full_grid:
        raise ValueError("Too many edges for a grid graph")

    num_edges_to_remove = num_edges_full_grid - num_edges

    # Create full grid graph edges
    edges = []
    for row in range(num_rows-1):
        for col in range(num_cols-1):
            source = row*num_cols + col
            edges.append((source, source+1))
            edges.append((source, (row+1)*num_cols + col))

        edges.append((row*num_cols + num_cols - 1, (row+1)*num_cols + num_cols - 1))

    for col in range(num_cols-1):
        edges.append(((num_rows-1)*num_cols + col, (num_rows-1)*num_cols + col + 1))

    # Randomly remove edges
    edges_to_remove = random.sample(edges, num_edges_to_remove)
    edges = list(set(edges) - set(edges_to_remove))

    # Shuffle node indices
    mapping = list(range(num_rows * num_cols))
    random.shuffle(mapping)
    edges = [(mapping[e[0]], mapping[e[1]]) for e in edges]

    edges = order_edges(edges)

    return edges, mapping

def grid_graph(
        num_rows: int, 
        num_cols: int, 
        num_edges: int, 
        img_size: int = 256, 
        drift: float = 0, 
        **draw_kwargs
        ) -> np.ndarray:
    """Generates and draws a grid graph with a specified number of rows, columns, and edges. First, a full 
    grid graph is created, and then edges are randomly removed to reach the desired number of edges. 
    `num_edges` must be at most `num_grid_graph_edges(num_rows, num_cols)`.

    Parameters:
        num_rows: The number of rows in the grid.   
        num_cols: The number of columns in the grid.
        num_edges: The total number of edges in the graph, after edge removal.
        img_size: The size of the output image in pixels.
        drift: The maximum random displacement applied to each node's position.
        draw_kwargs: Additional keyword arguments for the draw_graph function.
    Returns:
        img: A numpy array representing the drawn graph image.
    """

    edges, ordering = grid_graph_edges(num_rows, num_cols, num_edges)
    # We need to create the graph with `add_nodes_from` instead of `nx.Graph(edges)` because
    # the grid graph might have nodes with degree 0, which are not included in the edgelist
    graph = nx.Graph()
    graph.add_nodes_from(range(num_rows * num_cols))
    graph.add_edges_from(edges)

    # Create grid positions
    pos = {}
    for row in range(num_rows):
        for col in range(num_cols):
            idx = row * num_cols + col
            pos[ordering[idx]] = (col, -row)

    # Normalize between 0 and img_size
    for idx, p in pos.items():
        dx, dy = random.uniform(-drift, drift), random.uniform(-drift, drift)
        pos[idx] = (img_size*(p[0]/(num_cols-1)) + dx, img_size*(p[1]/(num_rows-1)) + dy)

    img = draw_graph(graph, pos, img_size, **draw_kwargs)

    return img, edges

def draw_graph(
        graph: nx.Graph, 
        pos: dict,
        img_size: int = 256,
        edge_color: str = "white",
        edge_width: int = 2,
        node_color: str = "red",
        node_size: int = 300,
        text_size: int = 12,
        text_color: str = "k"
        ) -> np.ndarray:
    """Draws a graph using NetworkX and Matplotlib.
    Parameters:
        graph: A NetworkX graph object.
        pos: A dictionary mapping node indices to (x, y) coordinates.
        img_size: The size of the output image in pixels.
        edge_color: The color for all the edges (e.g., 'blue', '#FF5733').
        edge_width: The width of the edges.
        node_color: The color of the nodes (e.g., 'red', '#00FF00').
        node_size: The size of the nodes.
        text_size: The font size of the index labels.
        text_color: The color of the index labels.
    Returns:
        img: A numpy array representing the drawn graph image.
    """

    num_nodes = len(graph)

    labels = {i: str(i) for i in range(num_nodes)}

    fig = plt.figure(facecolor="black", figsize=(img_size / 100, img_size / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("black")  # Black background
    ax.axis("off")

    nx.draw_networkx_edges(
        graph, 
        pos, 
        width=edge_width,
        edge_color=edge_color,
        ax=ax
        )

    nx.draw_networkx_nodes(
        graph, 
        pos, 
        node_size=node_size, 
        node_color=node_color, 
        linewidths=None, 
        edgecolors=None, 
        ax=ax
        )

    nx.draw_networkx_labels(
        graph, 
        pos,
        labels,
        font_size=text_size,
        font_color=text_color,
        ax=ax
    )

    ax.set_aspect("equal", adjustable="box")
    #ax.set_xlim((0, img_size))
    #ax.set_ylim((0, img_size))
    fig.canvas.draw()
    img = np.asarray(fig.canvas.renderer.buffer_rgba())[..., :3]
    plt.close(fig)

    return img
