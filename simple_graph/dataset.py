import json
from PIL import Image
from scipy.special import factorial
from .graph_generation import star_graph, grid_graph

def num_possible_star_graphs(num_nodes, num_edges):
    """Calculates the number of possible star graphs with a given number of nodes and edges."""
    
    num_leaf_edges = num_edges - (num_nodes - 1)

    # Number of possible edges between leaf nodes
    m = (num_nodes-1)*(num_nodes-2)/2
    # Choose num_leaf_edges from m
    possible_edges = factorial(m)/(factorial(num_leaf_edges)*factorial(m-num_leaf_edges))
    # Consider all possible hubs
    possible_graphs = num_nodes * possible_edges
    
    return possible_graphs

def num_possible_grid_graphs(num_rows, num_cols, num_edges):
    """Calculates the number of possible grid graphs with a given number of rows, columns, and edges."""

    num_nodes = num_rows * num_cols
    num_edges_full_grid = 2*(num_rows-1)*(num_cols-1) + (num_rows-1) + (num_cols-1)
    num_edges_to_remove = num_edges_full_grid - num_edges

    possible_removals = factorial(num_edges_full_grid)/(factorial(num_edges_to_remove)*factorial(num_edges))

    # Number of unique full grid graphs
    num_symmetries = 4 if num_rows != num_cols else 8
    num_graphs = factorial(num_nodes) / num_symmetries

    possible_graphs = possible_removals * num_graphs

    return possible_graphs

def generate_dataset(output_folder, num_items_per_class, graph_props):
    """Generates a dataset of star and grid graphs, saving images and edge lists to the specified folder.
    Parameters:
        output_folder: The folder where images and edge lists will be saved.
        num_items_per_class: The number of unique graphs to generate for each class (star and grid).
        graph_props: A dictionary containing properties for graph generation, including:
            - num_nodes: Number of nodes for star graphs.
            - num_edges: Number of edges for both star and grid graphs.
            - num_rows: Number of rows for grid graphs.
            - num_cols: Number of columns for grid graphs.
            - img_size (optional): Size of the output images in pixels (default is 256).
            - drift (optional): Maximum random displacement applied to each node's position (default is 0).
    """

    num_nodes = graph_props["num_nodes"]
    num_edges = graph_props["num_edges"]
    num_rows = graph_props["num_rows"]
    num_cols = graph_props["num_cols"]
    img_size = graph_props.get("img_size", 256)
    drift = graph_props.get("drift", 0)


    edgelists = {}
    graph_set = set()
    counter = 0
    while len(graph_set) < num_items_per_class:
        img, edges = star_graph(num_nodes, num_edges, img_size=img_size, drift=drift)
        graph_set.add(tuple(edges))

        Image.fromarray(img).save(f"{output_folder}/images/s{counter:04d}.png")
        edgelists[f"s{counter:04d}"] = edges

        counter += 1

    graph_set = set()
    counter = 0
    while len(graph_set) < num_items_per_class:
        img, edges = grid_graph(num_rows, num_cols, num_edges, img_size=img_size, drift=drift)
        graph_set.add(tuple(edges))

        Image.fromarray(img).save(f"{output_folder}/images/g{counter:04d}.png")
        edgelists[f"g{counter:04d}"] = edges

        counter += 1

    json.dump(edgelists, open(f"{output_folder}/edgelists.json", "w"))