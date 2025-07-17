import numpy as np
import matplotlib.pyplot as plt

def draw_curves_and_nodes(
        curves, 
        nodes, 
        curve_color="white", 
        curve_width=2, 
        node_color="red", 
        node_size=10, 
        text_size=8, 
        text_color="k",
        img_size=256
        ):
    """
    Draws curves and nodes from the given data.

    Args:
        curves: A list where each element is a list of (x, y) points.
        nodes: A list of (x, y) coordinates.
        curve_color: The color for all the curves (e.g., 'blue', '#FF5733').
        curve_width: The width of the curves.
        node_color: The color of the nodes (e.g., 'red', '#00FF00').
        node_size: The size of the nodes.
        text_size: The font size of the index labels.
        text_color: The color of the index labels.
    """

    fig = plt.figure(facecolor="black", figsize=(img_size / 100, img_size / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("black")
    ax.axis("off")

    for curve in curves:
        if len(curve) < 2:
            continue

        ax.plot(*curve.T, color=curve_color, linewidth=curve_width, zorder=1)

    # Draw all the points at once using a scatter plot
    ax.scatter(*nodes.T, s=node_size**2, color=node_color, zorder=2)

    # Loop through the points to add the index as text
    for index, (px, py) in enumerate(nodes):
        ax.text(px, py, index, color=text_color,
                fontsize=text_size, ha='center', va='center', zorder=3)
        
    ax.set_aspect("equal", adjustable="box")

    fig.canvas.draw()
    img = np.asarray(fig.canvas.renderer.buffer_rgba())[..., :3]
    plt.close(fig)

    #fig.savefig("my_shape.png", facecolor="black", bbox_inches="tight", pad_inches=0)

    return img