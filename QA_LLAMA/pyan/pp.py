from pyan import create_callgraph
# from graphviz import Digraph

callgraph = create_callgraph(__file__)  # Analyze current script (app.py)

# Optional: Visualize using graphviz
if __name__ == '__main__':  # Only if running as main script
    

    dot = Digraph()
    for node in callgraph.nodes:
        dot.node(node.name, label=node.name)
    for edge in callgraph.edges:
        dot.edge(edge.source, edge.target)

    dot.render('app_callgraph.png')  # Output as image
from pyan import create_callgraph

callgraph = create_callgraph('RAG.py')



# ... (visualization code from app.py)

# dot.render('RAG_callgraph.png')  # Output as image
