import networkx as nx
from networkx.readwrite import json_graph
from stanza.models.common.doc import Sentence

from tuw_nlp.graph.graph import Graph, UnconnectedGraphError
from tuw_nlp.graph.utils import preprocess_edge_alto


class UDGraph(Graph):
    @staticmethod
    def from_json(data):
        sen = Sentence(data["stanza_sen"])
        text, tokens = data["text"], data["tokens"]
        G = json_graph.adjacency_graph(data["graph"])
        assert G.graph["type"] == "ud"
        assert G.graph["text"] == text
        assert G.graph["tokens"] == tokens
        ud_graph = UDGraph(sen, text, tokens)
        ud_graph.G = G
        return ud_graph

    def __init__(self, sen, text=None, tokens=None):

        graph = self.convert_to_networkx(sen)
        super(UDGraph, self).__init__(graph=graph, text=text, tokens=tokens, type="ud")
        self.ud_graph = sen

    def to_json(self):
        data = {
            "graph": json_graph.adjacency_data(self.G),
            "text": self.text,
            "tokens": self.tokens,
            "stanza_sen": self.ud_graph.to_dict(),
        }
        return data

    def __str__(self):
        return f"UDGraph({self.str_nodes()})"

    def __repr__(self):
        return self.__str__()

    def str_nodes(self):
        return " ".join(
            f"{i}_{data['name']}" for i, data in sorted(self.G.nodes(data=True))
        )

    def copy(self):
        new_graph = UDGraph(self.ud_graph, text=self.text, tokens=self.tokens)
        new_graph.G = self.G.copy()
        return new_graph

    def remove_graph(self, other):
        g_to_remove = other.G.copy()
        tok_ids_to_remove = {
            data.get("token_id") for node, data in g_to_remove.nodes(data=True)
        }
        self.tokens = [
            tok if i + 1 not in tok_ids_to_remove else None
            for i, tok in enumerate(self.tokens)
        ]
        self.text = None
        self.G.remove_nodes_from(g_to_remove)

    def subgraph(self, nodes, handle_unconnected=None):
        # print("main graph:", self.str_nodes(), "nodes:", nodes)
        H = self.G.subgraph(nodes)
        if not nx.is_weakly_connected(H):
            if handle_unconnected is None:
                raise UnconnectedGraphError(
                    f"subgraph induced by nodes {nodes} is not connected and handle_unconnected is not specified"
                )
            elif handle_unconnected == "shortest_path":
                new_nodes = set(
                    nodes
                )  # a copy of the original nodes parameter to expand
                components = [
                    list(node_set) for node_set in nx.weakly_connected_components(H)
                ]
                src = components[0][0]  # a dedicated node in a dedicated component
                G_u = (
                    self.G.to_undirected()
                )  # an undirected version of G to search for shortest paths
                for comp in components[1:]:
                    path = nx.shortest_path(G_u, src, comp[0])
                    # print(f'shortest path between {src} and {comp[0]}: {path}')
                    for node in path:
                        if node not in new_nodes:
                            new_nodes.add(node)

                return self.subgraph(new_nodes)

            else:
                raise ValueError(
                    f"unknown value of handle_unconnected: {handle_unconnected}"
                )

        tok_ids_to_keep = {data.get("token_id") for node, data in H.nodes(data=True)}
        new_tokens = [
            tok if i + 1 in tok_ids_to_keep else None
            for i, tok in enumerate(self.tokens)
        ]
        # print("main tokens:", self.tokens)
        # print("H.nodes:", H.nodes(data=True))
        # print("new tokens:", new_tokens)

        H.graph["tokens"] = new_tokens
        new_graph = UDGraph(self.ud_graph, text=None, tokens=new_tokens)
        # print("new graph tokens:", new_graph.tokens)
        new_graph.G = H
        # print(
        #    "new graph (subgraph):",
        #    new_graph.str_nodes(),
        #    "nodes:",
        #    new_graph.G.nodes(),
        # )
        return new_graph

    def pos_edge_graph(self, vocab):
        H = self.G.copy()
        words = set()
        for u, v, d in H.edges(data=True):
            d["color"] = d["color"].lower()
        for node, data in self.G.nodes(data=True):
            word = data["name"]
            if word in words:
                word = f"{word}_"
            words.add(word)
            leaf_node_id = vocab.get_id(word, allow_new=True)
            H.add_node(leaf_node_id, name=word)
            H.add_edge(node, leaf_node_id, color=data["upos"])
            nx.set_node_attributes(H, {node: {"name": ""}})
        return Graph.from_networkx(H)

    def subgraph_from_tok_ids(self, ids):
        nodes = {
            node
            for node, data in self.G.nodes(data=True)
            if data.get("token_id") in ids
        }
        return self.subgraph(nodes)

    def convert_to_networkx(self, sen):
        """convert dependency-parsed stanza Sentence to nx.DiGraph"""
        G = nx.DiGraph()
        for word in sen.to_dict():
            if isinstance(word["id"], (list, tuple)):
                # token representing an mwe, e.g. "vom" ~ "von dem"
                continue
            G.add_node(
                word["id"], name=word["lemma"], token_id=word["id"], upos=word["upos"]
            )
            if word["deprel"] == "root":
                G.add_node(word["head"], name="root", upos="ROOT")
            G.add_edge(word["head"], word["id"])
            G[word["head"]][word["id"]].update(
                {"color": preprocess_edge_alto(word["deprel"])}
            )

        return G
