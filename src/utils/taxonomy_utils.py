"""
Utility functions for working with intent taxonomy trees
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.distillers.intent_tag_distiller import IntentNode


def export_taxonomy_text(node: 'IntentNode', indent: int = 0) -> str:
    """
    Export taxonomy as formatted text

    Args:
        node: Root or current IntentNode to export
        indent: Current indentation level

    Returns:
        Formatted taxonomy text with hierarchical structure
    """
    lines = []
    prefix = "  " * indent
    lines.append(f"{prefix}- {node.name}")

    for child in node.children:
        lines.append(export_taxonomy_text(child, indent + 1))

    return "\n".join(lines)
