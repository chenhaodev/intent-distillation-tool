"""
Intent Tag Distiller
Builds hierarchical intent taxonomies through iterative distillation
Based on easy-dataset's tag distillation workflow
"""
import logging
from typing import Dict, List, Optional, Any

from ..llm.client import LLMClient
from ..llm.prompts.distill_intent_tags import build_distill_intent_tags_prompt

logger = logging.getLogger(__name__)


class IntentNode:
    """Represents a node in the intent taxonomy tree"""

    def __init__(self, name: str, number: str = "", parent: Optional['IntentNode'] = None):
        self.name = name
        self.number = number  # e.g., "1.2.3"
        self.parent = parent
        self.children: List['IntentNode'] = []

    @property
    def full_name(self) -> str:
        """Get numbered name (e.g., '1.2 Password Reset')"""
        if self.number:
            return f"{self.number} {self.name}"
        return self.name

    @property
    def path(self) -> str:
        """Get full path (e.g., 'Support -> Account -> Password Reset')"""
        if self.parent:
            return f"{self.parent.path} -> {self.name}"
        return self.name

    @property
    def numbered_path(self) -> str:
        """Get numbered path (e.g., 'Support -> 1 Account -> 1.2 Password Reset')"""
        if self.parent:
            return f"{self.parent.numbered_path} -> {self.full_name}"
        return self.full_name

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "name": self.name,
            "number": self.number,
            "full_name": self.full_name,
            "path": self.path,
            "numbered_path": self.numbered_path,
            "children": [child.to_dict() for child in self.children]
        }


class IntentTagDistiller:
    """Distill hierarchical intent taxonomies"""

    def __init__(self, llm_client: LLMClient, language: str = "en"):
        """
        Initialize intent tag distiller

        Args:
            llm_client: LLM client instance
            language: Language for prompts ('zh' or 'en')
        """
        self.llm_client = llm_client
        self.language = language
        self.root: Optional[IntentNode] = None

    def distill_tags(
        self,
        parent_intent: str,
        count: int,
        parent_node: Optional[IntentNode] = None,
        existing_tags: Optional[List[str]] = None
    ) -> List[IntentNode]:
        """
        Distill sub-intent tags for a parent intent

        Args:
            parent_intent: Parent intent name
            count: Number of sub-tags to generate
            parent_node: Parent node in the tree
            existing_tags: Existing sibling tags to avoid

        Returns:
            List of IntentNode objects
        """
        logger.info(f"Distilling {count} sub-intents for: {parent_intent}")

        # Build intent path
        intent_path = parent_node.numbered_path if parent_node else parent_intent

        # Build prompt
        prompt = build_distill_intent_tags_prompt(
            parent_intent=parent_intent,
            count=count,
            intent_path=intent_path,
            existing_tags=existing_tags,
            language=self.language
        )

        # Get LLM response
        try:
            response = self.llm_client.get_json_response(prompt)

            # Parse tags
            if isinstance(response, list):
                tag_names = response
            elif isinstance(response, dict) and "tags" in response:
                tag_names = response["tags"]
            else:
                raise ValueError(f"Unexpected response format: {response}")

            # Create IntentNode objects
            nodes = []
            for tag_name in tag_names:
                # Extract number and name
                parts = tag_name.strip().split(" ", 1)
                if len(parts) == 2:
                    number, name = parts
                else:
                    number = ""
                    name = tag_name.strip()

                node = IntentNode(name=name, number=number, parent=parent_node)
                if parent_node:
                    parent_node.children.append(node)
                nodes.append(node)

            logger.info(f"Generated {len(nodes)} sub-intents: {[n.full_name for n in nodes]}")
            return nodes

        except Exception as e:
            logger.error(f"Error distilling intent tags: {e}")
            raise

    def build_taxonomy(
        self,
        root_topic: str,
        levels: int,
        tags_per_level: int,
        existing_root: Optional[IntentNode] = None
    ) -> IntentNode:
        """
        Build complete intent taxonomy tree

        Args:
            root_topic: Root topic/intent
            levels: Number of hierarchy levels
            tags_per_level: Number of tags to generate per level
            existing_root: Existing root node to extend (optional)

        Returns:
            Root IntentNode with full taxonomy tree
        """
        logger.info(f"Building intent taxonomy: topic={root_topic}, levels={levels}, tags_per_level={tags_per_level}")

        # Create or use existing root
        if existing_root:
            root = existing_root
        else:
            root = IntentNode(name=root_topic)
            self.root = root

        # Build tree level by level
        current_level_nodes = [root]

        for level in range(1, levels + 1):
            logger.info(f"Building level {level}/{levels}")
            next_level_nodes = []

            for parent_node in current_level_nodes:
                # Get existing children names to avoid duplicates
                existing_names = [child.name for child in parent_node.children]

                # Distill sub-intents
                try:
                    child_nodes = self.distill_tags(
                        parent_intent=parent_node.name,
                        count=tags_per_level,
                        parent_node=parent_node,
                        existing_tags=existing_names if existing_names else None
                    )
                    next_level_nodes.extend(child_nodes)

                except Exception as e:
                    logger.error(f"Failed to distill tags for {parent_node.full_name}: {e}")
                    continue

            current_level_nodes = next_level_nodes

            if not current_level_nodes:
                logger.warning(f"No nodes generated at level {level}, stopping")
                break

        logger.info(f"Taxonomy building complete. Total nodes: {self._count_nodes(root)}")
        return root

    def _count_nodes(self, node: IntentNode) -> int:
        """Count total nodes in tree"""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count

    def get_leaf_intents(self, node: Optional[IntentNode] = None) -> List[IntentNode]:
        """
        Get all leaf intent nodes (nodes without children)

        Args:
            node: Starting node (uses root if not provided)

        Returns:
            List of leaf IntentNode objects
        """
        if node is None:
            node = self.root

        if not node:
            return []

        if not node.children:
            return [node]

        leaves = []
        for child in node.children:
            leaves.extend(self.get_leaf_intents(child))

        return leaves

    def export_tree(self, node: Optional[IntentNode] = None) -> Dict[str, Any]:
        """
        Export taxonomy tree to dictionary

        Args:
            node: Starting node (uses root if not provided)

        Returns:
            Dictionary representation of tree
        """
        if node is None:
            node = self.root

        if not node:
            return {}

        return node.to_dict()

    def export_flat_list(self, node: Optional[IntentNode] = None) -> List[Dict[str, Any]]:
        """
        Export flattened list of all intents

        Args:
            node: Starting node (uses root if not provided)

        Returns:
            List of intent dictionaries
        """
        if node is None:
            node = self.root

        if not node:
            return []

        result = [{
            "name": node.name,
            "number": node.number,
            "full_name": node.full_name,
            "path": node.path,
            "numbered_path": node.numbered_path,
            "level": self._get_level(node)
        }]

        for child in node.children:
            result.extend(self.export_flat_list(child))

        return result

    def _get_level(self, node: IntentNode) -> int:
        """Get depth level of node"""
        level = 0
        current = node
        while current.parent:
            level += 1
            current = current.parent
        return level
