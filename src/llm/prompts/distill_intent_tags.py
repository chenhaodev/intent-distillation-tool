"""
Intent Tag Distillation Prompts
Adapted from easy-dataset: lib/llm/prompts/distillTags.js
Purpose: Generate hierarchical intent taxonomies
"""
from typing import List, Optional


DISTILL_INTENT_TAGS_PROMPT_ZH = """
# Role: 意图标签蒸馏专家
## Profile:
- Description: 你是一个专业的意图标签生成助手，专长于为特定主题创建细分的子意图标签体系。
- Task: 为主题"{parent_intent}"生成{count}个专业的子意图标签。
- Context: 意图完整链路是：{intent_path}

## Skills:
1. 深入理解意图领域，识别其核心子类别和专业细分方向
2. 设计简洁明确的标签命名，确保表意准确且易于理解
3. 规划标签间的差异化分布，避免重叠或模糊边界
4. 确保标签的实用性，能够有效支撑后续的问题生成工作

## Workflow:
1. **意图分析**：深入理解"{parent_intent}"的领域范围和核心要素
2. **子类识别**：识别该意图下的主要子类别和专业方向
3. **标签设计**：为每个子类别设计简洁明确的标签名称
4. **序号分配**：根据层级正确分配标签序号
5. **质量检查**：确保标签间区分明显，覆盖不同方面

## Constraints:
1. 标签内容要求：
   - 生成的标签应该是"{parent_intent}"领域内的专业子类别或子意图
   - 标签之间应该有明显的区分，覆盖不同的方面
   - 标签应该具有实用性，能够作为问题生成的基础
   - 标签应该代表用户的实际意图，而不是技术术语

2. 标签格式要求：
   - 每个标签应该简洁、明确，通常为2-8个字
   - 标签应该是名词或名词短语，表示用户的目的或需求
   - 标签必须有明显的序号

3. 序号规则：
   - 若父标签有序号（如"1 账户管理"），子标签格式为："1.1 密码重置"、"1.2 账户删除"、"1.3 个人信息"等
   - 若父标签无序号（如"客户支持"），说明当前在生成顶级标签，子标签格式为："1 技术支持"、"2 账户管理"、"3 产品咨询"等

4. 避重要求：
   - 不与已有标签重复或高度相似

{existing_tags_section}

## Examples of Good Intent Tags:
**Good**: "密码重置", "订单查询", "退款申请", "功能咨询"
**Bad**: "点击按钮", "输入信息", "查看页面" (too technical/not intent-focused)

## Output Format:
- 仅返回JSON数组格式，不包含额外解释或说明
- 格式示例：["序号 标签1", "序号 标签2", "序号 标签3", ...]
- 例如：["1.1 密码重置", "1.2 账户删除", "1.3 个人信息修改"]
"""

DISTILL_INTENT_TAGS_PROMPT_EN = """
# Role: Intent Tag Distillation Expert
## Profile:
- Description: You are a professional intent tag generation assistant, specializing in creating refined sub-intent tag systems for specific topics.
- Task: Generate {count} professional sub-intent tags for the topic "{parent_intent}".
- Context: The full intent chain is: {intent_path}

## Skills:
1. Deeply understand intent domains and identify core sub-categories and professional subdivisions
2. Design concise and clear tag naming that ensures accurate meaning and easy understanding
3. Plan differentiated distribution among tags to avoid overlap or blurred boundaries
4. Ensure tag practicality to effectively support subsequent question generation work

## Workflow:
1. **Intent Analysis**: Deeply understand the domain scope and core elements of "{parent_intent}"
2. **Sub-category Identification**: Identify major sub-categories and professional directions under this intent
3. **Tag Design**: Design concise and clear tag names for each sub-category
4. **Numbering Assignment**: Correctly assign tag numbers according to hierarchy
5. **Quality Check**: Ensure clear distinctions between tags, covering different aspects

## Constraints:
1. Tag content requirements:
   - Generated tags should be professional sub-categories or sub-intents within the "{parent_intent}" domain
   - Tags should be clearly distinguishable, covering different aspects
   - Tags should be practical and serve as a basis for question generation
   - Tags should represent actual user intents, not technical terms

2. Tag format requirements:
   - Each tag should be concise and clear, typically 2-8 words
   - Tags should be nouns or noun phrases representing user goals or needs
   - Tags must have explicit numbering

3. Numbering rules:
   - If parent tag has numbering (e.g., "1 Account Management"), sub-tags format: "1.1 Password Reset", "1.2 Account Deletion", "1.3 Personal Info", etc.
   - If parent tag is unnumbered (e.g., "Customer Support"), indicating top-level tag generation, sub-tags format: "1 Technical Support", "2 Account Management", "3 Product Inquiries", etc.

4. Duplication avoidance:
   - Do not duplicate or highly resemble existing tags

{existing_tags_section}

## Examples of Good Intent Tags:
**Good**: "Password Reset", "Order Inquiry", "Refund Request", "Feature Question"
**Bad**: "Click Button", "Enter Information", "View Page" (too technical/not intent-focused)

## Output Format:
- Return only JSON array format without additional explanations or descriptions
- Format example: ["Number Tag 1", "Number Tag 2", "Number Tag 3", ...]
- Example: ["1.1 Password Reset", "1.2 Account Deletion", "1.3 Personal Info Update"]
"""


def build_distill_intent_tags_prompt(
    parent_intent: str,
    count: int,
    intent_path: Optional[str] = None,
    existing_tags: Optional[List[str]] = None,
    language: str = "en"
) -> str:
    """
    Build intent tag distillation prompt

    Args:
        parent_intent: Parent intent name (e.g., "Customer Support")
        count: Number of sub-intent tags to generate
        intent_path: Full intent hierarchy path (e.g., "Customer Support -> Account Management")
        existing_tags: List of existing sibling tags to avoid duplicates
        language: Language code ('zh' or 'en')

    Returns:
        Formatted prompt string
    """
    # Build existing tags section
    if existing_tags and len(existing_tags) > 0:
        if language == "zh":
            existing_section = f"\n## Existing Tags:\n已有的子标签包括：{', '.join(existing_tags)}\n请不要生成与这些重复或相似的标签。"
        else:
            existing_section = f"\n## Existing Tags:\nExisting sub-tags include: {', '.join(existing_tags)}\nPlease do not generate duplicate or similar tags."
    else:
        existing_section = ""

    # Use parent intent as path if no path provided
    if not intent_path:
        intent_path = parent_intent

    # Select template
    template = (
        DISTILL_INTENT_TAGS_PROMPT_EN
        if language == "en"
        else DISTILL_INTENT_TAGS_PROMPT_ZH
    )

    # Fill in the template
    prompt = template.format(
        parent_intent=parent_intent,
        count=count,
        intent_path=intent_path,
        existing_tags_section=existing_section
    )

    return prompt
