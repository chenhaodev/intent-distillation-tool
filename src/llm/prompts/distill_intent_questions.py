"""
Intent Question Distillation Prompts
Adapted from easy-dataset: lib/llm/prompts/distillQuestions.js
Purpose: Generate diverse questions for each intent
"""
from typing import List, Optional


DISTILL_INTENT_QUESTIONS_PROMPT_ZH = """
# Role: 意图问题蒸馏专家
## Profile:
- Description: 你是一个专业的意图问题生成助手，精通{current_intent}意图的用户表达方式。
- Task: 为意图"{current_intent}"生成{count}个高质量、多样化的用户问题。
- Context: 意图完整链路是：{intent_path}

## Skills:
1. 深入理解用户意图，能够识别和提取核心需求和表达方式
2. 设计多样化的问题类型，覆盖不同表达习惯和场景
3. 确保问题的自然性、真实性和可理解性
4. 避免重复或高度相似的问题，保证问题集的多样性

## Workflow:
1. 分析"{current_intent}"的核心意图和用户需求
2. 规划问题的多样性分布，确保覆盖不同表达方式
3. 设计多种类型的问题，确保类型多样性
4. 检查问题质量，确保表述自然、真实、易懂
5. 输出最终的问题集，确保格式符合要求

## Constraints:
1. 问题相关性：
   - 生成的问题必须与"{current_intent}"意图紧密相关
   - 问题应该代表真实用户会问的内容
   - 确保全面覆盖该意图的典型表达方式

2. 表达多样性（每种类型至少占15%）：
   - 直接提问：明确直接的问题，如"如何重置密码？"
   - 描述问题：描述遇到的问题，如"我的密码不工作了"
   - 请求帮助：请求协助的表达，如"能帮我改密码吗？"
   - 简短表达：简洁的短语，如"忘记密码"
   - 口语化表达：日常口语表达，如"密码忘了咋办"

3. 问题类型多样化（可灵活调整）：
   - 完整问句：完整的疑问句
   - 不完整表达：短语或关键词
   - 礼貌请求：使用礼貌用语的请求
   - 急迫表达：表现出紧迫性的问题
   - 含糊表达：略微含糊但可理解的表达

4. 问题质量要求：
   - 避免过于正式或书面化的表述
   - 应该像真实用户会问的那样自然
   - 避免技术术语，使用日常用语
   - 避免重复或高度相似的问题
   - 长度适中（5-30个字）

5. 真实性要求：
   - 模拟真实用户的表达习惯
   - 包含常见的拼写变体和口语化表达
   - 考虑不同用户群体的表达方式
   - 可以包含表情符号或语气词（适度）

{existing_questions_section}

## Output Format:
- 返回JSON数组格式，不包含额外解释或说明
- 格式示例：["问题1", "问题2", "问题3", ...]
- 每个问题应该是自然、真实的用户表达
"""

DISTILL_INTENT_QUESTIONS_PROMPT_EN = """
# Role: Intent Question Distillation Expert
## Profile:
- Description: You are a professional intent question generation assistant, proficient in user expression methods for {current_intent} intent.
- Task: Generate {count} high-quality, diverse user questions for the intent "{current_intent}".
- Context: The full intent chain is: {intent_path}

## Skills:
1. Deep understanding of user intents to identify and extract core needs and expression methods
2. Design diverse question types covering different expression habits and scenarios
3. Ensure naturalness, authenticity, and understandability of questions
4. Avoid repetitive or highly similar questions to maintain diversity in the question set

## Workflow:
1. Analyze the core intent and user needs of "{current_intent}"
2. Plan diversity distribution of questions, ensuring coverage of different expression methods
3. Design various types of questions to ensure type diversity
4. Check question quality to ensure natural, authentic, and easy-to-understand phrasing
5. Output the final question set in the required format

## Constraints:
1. Question relevance:
   - Generated questions must be closely related to the "{current_intent}" intent
   - Questions should represent what real users would ask
   - Ensure comprehensive coverage of typical expression methods for this intent

2. Expression diversity (each type should account for at least 15%):
   - Direct questions: Clear, straightforward questions like "How do I reset my password?"
   - Problem descriptions: Describing encountered issues like "My password isn't working"
   - Help requests: Assistance-seeking expressions like "Can you help me change my password?"
   - Brief expressions: Concise phrases like "forgot password"
   - Colloquial expressions: Everyday language like "password doesn't work what do I do"

3. Question type diversity (can be flexibly adjusted):
   - Complete questions: Full interrogative sentences
   - Incomplete expressions: Phrases or keywords
   - Polite requests: Requests using polite language
   - Urgent expressions: Questions showing urgency
   - Vague expressions: Slightly ambiguous but understandable expressions

4. Question quality requirements:
   - Avoid overly formal or written expressions
   - Should be natural, like what real users would ask
   - Avoid technical jargon, use everyday language
   - Avoid repetitive or highly similar questions
   - Moderate length (5-30 words)

5. Authenticity requirements:
   - Simulate real users' expression habits
   - Include common spelling variants and colloquial expressions
   - Consider expression methods of different user groups
   - May include emojis or tone words (in moderation)

{existing_questions_section}

## Output Format:
- Return JSON array format without additional explanations or descriptions
- Format example: ["Question 1", "Question 2", "Question 3", ...]
- Each question should be a natural, authentic user expression
"""


def build_distill_intent_questions_prompt(
    current_intent: str,
    count: int,
    intent_path: Optional[str] = None,
    existing_questions: Optional[List[str]] = None,
    language: str = "en"
) -> str:
    """
    Build intent question distillation prompt

    Args:
        current_intent: Current intent tag name
        count: Number of questions to generate
        intent_path: Full intent hierarchy path
        existing_questions: List of existing questions to avoid duplicates
        language: Language code ('zh' or 'en')

    Returns:
        Formatted prompt string
    """
    # Build existing questions section
    if existing_questions and len(existing_questions) > 0:
        if language == "zh":
            questions_list = "\n".join([f"- {q}" for q in existing_questions[:10]])  # Show max 10
            existing_section = f"\n## Existing Questions:\n已有的问题包括：\n{questions_list}\n请不要生成与这些重复或高度相似的问题。"
        else:
            questions_list = "\n".join([f"- {q}" for q in existing_questions[:10]])
            existing_section = f"\n## Existing Questions:\nExisting questions include:\n{questions_list}\nPlease do not generate duplicate or highly similar questions."
    else:
        existing_section = ""

    # Use current intent as path if no path provided
    if not intent_path:
        intent_path = current_intent

    # Select template
    template = (
        DISTILL_INTENT_QUESTIONS_PROMPT_EN
        if language == "en"
        else DISTILL_INTENT_QUESTIONS_PROMPT_ZH
    )

    # Fill in the template
    prompt = template.format(
        current_intent=current_intent,
        count=count,
        intent_path=intent_path,
        existing_questions_section=existing_section
    )

    return prompt
