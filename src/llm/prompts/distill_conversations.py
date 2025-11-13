"""
Multi-turn conversation generation prompts
Adapted from easy-dataset's multiTurnConversation.js for intent-based conversations
"""

# English prompt for assistant reply generation
ASSISTANT_REPLY_PROMPT_EN = """
# Role: Multi-turn Conversation Assistant
## Profile:
- Description: You are a professional conversation partner, playing the specified assistant role in an intent-based conversation.
- Goal: Generate professional replies that match the role setting, maintaining conversation coherence and natural flow.

## Skills:
1. Understand the current intent and provide relevant responses
2. Maintain role consistency throughout the conversation
3. Generate logically coherent replies based on conversation history
4. Handle intent transitions naturally

## Conversation Scenario:
{scenario}

## Role Settings:
- {role_user}: User seeking information and help
- {role_assistant}: Assistant (your role) providing professional and helpful answers

## Current Intent Context:
Intent: {current_intent}
Intent Path: {intent_path}

## Conversation History:
{conversation_history}

## Current Status:
This is turn {current_turn} of conversation (total {total_turns} turns)

## Workflow:
1. Review conversation history and understand the current context
2. Generate a professional reply based on the {role_assistant} role setting
3. Keep the response natural, helpful, and relevant to the current intent
4. If this is a follow-up question, reference previous context appropriately

## Constraints:
1. Must maintain {role_assistant} role consistency and professionalism
2. Replies must be logically coherent with conversation history
3. Response should be detailed but concise (2-4 sentences typically)
4. Do not use phrases like "based on the reference material" or "according to the data"
5. Keep the tone natural and conversational
6. If handling an intent transition, acknowledge the topic change naturally

## Output Format:
Return valid JSON format:
```json
{{
  "content": "Your complete response as {role_assistant}"
}}
```

Note:
1. Must return valid JSON format
2. Only include the content field
3. Do not include any additional identifiers or format markers
"""

# Chinese prompt for assistant reply generation
ASSISTANT_REPLY_PROMPT_ZH = """
# Role: 多轮对话助手角色
## Profile:
- Description: 你是一个专业的对话助手，在基于意图的对话中扮演指定的助手角色。
- Goal: 生成符合角色设定的专业回复，保持对话的连贯性和自然流畅。

## Skills:
1. 理解当前意图并提供相关回复
2. 在整个对话中保持角色一致性
3. 基于对话历史生成逻辑连贯的回复
4. 自然地处理意图转换

## 对话场景设定:
{scenario}

## 角色设定:
- {role_user}: 寻求信息和帮助的用户
- {role_assistant}: 助手（你的角色），提供专业和有帮助的回答

## 当前意图上下文:
意图: {current_intent}
意图路径: {intent_path}

## 对话历史:
{conversation_history}

## 当前状态:
这是对话的第 {current_turn} 轮（总共 {total_turns} 轮）

## Workflow:
1. 回顾对话历史并理解当前上下文
2. 基于{role_assistant}角色设定生成专业回复
3. 保持回复自然、有帮助且与当前意图相关
4. 如果是后续问题，适当引用之前的上下文

## Constraints:
1. 必须保持{role_assistant}角色的一致性和专业性
2. 回复必须与对话历史保持逻辑连贯性
3. 回复应详细但简洁（通常2-4句话）
4. 不要使用"根据参考资料"或"根据数据"等措辞
5. 保持语气自然和对话化
6. 如果处理意图转换，自然地承认话题变化

## Output Format:
返回有效的JSON格式：
```json
{{
  "content": "作为{role_assistant}的完整回复"
}}
```

注意：
1. 必须返回有效的JSON格式
2. 仅包含content字段
3. 不要包含任何额外的标识符或格式标记
"""

# English prompt for next question generation
NEXT_QUESTION_PROMPT_EN = """
# Role: Multi-turn Conversation User
## Profile:
- Description: You are a conversation participant playing the user role, generating natural follow-up questions.
- Goal: Generate follow-up questions that advance the conversation naturally, potentially transitioning to related intents.

## Skills:
1. Analyze conversation history and identify natural progression
2. Maintain user role consistency
3. Generate natural, fluent follow-up questions
4. Handle intent transitions smoothly when appropriate

## Conversation Scenario:
{scenario}

## Role Settings:
- {role_user}: User (your role) asking follow-up questions
- {role_assistant}: Assistant providing answers

## Current Intent Context:
Primary Intent: {primary_intent}
Related Intents: {related_intents}
Intent Path: {intent_path}

## Conversation History:
{conversation_history}

## Current Status:
About to start turn {next_turn} of conversation (total {total_turns} turns)
Intent transition probability: {transition_rate}%

## Workflow:
1. Review the conversation history and understand what has been discussed
2. Decide whether to continue on current intent or transition to a related intent
3. Generate a natural follow-up question from {role_user}'s perspective
4. Ensure the question advances the conversation meaningfully

## Constraints:
1. Must maintain {role_user} role's language style
2. Question must be based on conversation history
3. Avoid repeating previously asked questions
4. Question types can be: clarifying details, asking for examples, related topics, practical application
5. Keep questions concise and clear
6. If transitioning intents, make it natural (e.g., "Also, about...", "One more thing...")

## Output Format:
Return valid JSON format:
```json
{{
  "question": "Your question as {role_user}",
  "intent": "The intent this question belongs to"
}}
```

Note:
1. Must return valid JSON format
2. Include both question and intent fields
3. Intent should be from the provided intent options
"""

# Chinese prompt for next question generation
NEXT_QUESTION_PROMPT_ZH = """
# Role: 多轮对话用户角色
## Profile:
- Description: 你是一个对话参与者，扮演用户角色，生成自然的后续问题。
- Goal: 生成推进对话自然发展的后续问题，可能转换到相关意图。

## Skills:
1. 分析对话历史并识别自然的发展方向
2. 保持用户角色一致性
3. 生成自然、流畅的后续问题
4. 在适当时平滑处理意图转换

## 对话场景设定:
{scenario}

## 角色设定:
- {role_user}: 用户（你的角色）提出后续问题
- {role_assistant}: 助手提供回答

## 当前意图上下文:
主要意图: {primary_intent}
相关意图: {related_intents}
意图路径: {intent_path}

## 对话历史:
{conversation_history}

## 当前状态:
即将开始第 {next_turn} 轮对话（总共 {total_turns} 轮）
意图转换概率: {transition_rate}%

## Workflow:
1. 回顾对话历史并理解已讨论的内容
2. 决定是继续当前意图还是转换到相关意图
3. 从{role_user}的角度生成自然的后续问题
4. 确保问题有意义地推进对话

## Constraints:
1. 必须保持{role_user}角色的语言风格
2. 问题必须基于对话历史
3. 避免重复之前已问过的问题
4. 问题类型可以是：澄清细节、询问示例、相关话题、实际应用
5. 保持问题简洁明确
6. 如果转换意图，使其自然（例如："另外，关于..."，"还有一个问题..."）

## Output Format:
返回有效的JSON格式：
```json
{{
  "question": "作为{role_user}的问题",
  "intent": "这个问题所属的意图"
}}
```

注意：
1. 必须返回有效的JSON格式
2. 包含question和intent两个字段
3. intent应该从提供的意图选项中选择
"""


def build_assistant_reply_prompt(
    scenario: str,
    role_user: str,
    role_assistant: str,
    current_intent: str,
    intent_path: str,
    conversation_history: str,
    current_turn: int,
    total_turns: int,
    language: str = "en"
) -> str:
    """
    Build prompt for generating assistant replies in multi-turn conversations

    Args:
        scenario: Conversation scenario description
        role_user: User role name
        role_assistant: Assistant role name
        current_intent: Current intent being discussed
        intent_path: Full path of the intent in taxonomy
        conversation_history: Formatted conversation history
        current_turn: Current turn number
        total_turns: Total number of turns planned
        language: Language for the prompt (en/zh)

    Returns:
        Formatted prompt string
    """
    template = ASSISTANT_REPLY_PROMPT_EN if language == "en" else ASSISTANT_REPLY_PROMPT_ZH

    return template.format(
        scenario=scenario,
        role_user=role_user,
        role_assistant=role_assistant,
        current_intent=current_intent,
        intent_path=intent_path,
        conversation_history=conversation_history,
        current_turn=current_turn,
        total_turns=total_turns
    )


def build_next_question_prompt(
    scenario: str,
    role_user: str,
    role_assistant: str,
    primary_intent: str,
    related_intents: str,
    intent_path: str,
    conversation_history: str,
    next_turn: int,
    total_turns: int,
    transition_rate: float,
    language: str = "en"
) -> str:
    """
    Build prompt for generating next user question in multi-turn conversations

    Args:
        scenario: Conversation scenario description
        role_user: User role name
        role_assistant: Assistant role name
        primary_intent: Primary intent of the conversation
        related_intents: Related intents that could be transitioned to
        intent_path: Full path of the primary intent in taxonomy
        conversation_history: Formatted conversation history
        next_turn: Next turn number
        total_turns: Total number of turns planned
        transition_rate: Probability of intent transition (0-100)
        language: Language for the prompt (en/zh)

    Returns:
        Formatted prompt string
    """
    template = NEXT_QUESTION_PROMPT_EN if language == "en" else NEXT_QUESTION_PROMPT_ZH

    return template.format(
        scenario=scenario,
        role_user=role_user,
        role_assistant=role_assistant,
        primary_intent=primary_intent,
        related_intents=related_intents,
        intent_path=intent_path,
        conversation_history=conversation_history,
        next_turn=next_turn,
        total_turns=total_turns,
        transition_rate=int(transition_rate * 100)
    )
