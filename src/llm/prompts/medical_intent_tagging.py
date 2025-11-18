"""
Prompt templates for medical conversation intent tagging
"""


def get_medical_intent_tagging_prompt(language: str = "en", taxonomy_text: str = None) -> dict:
    """Get prompt for generating intent tags for medical conversations"""

    # Add taxonomy context if provided
    taxonomy_context = ""
    if taxonomy_text:
        if language == "zh":
            taxonomy_context = f"\n\n可用的意图分类树：\n{taxonomy_text}\n\n请确保生成的意图标签遵循这个分类树的结构。"
        else:
            taxonomy_context = f"\n\nAvailable Intent Taxonomy:\n{taxonomy_text}\n\nEnsure your generated intent tags follow this taxonomy structure."

    if language == "zh":
        return {
            "system": f"""你是一位医疗对话分析专家。你的任务是分析医患对话，为每一轮患者的发言生成准确的意图标签。

意图标签应该：
1. 具体且有意义（例如："症状报告 - 泌尿问题"而不是"提问"）
2. 采用层次化结构（例如："医疗咨询 -> 主诉 -> 症状描述"）
3. 考虑对话上下文和医疗领域知识
4. 涵盖常见医疗对话场景：症状报告、病史询问、体格检查、诊断讨论、治疗计划、用药指导、随访安排等
{taxonomy_context}

请为每轮患者发言生成结构化的意图标签。""",

            "user": """请分析以下医患对话，为每一轮患者的发言生成意图标签。

对话内容：
{conversation}

请以JSON格式返回结果，包含：
1. user_turns: 每轮患者发言的意图信息列表
   - turn_index: 发言在对话中的索引（从0开始）
   - content: 发言内容（前50个字符）
   - intent: 简短的意图名称（例如："症状报告"）
   - intent_path: 层次化意图路径（例如："医疗咨询 -> 主诉 -> 症状描述"）
   - reasoning: 简短的意图判断理由

2. primary_intent: 整个对话的主要意图
3. all_intents: 对话中出现的所有意图列表（去重）
4. conversation_summary: 对话的简短摘要（1-2句话）

请确保返回有效的JSON格式。"""
        }

    # English version
    return {
        "system": f"""You are a medical dialogue analysis expert. Your task is to analyze doctor-patient conversations and generate accurate intent tags for each patient turn.

Intent tags should be:
1. Specific and meaningful (e.g., "Symptom Reporting - Urination Issues" not just "Question")
2. Hierarchical in structure (e.g., "Medical Consultation -> Chief Complaint -> Symptom Description")
3. Context-aware, considering the conversation flow and medical domain
4. Cover common medical dialogue scenarios: symptom reporting, medical history, physical examination, diagnosis discussion, treatment planning, medication guidance, follow-up scheduling, etc.
{taxonomy_context}

Generate structured intent tags for each patient turn.""",

        "user": """Analyze the following doctor-patient conversation and generate intent tags for each patient turn.

Conversation:
{conversation}

Return the result in JSON format with:
1. user_turns: List of intent information for each patient turn
   - turn_index: Index of the turn in conversation (starting from 0)
   - content: Turn content (first 50 characters)
   - intent: Short intent name (e.g., "Symptom Reporting")
   - intent_path: Hierarchical intent path (e.g., "Medical Consultation -> Chief Complaint -> Symptom Description")
   - reasoning: Brief explanation for the intent classification

2. primary_intent: Primary intent of the entire conversation
3. all_intents: List of all unique intents in the conversation
4. conversation_summary: Brief summary of the conversation (1-2 sentences)

Ensure you return valid JSON format."""
    }
