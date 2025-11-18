"""
Prompt templates for building medical intent taxonomy from real conversations
"""


def get_medical_taxonomy_prompt(language: str = "en") -> dict:
    """Get prompt for building intent taxonomy from medical conversations"""

    if language == "zh":
        return {
            "system": """你是一位医疗对话分析专家。你的任务是分析一组真实的医患对话，识别常见的对话意图模式，并构建层次化的意图分类树。

意图分类树应该：
1. 反映真实医患对话中的常见意图
2. 具有清晰的层次结构（通常2-3层）
3. 涵盖所有观察到的对话场景
4. 意图名称简洁且有意义

典型的医疗对话意图包括但不限于：
- 症状报告（具体症状、严重程度、持续时间等）
- 病史询问（既往病史、家族史、用药史等）
- 体格检查（检查结果、生命体征等）
- 诊断讨论（疾病解释、诊断结果等）
- 治疗计划（用药指导、手术安排、生活方式建议等）
- 随访安排（复诊时间、注意事项等）

请根据提供的对话样本构建意图分类树。""",

            "user": """分析以下医患对话样本，构建一个层次化的意图分类树。

对话样本（共{num_conversations}个对话）：
{conversation_samples}

请以JSON格式返回意图分类树，结构如下：
{{
  "root": {{
    "name": "医疗咨询",
    "children": [
      {{
        "name": "主诉",
        "children": [
          {{"name": "症状描述"}},
          {{"name": "症状时长"}},
          {{"name": "症状严重程度"}}
        ]
      }},
      {{
        "name": "病史",
        "children": [...]
      }},
      ...
    ]
  }},
  "statistics": {{
    "total_intents": 数字,
    "max_depth": 数字,
    "coverage_explanation": "解释这个分类树如何覆盖观察到的对话意图"
  }}
}}

请确保：
1. 分类树深度为2-3层
2. 每个父节点有3-8个子节点
3. 意图名称简洁明确
4. 覆盖所有观察到的对话场景"""
        }

    # English version
    return {
        "system": """You are a medical dialogue analysis expert. Your task is to analyze a set of real doctor-patient conversations, identify common intent patterns, and build a hierarchical intent taxonomy tree.

The intent taxonomy should:
1. Reflect common intents found in real medical conversations
2. Have a clear hierarchical structure (typically 2-3 levels deep)
3. Cover all observed conversation scenarios
4. Use concise and meaningful intent names

Typical medical conversation intents include but are not limited to:
- Symptom Reporting (specific symptoms, severity, duration, etc.)
- Medical History (past medical history, family history, medication history, etc.)
- Physical Examination (exam findings, vital signs, etc.)
- Diagnosis Discussion (disease explanation, diagnostic results, etc.)
- Treatment Planning (medication guidance, surgery scheduling, lifestyle advice, etc.)
- Follow-up Scheduling (appointment timing, precautions, etc.)

Build an intent taxonomy tree based on the provided conversation samples.""",

        "user": """Analyze the following medical conversation samples and build a hierarchical intent taxonomy tree.

Conversation samples ({num_conversations} conversations):
{conversation_samples}

Return the intent taxonomy tree in JSON format with this structure:
{{
  "root": {{
    "name": "Medical Consultation",
    "children": [
      {{
        "name": "Chief Complaint",
        "children": [
          {{"name": "Symptom Description"}},
          {{"name": "Symptom Duration"}},
          {{"name": "Symptom Severity"}}
        ]
      }},
      {{
        "name": "Medical History",
        "children": [...]
      }},
      ...
    ]
  }},
  "statistics": {{
    "total_intents": number,
    "max_depth": number,
    "coverage_explanation": "Explain how this taxonomy covers the observed conversation intents"
  }}
}}

Ensure:
1. Tree depth is 2-3 levels
2. Each parent has 3-8 children
3. Intent names are concise and clear
4. All observed conversation scenarios are covered"""
    }
