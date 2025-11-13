## Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http:##www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

from typing import List, Dict


class AnswerCorrectnessTemplate:
    @staticmethod
    def generate_statements(question: str, answer: str):
        return f"""Given a question and an answer, analyze the complexity of each sentence in the answer. Break down each sentence into one or more fully understandable statements. Ensure that no pronouns are used in any statement.


Example:
Example Question: "Who was Albert Einstein and what is he best known for?"
Example Answer: "He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics."

Example JSON:
{{
    "statements": [
        "Albert Einstein was a German-born theoretical physicist.",
        "Albert Einstein is recognized as one of the greatest and most influential physicists of all time.",
        "Albert Einstein was best known for developing the theory of relativity.",
        "Albert Einstein also made important contributions to the development of the theory of quantum mechanics."
    ]
}}
===== END OF EXAMPLE ======

**
IMPORTANT: Please make sure to only return in JSON format, with the 'statements' key as a list of strings.
**

Question:
{question}

Answer:
{answer}

JSON:
"""

    @staticmethod
    def generate_verdicts(
        question: str, answer_statements: List[str], ground_truth_statements: List[str]
    ):
        return f"""Given a question, answer statements, and ground truth statements, analyze each answer statement and classify them in one of the following categories:
- TP (true positive): statements that are present in answer that are also directly supported by one or more statements in ground truth
- FP (false positive): statements present in the answer but not directly supported by any statement in ground truth  
- FN (false negative): statements found in the ground truth but not present in answer

Each statement can only belong to one of the categories. Provide a reason for each classification.


Example:
Example Question: "What powers the sun and what is its primary function?"
Example Answer Statements:
[
    "The sun is powered by nuclear fission, similar to nuclear reactors on Earth.",
    "The primary function of the sun is to provide light to the solar system.",
]
Example Ground Truth Statements:
[
    "The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.",
    "This fusion process in the sun's core releases a tremendous amount of energy.",
    "The energy from the sun provides heat and light, which are essential for life on Earth.",
    "The sun's light plays a critical role in Earth's climate system.",
    "Sunlight helps to drive the weather and ocean currents.",
]

Example JSON:
{{
    "verdicts": [
        {{
            "statement": "The sun is powered by nuclear fission, similar to nuclear reactors on Earth.",
            "verdict": "FP",
            "reason": "This statement is incorrect and contradicts the ground truth which states that the sun is powered by nuclear fusion."
        }},
        {{
            "statement": "The primary function of the sun is to provide light to the solar system.",
            "verdict": "TP", 
            "reason": "This statement is somewhat supported by the ground truth mentioning the sun providing light and its roles, though it focuses more broadly on the sun's energy."
        }},
        {{
            "verdict": "FN", 
            "statement": "The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.",
            "reason""This accurate description of the sun’s power source is not included in the answer."
        }},
        {{
            "verdict": "FN", 
            "statement": "This fusion process in the sun's core releases a tremendous amount of energy.",
            "reason": "This process and its significance are not mentioned in the answer."
        }},
        {{
            "verdict": "FN", 
            "statement": "The energy from the sun provides heat and light, which are essential for life on Earth.",
            "reason": "The answer only mentions light, omitting the essential aspects of heat and its necessity for life, which the ground truth covers."
        }},
        {{
            "verdict": "FN", 
            "statement": "The sun's light plays a critical role in Earth's climate system.",
            "reason": "This broader impact of the sun’s light on Earth's climate system is not addressed in the answer."
        }},
        {{
            "verdict": "FN", 
            "statement": "Sunlight helps to drive the weather and ocean currents.",
            "reason": "The effect of sunlight on weather patterns and ocean currents is omitted in the answer."
        }}
    ]
}}
===== END OF EXAMPLE ======

**
IMPORTANT: Please make sure to only return in JSON format, with the 'verdicts' key as a list of JSON objects. Each object contains 'statement', 'verdict' (TP/FP/FN), and 'reason' keys.No words or explanation is needed.
**

Question:
{question}

Answer Statements:
{answer_statements}

Ground Truth Statements:
{ground_truth_statements}

JSON:
"""

    @staticmethod
    def generate_reason(
        question: str, score: float, verdicts: List[Dict[str, str]]
    ):
        return f"""Given the question, answer correctness score, and classification verdicts, provide a CONCISE summary for the score. Explain why it is not higher, but also why it is at its current score.

The verdicts contain classifications of statements as TP (true positive), FP (false positive), or FN (false negative). Answer correctness represents how well the answer aligns with the ground truth in terms of factual accuracy and completeness.

**
IMPORTANT: Please make sure to only return in JSON format, with the 'reason' key providing the reason.
Example JSON:
{{
    "reason": "The score is <answer_correctness_score> because <your_reason>."
}}

In your reason, you MUST USE the verdicts and their reasons to explain the score. Focus on the balance between correct information (TP), incorrect information (FP), and missing information (FN).
If the score is 1, keep it short and say something positive with an upbeat tone.
**

Answer Correctness Score:
{score}

Question:
{question}

Verdicts:
{verdicts}

JSON:
"""