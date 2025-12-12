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


class LongMemEvalTemplate:
    @staticmethod
    def get_anscheck_prompt(task: str, question: str, answer: str, response: str, abstention: bool = False):
        """Generate evaluation prompt based on task type."""
        if not abstention:
            if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
                template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
                prompt = template.format(question, answer, response)
            elif task == 'temporal-reasoning':
                template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
                prompt = template.format(question, answer, response)
            elif task == 'knowledge-update':
                template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
                prompt = template.format(question, answer, response)
            elif task == 'single-session-preference':
                template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
                prompt = template.format(question, answer, response)
            else:
                # Default case for unknown task types
                template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
                prompt = template.format(question, answer, response)
        else:
            template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
            prompt = template.format(question, answer, response)
        
        return prompt

    @staticmethod
    def generate_evaluation(
        task: str,
        question: str,
        correct_answer: str,
        model_response: str,
        abstention: bool = False
    ):
        """Generate evaluation prompt for LongMemEval."""
        prompt = LongMemEvalTemplate.get_anscheck_prompt(
            task, question, correct_answer, model_response, abstention
        )
        
        return f"""{prompt}

**
IMPORTANT: Please make sure to only return in JSON format, with the 'is_correct' key as a boolean and 'explanation' key as a string explaining your decision.
Example JSON:
{{
    "is_correct": true,
    "explanation": "The model response correctly answers the question because..."
}}
**

JSON:
"""

    @staticmethod
    def generate_reason(
        task: str,
        question: str,
        correct_answer: str,
        model_response: str,
        is_correct: bool,
        explanation: str
    ):
        """Generate reason for the evaluation result."""
        status = "correct" if is_correct else "incorrect"
        return f"""LongMemEval Assessment:

Task Type: {task}
Question: {question}
Correct Answer: {correct_answer}
Model Response: {model_response}

Evaluation: The model response is {status}.
Explanation: {explanation}

This assessment is based on the specific criteria for {task} tasks in the LongMemEval benchmark."""
