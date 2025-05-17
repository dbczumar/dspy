import dspy.retrievers
from dspy.dsp.utils.settings import settings
from dspy.predict import *
from dspy.primitives import *
from dspy.retrieve import *
from dspy.signatures import *
from dspy.teleprompt import *
from dspy.utils.asyncify import asyncify
from dspy.utils.logging_utils import configure_dspy_loggers, disable_logging, enable_logging
from dspy.utils.saving import load
from dspy.utils.streaming import streamify
from dspy.utils.usage_tracker import track_usage

from dspy.evaluate import Evaluate  # isort: skip
from dspy.clients import *  # isort: skip
from dspy.adapters import Adapter, ChatAdapter, JSONAdapter, Image, History  # isort: skip


configure_dspy_loggers(__name__)

from dspy.dsp.colbertv2 import ColBERTv2

# from dspy.dsp.you import You

configure = settings.configure
context = settings.context

BootstrapRS = BootstrapFewShotWithRandomSearch

from typing import Any, Dict, List, Optional, Tuple

from .__metadata__ import __author__, __author_email__, __description__, __name__, __url__, __version__


# Add some parameters ... max iterations, etc?
def optimize_prompts(
    function, dataset: List[Tuple[Any, Any]], prompts: List[str] = None, correctness_threshold: float = 0.75
) -> Dict[str, str]:
    """
    Optimizes the registered prompts in the specified function.

    Note: This function currently only supports prompts registered with the MLflow Prompt Registry.

    Args:
        function: The function containing registered prompts to optimize.
        dataset: A list of tuples containing the input to the function and expected output.
        prompts: A list of names of registered prompts to optimize. If unspecified, all registered
                 prompts within the function will be optimized.
        correctness_threshold: The minimum correctness score for the optimized prompts.
    """
    import mlflow

    optimizer_inputs = _get_optimizer_inputs(function, dataset)
    improved_prompts = _generate_improved_prompts(function, dataset, optimizer_inputs, correctness_threshold)
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_registry_uri("mlruns")
    for prompt_name, prompt_template in improved_prompts.items():
        # Register the new prompt with the MLflow Prompt Registry
        mlflow.register_prompt(
            name=prompt_name,
            template=prompt_template,
        )
    return improved_prompts


def _get_optimizer_inputs(
    function, dataset: List[Tuple[Any, Any]], prompt_template_replacements: Optional[Dict[str, str]] = None
):
    import os

    import mlflow
    from mlflow.utils import gorilla

    os.environ.pop("DATABRICKS_HOST", None)
    os.environ.pop("DATABRICKS_TOKEN", None)
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_registry_uri("mlruns")

    prompt_template_replacements = prompt_template_replacements or {}

    extracted_prompt_templates = {}
    extracted_prompt_variables = {}

    def _mlflow_prompt_with_extraction(name: str, default: Optional[str] = None) -> mlflow.entities.Prompt:
        nonlocal extracted_prompt_templates
        original_fn = gorilla.get_original_attribute(mlflow, "prompt", bypass_descriptor_protocol=False)
        prompt = original_fn(name, default)
        if name in prompt_template_replacements:
            extracted_prompt_templates[name] = prompt_template_replacements[name]
        else:
            extracted_prompt_templates[name] = prompt.template

        return prompt

    def _mlflow_prompt_format_with_extraction(self, allow_partial: bool = False, **kwargs):
        nonlocal extracted_prompt_variables
        extracted_prompt_variables[self.name] = kwargs
        if self.name in prompt_template_replacements:
            return prompt_template_replacements[self.name].replace("{{ country }}", kwargs["country"])
        else:
            original_fn = gorilla.get_original_attribute(
                mlflow.entities.Prompt, "format", bypass_descriptor_protocol=False
            )
            return original_fn(self, allow_partial=allow_partial, **kwargs)

    patches = []
    try:
        patches.append(
            _monkey_patch(
                destination=mlflow,
                name="prompt",
                replacement=_mlflow_prompt_with_extraction,
            )
        )
        patches.append(
            _monkey_patch(
                destination=mlflow.entities.Prompt,
                name="format",
                replacement=_mlflow_prompt_format_with_extraction,
            )
        )
        optimizer_inputs = []
        for function_input, expected_output in dataset:
            with mlflow.start_span():
                function_output = function(function_input)
            trace = mlflow.get_last_active_trace()
            optimizer_inputs.append(
                {
                    "function_input": function_input,
                    "function_output": function_output,
                    "expected_output": expected_output,
                    "extracted_prompt_templates": dict(extracted_prompt_templates),
                    "extracted_prompt_variables": dict(extracted_prompt_variables),
                    "trace": trace.data.spans,
                }
            )
            extracted_prompt_templates.clear()
            extracted_prompt_variables.clear()

    finally:
        for patch in patches:
            gorilla.revert(patch)

    return optimizer_inputs


def _monkey_patch(destination, name, replacement):
    from mlflow.utils import gorilla

    patch = gorilla.Patch(
        destination=destination, name=name, obj=replacement, settings=gorilla.Settings(allow_hit=True)
    )
    gorilla.apply(patch)
    return patch


def _generate_improved_prompts(function, dataset, optimizer_inputs, correctness_threshold) -> Dict[str, str]:
    for _ in range(10):
        print(optimizer_inputs)
        # Extract prompts
        prompt_names = {
            key
            for item in optimizer_inputs
            if "extracted_prompt_templates" in item
            for key in item["extracted_prompt_templates"]
        }

        # Evaluate
        score = reward_function(
            [(item["function_input"], item["function_output"], item["expected_output"]) for item in optimizer_inputs]
        )
        print("SCORE:", score)
        if score >= correctness_threshold:
            return {
                key: value
                for item in optimizer_inputs
                if "extracted_prompt_templates" in item
                for key, value in item["extracted_prompt_templates"].items()
            }

        # Propose
        new_prompts = {}
        for prompt_name in prompt_names:
            # Call the function to suggest an improved prompt
            # This is a placeholder for the actual implementation
            # In practice, you would call the function that generates the improved prompt
            # For example:
            # new_prompt = suggest_improved_prompt(prompt_name, optimizer_inputs)
            # Register the new prompt with the MLflow Prompt Registry
            print(f"Suggesting improved prompt for {prompt_name}")
            new_prompt = _suggest_improved_prompt(prompt_name, optimizer_inputs)
            new_prompts[prompt_name] = new_prompt
            print(f"New prompt for {prompt_name}: {new_prompt}")

        optimizer_inputs = _get_optimizer_inputs(function, dataset, prompt_template_replacements=new_prompts)
    return {}


def _suggest_improved_prompt(prompt_name, optimizer_inputs) -> str:
    """
    Suggests a new prompt based on the inputs, outputs, and labels.

    Args:
        prompt_name: The name of the prompt for which to suggest a new version.
        inputs_outputs_labels: A list of tuples containing the input to the function,
                               the output of the function, and the expected output.

    Returns:
        A suggested prompt.
    """
    import os

    from openai import OpenAI
    from pydantic import BaseModel

    os.environ["OPENAI_API_KEY"] = "..."  # Replace with your OpenAI API key

    client = OpenAI()

    class SuggestedImprovedPrompt(BaseModel):
        improved_prompt_template: str

    system_prompt = (
        " An application developer is writing a generative AI application that uses LLM prompting"
        " to generate responses to users inputs."
        " Your job is to improve prompt templates that are used by the large language model to generate"
        " responses to a given input so that the developer's application produces the correct output."
        " Each prompt template is a string with placeholders delimited by double curly braces. YOU"
        " MUST NOT CHANGE THE PLACEHOLDERS; YOU CAN MOVE THE PLACEHOLDERS, BUT YOU MUST NOT CHANGE THEIR CONTENTS.\n\n"
        "You will be given a list of trials in the following format, and your job is to respond"
        " with an improved prompt template. Note that the trials format also includes a trace,"
        " which you can use to understand how the prompt template is used within the application;"
        " PLEASE READ THE TRACE CAREFULLY SO THAT YOU CAN INFER WHERE EACH TEMPLATE IS USED, THOUGH"
        " THERE MAY NOT BE ENOUGH INFORMATION TO INFER THIS (THAT'S OKAY, PROCEED TO GENERATE A NEW PROMPT TEMPLATE ANYWAY!)\n\n"
        " MAKE SURE TO TRY TO GET THE CORRECT RESPONSE. ASSUME THE ORIGINAL TEMPLATE IS BAD AND TRY TO FIX IT!!!!!\n\n"
        "### Trial 1\n\n"
        "Application Inputs: <inputs>\n\n"
        "Application Outputs: <outputs>\n\n"
        "Expected Outputs: <expected_outputs>\n\n"
        "Prompt Template: <template>\n\n"
        "Prompt Variables: <variables>\n\n"
        "Trace: <trace>\n\n"
        "### Trial 2\n\n"
        "..."
    )

    content = "\n\n".join(
        [
            (
                f"### Trial {idx + 1}\n\n"
                f"Application Inputs: {trial['function_input']}\n\n"
                f"Application Outputs: {trial['function_output']}\n\n"
                f"Expected Outputs: {trial['expected_output']}\n\n"
                f"Prompt Template: {trial['extracted_prompt_templates'][prompt_name]}\n\n"
                f"Prompt Variables: {trial['extracted_prompt_variables'][prompt_name]}\n\n"
                f"Trace: {trial['trace']}"
            )
            for idx, trial in enumerate(optimizer_inputs)
        ]
    )

    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        response_format=SuggestedImprovedPrompt,
    )

    event = completion.choices[0].message.parsed
    return event.improved_prompt_template


def reward_function(inputs_outputs_labels: List[Tuple[Any, Any, Any]]) -> float:
    import os

    from databricks.agents.evals import judges

    os.environ["DATABRICKS_HOST"] = "..."
    os.environ["DATABRICKS_TOKEN"] = "..."

    total_correct = 0
    for inp, output, label in inputs_outputs_labels:
        is_correct = judges.correctness(
            request=inp,
            response=output,
            expected_response=label,
        )
        if "yes" in str(is_correct.value).lower():
            total_correct += 1

    return float(total_correct) / len(inputs_outputs_labels)
