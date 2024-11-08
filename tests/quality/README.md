# DSPy Quality Tests

This directory contains quality tests for DSPy programs. The purpose of these tests is to verify that DSPy programs produce high-quality outputs across multiple large language models (LLMs), regardless of model size or capability. These tests are designed to ensure that DSPy programs maintain robustness and accuracy across diverse LLM configurations.

### Overview

Each test in this directory executes a DSPy program using various LLMs. By running the same tests across different models, these tests help validate that DSPy programs handle a wide range of inputs effectively and produce reliable outputs, even in cases where the model might struggle with the input or task.

### Key Features

- **Diverse LLMs**: Each DSPy program is tested with multiple LLMs, ranging from smaller models to more advanced, high-performance models. This approach allows us to assess the consistency and generality of DSPy program outputs across different model capabilities.
- **Challenging and Adversarial Tests**: Some of the tests are intentionally challenging or adversarial, crafted to push the boundaries of DSPy. These challenging cases allow us to gauge the robustness of DSPy and identify areas for potential improvement.
- **Cross-Model Compatibility**: By testing with different LLMs, we aim to ensure that DSPy programs perform well across model types and configurations, reducing model-specific edge cases and enhancing program versatility.

### Running the Tests

- First, populate the configuration file `quality_tests_conf.yaml` (located in this directory) with the necessary LiteLLM model/provider names and access credentials for 1. each LLM you want to test and 2. the LLM judge that you want to use for assessing the correctness of outputs in certain test cases. These should be placed in the `litellm_params` section. You can also use `litellm_params` to specify values for LLM hyperparameters like `temperature`. Any model that lacks configured `litellm_params` in this file will be ignored during testing.

  The configuration must also specify a DSPy adapter to use when testing, e.g. `"chat"` (for `dspy.ChatAdapter`) or `"json"` (for `dspy.JSONAdapter`)

  An example of `quality_tests_conf.yaml`:

      ```yaml
      adapter: chat
      model_list:
        # The model to use for judging the correctness of program
        # outputs throughout quality test suites. We recommend using
        # a high quality model as the judge, such as OpenAI GPT-4o
        - model_name: "judge"
          litellm_params:
            model: "openai/gpt-4o"
            api_key: "<my_openai_api_key>"
        - model_name: "gpt-4o"
          litellm_params:
            model: "openai/gpt-4o"
            api_key: "<my_openai_api_key>"
        - model_name: "claude-3.5-sonnet"
          litellm_params:
            model: "openai/claude-3.5"
            api_key: "<my_anthropic_api_key>"

- Second, to run the tests, run the following command from this directory:

  ```bash
      pytest .
  ```

  This will execute all tests for the configured models and display detailed results for each model configuration. Tests are set up to mark expected failures for known challenging cases where a specific model might struggle, while actual (unexpected) DSPy quality issues are flagged as failures (see below).

### Known Failing Models

Some tests may be expected to fail with certain models, especially in challenging cases. These known failures are logged but do not affect the overall test result. This setup allows us to keep track of model-specific limitations without obstructing general test outcomes. Models that are known to fail a particular test case are specified using the `@known_failing_models` decorator. For example:

```
@known_failing_models("llama-3.2-3b-instruct")
def test_program_with_complex_deeply_nested_output_structure():
    ...
```