Feature: Game Loop Ollama Configuration
  As a game developer
  I want to configure and manage the Ollama LLM service for the Game Loop application
  So that natural language processing works consistently with optimal performance

  Background:
    Given Ollama is installed on the system
    And a compatible language model is available

  Scenario: Initialize Ollama service with local model
    When I configure Ollama with a local model "mistral-7b"
    Then the Ollama service should load the model successfully
    And the system should be able to make embedding requests
    And the system should be able to make completion requests

  Scenario: Configure Ollama parameters via command line
    When I start the game with command line parameters:
      | parameter      | value           |
      | --model        | nous-hermes-2   |
      | --temperature  | 0.7             |
      | --context-len  | 8192            |
    Then the Ollama service should use the specified model "nous-hermes-2"
    And the Ollama service should use temperature 0.7
    And the Ollama service should use context length 8192

  Scenario: Configure Ollama parameters via configuration file
    Given a configuration file with Ollama settings:
      """
      llm:
        model: llama2-13b
        temperature: 0.8
        url: http://localhost:11434
        context_length: 4096
      """
    When I start the game with the configuration file
    Then the Ollama service should use the settings from the configuration file
    And the settings should override the default values

  Scenario: Override configuration file with command line parameters
    Given a configuration file with Ollama model "llama2-13b"
    When I start the game with command line parameter "--model=mixtral-8x7b"
    Then the Ollama service should use the model "mixtral-8x7b"
    And other settings should be loaded from the configuration file

  Scenario: Configure embedding model separately
    When I configure different models for:
      | purpose    | model         |
      | embeddings | nomic-embed-text |
      | completion | mistral-7b    |
    Then the system should use "nomic-embed-text" for generating embeddings
    And the system should use "mistral-7b" for generating text completions

  Scenario: Handle model loading errors
    Given a non-existent model "invalid-model"
    When I configure Ollama with the non-existent model
    Then the system should detect the model loading error
    And the system should fall back to a default model
    And the system should log the model loading error

  Scenario: Configure retry behavior
    When I configure Ollama with retry settings:
      | setting                | value |
      | max_retries           | 3     |
      | retry_delay_seconds   | 2     |
      | timeout_seconds       | 30    |
    Then the system should retry failed LLM requests up to 3 times
    And the system should wait 2 seconds between retries
    And the system should timeout requests after 30 seconds

  Scenario: Test connection to Ollama service
    When I test the connection to the Ollama service
    Then the system should verify the service is running
    And the system should report the available models
    And the system should confirm the API is accessible

  Scenario: Configure Ollama with hardware acceleration
    Given the system has GPU acceleration available
    When I enable hardware acceleration for Ollama
    Then the Ollama service should use GPU for inference
    And the model loading time should be optimized
    And the inference speed should be improved

  Scenario: Configure prompt templates for game functions
    When I configure custom prompt templates for:
      | function               | template_name        |
      | intent_recognition     | game_intent.prompt   |
      | location_generation    | location_gen.prompt  |
      | npc_conversation       | npc_dialog.prompt    |
      | object_interaction     | interaction.prompt   |
    Then the system should load the custom prompt templates
    And the LLM requests should use the appropriate templates for each function

  Scenario: Validate prompt templates
    Given a custom prompt template file "invalid_prompt.txt"
    When I validate the prompt template
    Then the system should check for required placeholder variables
    And the system should verify the template format
    And the system should report any missing placeholders

  Scenario: Configure streaming responses
    When I enable streaming responses for the Ollama service
    Then the system should process LLM responses incrementally
    And the user interface should display text as it's generated
    And the perceived response time should be improved

  Scenario: Configure response caching
    When I enable response caching with:
      | setting              | value |
      | cache_size_items     | 1000  |
      | cache_ttl_minutes    | 60    |
    Then the system should cache similar LLM requests
    And repeated requests should be served from cache
    And the cache should expire items after 60 minutes

  Scenario: Monitoring Ollama performance
    When I enable performance monitoring
    Then the system should track:
      | metric                | type    |
      | response_time_ms      | average |
      | tokens_per_second     | average |
      | requests_per_minute   | count   |
      | error_rate            | percent |
    And performance metrics should be available for analysis
    And alerts should trigger when metrics exceed thresholds

  Scenario Outline: Configure inference parameters for different scenarios
    When I configure scenario-specific LLM settings for "<scenario>"
    Then the system should use temperature "<temperature>" for that scenario
    And the system should use top_p "<top_p>" for that scenario

    Examples:
      | scenario           | temperature | top_p |
      | combat             | 0.5         | 0.9   |
      | puzzle_solving     | 0.2         | 0.95  |
      | npc_conversation   | 0.8         | 0.9   |
      | world_discovery    | 0.7         | 0.9   |
      | object_description | 0.4         | 0.8   |
