# Mixture-of-Agents Demo Powered by Cerebras

This Streamlit application showcases the Mixture of Agents (MOA) architecture proposed by Together AI, powered by Cerebras LLMs. It allows users to interact with a configurable multi-agent system for enhanced AI-driven conversations.

![MOA Architecture](./static/moa_cerebras.svg)
*Source: Adaptation of [Together AI Blog - Mixture of Agents](https://www.together.ai/blog/together-moa)*

## Features

- Interactive chat interface powered by MOA
- Configurable main model and layer agents
- Real-time streaming of responses
- Visualization of intermediate layer outputs
- Customizable agent parameters through the UI

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/kevint-cerebras/cerebras-moa.git
   cd cerebras-moa
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   Create a `.env` file in the root directory and add your Cerebras API key:
   ```
   CEREBRAS_API_KEY=your_api_key_here
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your browser and navigate to `http://localhost:8501`

3. The application provides two main sections:
   - Left sidebar: Configure the MOA system
   - Main panel: Chat interface

4. You can adjust various parameters in real-time, including:
   - Main model selection
   - Number of cycles (layers)
   - Temperature settings
   - Layer agent configurations

## How It Works

The Mixture of Agents (MOA) architecture involves:

1. **Main Agent**: The primary LLM that generates the final response
2. **Layer Agents**: Multiple LLMs that analyze the query and provide insights
3. **Cycles**: Iterative process where layer agents contribute to enhancing the response

When you submit a query, it goes through these steps:

1. Layer agents process the query in parallel
2. Their outputs are combined and formatted
3. This combined insight is passed to the main agent
4. The main agent generates the final response

## Implementation Details

This implementation uses the Cerebras Cloud SDK directly without any LangChain dependencies, providing:

- Direct API calls to Cerebras' high-performance inference endpoints
- Custom conversation memory management
- Efficient handling of parallel agent execution
- Streamlined prompt formatting and response handling

## Advanced Configuration

The application allows you to customize the system by editing JSON configurations directly in the UI. You can:

- Modify system prompts
- Change models for individual layer agents
- Adjust temperature and other parameters
- Save and load configurations

## Models

This demo uses Cerebras' API to access various LLMs, including:

- Llama-3.3-70B
- Llama3.1-8B
- Llama-4-scout-17b-16e-instruct
- Qwen-3-32B

## Cerebras Technology

Cerebras Systems has developed the world's largest and fastest AI processor, the Wafer-Scale Engine (WSE). This technology powers the inference API used in this application, providing:

- Unprecedented speed for AI inference workloads
- High throughput for commercial applications
- Seamless scaling for complex AI tasks

## Credits

- MOA architecture: [Together AI](https://www.together.ai/blog/together-moa)
- LLM inference: [Cerebras](https://cerebras.ai/)
- Original research paper: [arXiv:2406.04692](https://arxiv.org/abs/2406.04692)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions to this project are welcome! Please follow these steps to contribute:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes and commit them with descriptive commit messages
4. Push your changes to your fork
5. Submit a pull request to the main repository

Please ensure that your code adheres to the project's coding standards and includes appropriate tests and documentation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Cerebras](https://cerebras.ai/) for providing the underlying language models
- [Together AI](https://www.together.ai/) for proposing the Mixture of Agents architecture and providing the conceptual image
- [Streamlit](https://streamlit.io/) for the web application framework

## Citation

This project implements the Mixture-of-Agents architecture proposed in the following paper:

```
@article{wang2024mixture,
  title={Mixture-of-Agents Enhances Large Language Model Capabilities},
  author={Wang, Junlin and Wang, Jue and Athiwaratkun, Ben and Zhang, Ce and Zou, James},
  journal={arXiv preprint arXiv:2406.04692},
  year={2024}
}
```

For more information about the Mixture-of-Agents concept, please refer to the [original research paper](https://arxiv.org/abs/2406.04692) and the [Together AI blog post](https://www.together.ai/blog/together-moa).

## Contact

For questions or support, please open an issue on the GitHub repository.