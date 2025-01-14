# Optimizing Llama 2 for Knowledge-Grounded Dialogue
## About The Project
This research focuses on fine-tuning Llama 2 for knowledge-grounded dialogue generation in resource-constrained environments, utilizing techniques like QLoRA and PEFT to improve contextual relevance and efficiency.

### Key Results
* Model size reduction: **99.5%** (from **12GB** to **60MB**)
* Evaluation across **200** questions covering **10** diverse topics
* Performance metrics:
  * Base Llama-2: **122** correct, **68** wrong, **10** ambiguous
  * Wiki-Llama-2 w/o QLoRA: **152** correct, **45** wrong, **3** ambiguous 
  * Wiki-Llama-2 (final): **149** correct, **48** wrong, **3** ambiguous

### Key Features
* Advanced fine-tuning techniques:
  * QLoRA for query-based learning optimization
  * PEFT for efficient parameter tuning
* Comprehensive evaluation methodology:
  * Manual assessment
  * Confusion matrix analysis
  * Topic-specific performance visualization

## Implementation Details

### Dataset Processing
* Wizard of Wikipedia dataset:
  * **22,311** dialogues
  * **201,999** conversational turns
  * **2,437** diverse topics
* Four-stage preprocessing pipeline:
  * Data extraction and initial processing
  * Llama 2 compatibility conversion
  * Instruction-based tuning integration
  * Training-ready format compilation

### Model Architecture
* Base model: Llama 2 (**7 billion** parameters)
* Fine-tuning approaches:
  * Direct fine-tuning on WoW dataset
  * QLoRA + PEFT optimization
* Evaluation metrics:
  * Response accuracy
  * Contextual relevance
  * Runtime efficiency

## Performance Analysis

### Trade-offs
* Size vs Performance:
  * Significant size reduction (**99.5%**)
  * Increased initialization and response time
* Accuracy improvements:
  * Higher correct generation rate
  * Reduced ambiguous responses
* Resource efficiency:
  * Optimized for limited GPU memory
  * Runtime within practical limits

### Topic-Specific Performance
* Strong performance in:
  * Technology
  * Environment
  * Sports
* Areas for improvement:
  * Art
  * Celebrities
  * Food

## Future Directions
* Optimization strategies for real-time applications
* Enhanced automated evaluation techniques
* Broader dataset integration
* Improved scalability solutions
* Runtime efficiency optimization

## Acknowledgements
I would like to express my sincere gratitude to my teammates for their invaluable contributions to this research:

* Kamran Gasimov (kamran.qasimoff at kaist.ac.kr)
* Junghun Park (jokjeb2 at kaist.ac.kr)
* Nurlykhan Kopenov (knurlykhan at kaist.ac.kr)
* Bryan Rakoto Dit Sedson (b.rakotoditsedson at kaist.ac.kr)

Their dedication, expertise, and collaborative spirit were essential in navigating the challenges of optimizing large language models under resource constraints.

## Open Collaboration
We encourage researchers and practitioners to build upon this work. The field of resource-efficient AI is rapidly evolving, and there are numerous exciting directions to explore:

* Developing more efficient quantization techniques
* Improving real-time response capabilities
* Creating better evaluation metrics for constrained environments
* Exploring alternative model compression strategies
* Enhancing knowledge integration methods

We welcome contributions, feedback, and collaborations from the community to advance the state of knowledge-grounded dialogue systems.

## License
Research conducted at KAIST. See LICENSE.txt for details.

For questions or collaborations, please feel free to contact.