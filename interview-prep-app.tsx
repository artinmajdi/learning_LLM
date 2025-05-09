import React, { useState, useEffect } from 'react';
import { Search, BookmarkCheck, Award, CheckCircle, XCircle, PlusCircle, MinusCircle, Bookmark, CheckSquare, Square, ChevronLeft, ChevronRight } from 'lucide-react';

// Add CSS for 3D card effect
const styles = `
  .perspective-1000 {
    perspective: 1000px;
  }
  
  .transform-style-3d {
    transform-style: preserve-3d;
  }
  
  .backface-hidden {
    backface-visibility: hidden;
  }
  
  .rotate-y-180 {
    transform: rotateY(180deg);
  }
`;

const App = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [activeTopic, setActiveTopic] = useState('llm');
  const [searchQuery, setSearchQuery] = useState('');
  const [flashcardIndex, setFlashcardIndex] = useState(0);
  const [showFlashcardAnswer, setShowFlashcardAnswer] = useState(false);
  const [completedTopics, setCompletedTopics] = useState({});
  const [showAnswers, setShowAnswers] = useState({});
  const [checkedItems, setCheckedItems] = useState({});

  // Track progress
  const [progress, setProgress] = useState(0);
  
  useEffect(() => {
    // Calculate progress based on completed topics and checked items
    const topicCount = Object.keys(techTopics).length;
    const checkedCount = Object.values(checkedItems).filter(val => val).length;
    const completedCount = Object.values(completedTopics).filter(val => val).length;
    
    const cramListTotal = cramList.length;
    const cramListCompleted = Object.entries(checkedItems)
      .filter(([key, val]) => key.startsWith('cram-') && val)
      .length;
    
    const totalItems = topicCount + cramListTotal;
    const totalCompleted = completedCount + cramListCompleted;
    
    setProgress(Math.round((totalCompleted / totalItems) * 100));
  }, [completedTopics, checkedItems]);

  const techTopics = {
    llm: {
      title: "LLMs & Gen AI",
      description: "Large Language Models and Generative AI concepts",
      subtopics: [
        {
          name: "RAG (Retrieval-Augmented Generation)",
          description: "A technique that enhances LLM responses by retrieving relevant context from external knowledge sources.",
          keyPoints: [
            "Combines information retrieval with text generation",
            "Helps ground LLM outputs in factual information",
            "Reduces hallucinations by providing specific context",
            "Critical for enterprise applications using proprietary data"
          ],
          airlineContext: "At American Airlines, RAG can be used to ground LLM responses in airline-specific documents like maintenance manuals, flight operations procedures, IT documentation, and company policies."
        },
        {
          name: "Fine-tuning",
          description: "The process of further training a pre-trained model on a specific dataset to adapt it to a particular domain or task.",
          keyPoints: [
            "Adapts model to domain-specific language and tasks",
            "Requires less data than training from scratch",
            "Can improve performance on specialized tasks",
            "More computationally intensive than RAG"
          ],
          airlineContext: "Fine-tuning could help American Airlines create models specialized in aviation terminology, maintenance procedures, or customer service scenarios specific to air travel."
        },
        {
          name: "FlashAttention",
          description: "An optimization algorithm that makes attention computations in transformer models more memory-efficient.",
          keyPoints: [
            "Reduces memory usage during training and inference",
            "Improves computational efficiency",
            "Enables working with longer sequences",
            "Critical for deploying models in resource-constrained environments"
          ],
          airlineContext: "FlashAttention could help optimize LLM deployment at airport gates or on mobile devices used by AA staff, reducing latency for real-time decision making."
        },
        {
          name: "KV-Cache",
          description: "A technique that stores previously computed key and value tensors in transformer models to speed up inference.",
          keyPoints: [
            "Reduces redundant computations",
            "Significantly improves inference speed for autoregressive generation",
            "Trades memory for computational efficiency",
            "Essential for real-time applications"
          ],
          airlineContext: "KV-cache optimization would be critical for deploying responsive chatbots on AA's mobile app or kiosks, ensuring quick response times for customer inquiries."
        },
        {
          name: "Vector Databases",
          description: "Specialized databases designed to store and efficiently query vector embeddings for similarity search.",
          keyPoints: [
            "Enable semantic search based on meaning rather than keywords",
            "Core component of RAG systems",
            "Options include Pinecone, FAISS, Weaviate",
            "Support for hybrid search (combining vector and keyword search)"
          ],
          airlineContext: "AA could use vector databases to index maintenance logs, flight operation manuals, and IT documentation to enable quick retrieval of relevant information when needed."
        },
        {
          name: "Hallucination Testing",
          description: "Methods to evaluate and detect when an LLM generates false or unfactual information.",
          keyPoints: [
            "Critical for ensuring reliable AI systems",
            "Involves comparing outputs against ground truth",
            "Can be automated in CI/CD pipelines",
            "Essential for regulatory compliance"
          ],
          airlineContext: "For AA's IT operations, hallucination testing would be crucial to ensure LLM-based systems don't generate misleading information about maintenance procedures or flight operations."
        }
      ]
    },
    mlops: {
      title: "MLOps & Cloud",
      description: "Machine Learning Operations and Cloud infrastructure",
      subtopics: [
        {
          name: "MLflow",
          description: "An open-source platform for managing the ML lifecycle, including experimentation, reproducibility, and deployment.",
          keyPoints: [
            "Tracking experiments and parameters",
            "Model registry for versioning",
            "Model deployment capabilities",
            "Integration with Azure ML"
          ],
          airlineContext: "American Airlines could use MLflow to track experiments when developing predictive maintenance models or passenger demand forecasting systems, ensuring reproducibility and proper versioning."
        },
        {
          name: "Azure AI Foundry",
          description: "Microsoft's platform for building, training, deploying, and managing generative AI applications.",
          keyPoints: [
            "Access to foundation models (Azure OpenAI Service)",
            "Tools for prompt engineering and evaluation",
            "Capabilities for fine-tuning and RAG",
            "Deployment options (serverless APIs, managed compute)"
          ],
          airlineContext: "AA appears to be standardizing on Azure AI Foundry for its GenAI initiatives, making it essential to understand this platform's capabilities and deployment patterns."
        },
        {
          name: "Infrastructure as Code (IaC)",
          description: "Managing and provisioning infrastructure through code rather than manual processes.",
          keyPoints: [
            "Terraform vs. Bicep for Azure resources",
            "Reproducibility and version control",
            "Automated deployment and scaling",
            "Important for GPU resource management"
          ],
          airlineContext: "IaC would help AA manage its cloud resources efficiently, especially for deploying GPU-enabled infrastructure needed for LLM training and inference."
        },
        {
          name: "Blue-Green Deployment",
          description: "A technique for releasing applications by shifting traffic between two identical environments running different versions.",
          keyPoints: [
            "Minimizes downtime during deployments",
            "Enables easy rollback if issues occur",
            "Requires metrics for monitoring deployment health",
            "Important for business-critical systems"
          ],
          airlineContext: "For critical AI systems like gate assignment or crew scheduling, AA would benefit from blue-green deployments to ensure seamless updates without disrupting operations."
        },
        {
          name: "Data Drift Monitoring",
          description: "Tracking changes in data distributions that might affect model performance over time.",
          keyPoints: [
            "Essential for maintaining model accuracy",
            "Involves statistical comparisons between training and production data",
            "Can trigger retraining when significant drift is detected",
            "Critical for systems operating in changing environments"
          ],
          airlineContext: "AA would need to monitor data drift in customer chat logs or maintenance records to ensure AI systems continue to perform as expected despite changing patterns."
        }
      ]
    },
    airline: {
      title: "Airline Operations",
      description: "Applications of AI/ML in airline-specific contexts",
      subtopics: [
        {
          name: "Smart Gating",
          description: "AI-driven gate assignment system that optimizes airport operations and reduces taxi time.",
          keyPoints: [
            "Currently saves 17 hours of taxi time per day",
            "Reduces fuel consumption (1.4M gallons saved)",
            "Opportunity for Gen AI enhancements",
            "Requires effective A/B testing methodology"
          ],
          airlineContext: "This is a flagship AI initiative at American Airlines that has already demonstrated significant operational benefits. Understanding how Gen AI could further enhance this system is valuable."
        },
        {
          name: "Crew Recovery Optimization",
          description: "Systems that re-assign crew members when disruptions occur, balancing operational needs and crew preferences.",
          keyPoints: [
            "Combines traditional optimization (MILP) with ML/LLM components",
            "Must ensure fairness and comply with regulations",
            "Time-sensitive process during irregular operations",
            "High business impact potential"
          ],
          airlineContext: "During weather disruptions or other irregular operations, AA needs efficient systems to reassign crew while maintaining fairness and minimizing costs."
        },
        {
          name: "Maintenance Log Analysis",
          description: "Using NLP and ML to analyze maintenance logs for anomaly detection and predictive maintenance.",
          keyPoints: [
            "Retrieval schema for efficient information access",
            "Anomaly detection in maintenance patterns",
            "Explainable outputs for Technical Operations staff",
            "Critical for safety and reliability"
          ],
          airlineContext: "AA's maintenance operations could benefit from AI systems that analyze logs to identify potential issues before they cause operational disruptions."
        }
      ]
    },
    governance: {
      title: "Responsible AI",
      description: "Governance, ethics, and responsible use of AI systems",
      subtopics: [
        {
          name: "AI Governance Framework",
          description: "Organizational policies and procedures for responsible AI development and deployment.",
          keyPoints: [
            "American Airlines has a governance-first culture",
            "Includes safety gates at different development stages",
            "Prioritizes reliability in operational contexts",
            "Ensures compliance with regulations"
          ],
          airlineContext: "AA's governance framework would be particularly focused on ensuring AI systems meet the high safety standards required in aviation."
        },
        {
          name: "Bias Detection",
          description: "Methods to identify and mitigate biases in AI systems that could lead to unfair outcomes.",
          keyPoints: [
            "Statistical techniques to measure disparate impact",
            "Testing across different demographic groups",
            "Continuous monitoring for emergent biases",
            "Essential for systems affecting employees or customers"
          ],
          airlineContext: "For crew scheduling or customer service applications, AA would need robust bias detection to ensure fair treatment of both employees and passengers."
        },
        {
          name: "Explainability",
          description: "Making AI decisions understandable to humans, particularly important in high-stakes contexts.",
          keyPoints: [
            "Local vs. global explanations",
            "SHAP values and other interpretation techniques",
            "Trade-off between complexity and explainability",
            "Required for user trust and regulatory compliance"
          ],
          airlineContext: "When AI recommends maintenance actions or operational changes, AA staff would need clear explanations of why these recommendations are being made."
        }
      ]
    },
    fundamentals: {
      title: "DS Fundamentals",
      description: "Core data science concepts that remain relevant",
      subtopics: [
        {
          name: "Regularization (Ridge vs. Lasso)",
          description: "Techniques to prevent overfitting in linear models by penalizing large coefficients.",
          keyPoints: [
            "Ridge uses L2 regularization (sum of squared coefficients)",
            "Lasso uses L1 regularization (sum of absolute coefficients)",
            "Lasso can perform feature selection by zeroing out coefficients",
            "Choice depends on dataset characteristics and goals"
          ],
          airlineContext: "When building delay prediction models, regularization helps create more generalizable models that work across different airports and conditions."
        },
        {
          name: "Feature Importance",
          description: "Methods to identify which features contribute most to a model's predictions.",
          keyPoints: [
            "SHAP (SHapley Additive exPlanations) values",
            "Built-in methods in tree-based models",
            "Permutation importance",
            "Global vs. local importance"
          ],
          airlineContext: "Understanding which factors most influence predictions of passenger demand or airport congestion helps AA prioritize operational improvements."
        },
        {
          name: "Time Series Cross-Validation",
          description: "Specialized CV techniques that respect the temporal nature of time series data.",
          keyPoints: [
            "Prevents data leakage from future to past",
            "Often uses rolling windows or expanding windows",
            "K-fold adaptation for temporal data",
            "Critical for accurate model evaluation"
          ],
          airlineContext: "For forecasting departure delays or passenger volume, proper time series CV ensures models will perform well on future, unseen data."
        },
        {
          name: "Forecasting Models (LSTM vs. Prophet)",
          description: "Different approaches to predicting future values in time series data.",
          keyPoints: [
            "LSTM: Deep learning approach that captures complex patterns",
            "Prophet: Decomposition-based approach handling seasonality well",
            "Trade-offs in complexity, interpretability, and data requirements",
            "Ensemble approaches often perform best"
          ],
          airlineContext: "AA might use these models for demand forecasting, helping to optimize staffing levels, aircraft allocation, and pricing strategies."
        }
      ]
    }
  };

  const practiceQuestions = {
    llm: [
      {
        question: "Explain the differences between RAG and fine-tuning in the context of airline maintenance manuals, and when you would choose each approach.",
        answer: "RAG is ideal for maintenance manuals when we need to ground LLM responses in specific, frequently updated technical documentation without retraining. I'd use RAG when:\n\n• The maintenance documentation changes frequently (new aircraft, procedures, or compliance requirements)\n• We need high accuracy on specific technical details and part numbers\n• We want to minimize hallucinations about critical maintenance procedures\n• We need to deploy quickly with limited GPU resources\n\nI'd choose fine-tuning when:\n\n• We need the model to deeply understand maintenance terminology and patterns across documents\n• Response format consistency is critical (e.g., always following a specific troubleshooting format)\n• The knowledge corpus is relatively stable\n• We need lower latency responses without retrieval overhead\n\nAt American Airlines, I'd likely implement a hybrid approach: fine-tune a model on general aviation maintenance concepts, then use RAG to ground responses in AA's specific maintenance manuals, especially for the newer aircraft in our fleet."
      },
      {
        question: "Walk us through the process of logging a GPT-4o fine-tune in MLflow, promoting it to Production, and then deploying it to AKS.",
        answer: "At American Airlines, I would approach this process as follows:\n\n1. **Fine-tuning setup**:\n   • Prepare AA-specific datasets (e.g., customer service interactions or maintenance logs)\n   • Initialize Azure OpenAI fine-tuning with GPT-4o\n   • Define evaluation metrics relevant to our use case (accuracy on aviation terms, response quality)\n\n2. **MLflow tracking**:\n   • Use `mlflow.start_run()` to begin experiment tracking\n   • Log hyperparameters (learning rate, epochs, prompt templates)\n   • Log the training and evaluation datasets with `mlflow.log_artifact()`\n   • Record metrics during training loops with `mlflow.log_metric()`\n   • Register the fine-tuned model in MLflow Model Registry with aviation-specific metadata\n\n3. **Model promotion**:\n   • Evaluate the model against AA's internal benchmarks (e.g., accuracy on airline terminology)\n   • Request model review from the AI governance team\n   • Use MLflow API to transition the model to 'Staging' then 'Production' stage after approval\n   • Add model description and version notes for compliance tracking\n\n4. **AKS deployment**:\n   • Create deployment configuration with appropriate compute resources (GPU for inference)\n   • Set scaling parameters based on expected load patterns (higher during peak travel times)\n   • Deploy using `mlflow.azureml.deploy()` specifying AKS as target\n   • Set up network security policies compliant with AA's requirements\n   • Implement blue-green deployment strategy to ensure zero downtime\n\n5. **Post-deployment monitoring**:\n   • Set up logging of inference requests and performance metrics\n   • Configure alerts for latency spikes or error rates\n   • Implement feedback loop for continuous improvement\n\nThis end-to-end process ensures our fine-tuned model is properly tracked, evaluated, promoted, and deployed in a production environment with the necessary governance and monitoring."
      },
      {
        question: "What is FlashAttention and why does it matter for inference at the gate?",
        answer: "FlashAttention is an optimization algorithm that makes attention computations in transformer models more memory-efficient and faster by:\n\n• Reducing memory access costs through IO-aware implementation\n• Using block-wise computation to fit operations in fast GPU SRAM\n• Avoiding materialization of large attention matrices\n• Lowering both time and space complexity compared to standard attention\n\nFor American Airlines' gate operations, this matters significantly because:\n\n• **Real-time decision making**: Gate agents need immediate responses when dealing with rebookings, upgrades, or special situations, where even seconds of latency impact customer satisfaction\n\n• **Resource constraints**: Gate workstations and mobile devices have limited GPU memory compared to data centers, and FlashAttention enables running larger models on these constrained devices\n\n• **Handling longer contexts**: With FlashAttention, our LLMs can process longer passenger manifests, connection information, and operational updates in a single prompt\n\n• **Battery efficiency**: For mobile devices used by our staff, the reduced computation translates to better battery life during long shifts\n\n• **Scaling to peak periods**: During irregular operations (IROPS) like weather delays, the optimization allows our systems to handle the surge in queries without degradation\n\nBy implementing FlashAttention in our gate operation models, we can deliver Smart Gating 2.0 with real-time passenger-specific insights while maintaining responsive performance under the demanding conditions of airport operations."
      },
      {
        question: "Define KV-cache and how it slashes token-latency on mobile devices used by flight attendants.",
        answer: "**KV-cache** is a technique that stores previously computed key and value tensors during transformer inference to avoid redundant computations when generating text autoregressively.\n\nFor American Airlines' flight attendants using mobile devices, KV-cache significantly reduces token latency by:\n\n• **Eliminating recalculation**: When generating each new token in a response about passenger services or onboard procedures, the model doesn't need to recompute attention for tokens it has already processed\n\n• **Memory trade-off**: Uses slightly more memory to store cached values, but dramatically reduces computation needs – critical for the limited processing power of flight attendants' mobile devices\n\n• **Progressive response generation**: Enables smooth, word-by-word response generation when attendants query about special meal requirements or handling service recovery situations\n\n• **Battery conservation**: Reduced computation translates directly to longer battery life during long-haul flights where charging may be limited\n\n• **Offline functionality**: Enables responsive LLM functionality even when aircraft connectivity is limited at cruising altitude\n\nIn practice, this means when our flight attendants use the AA Assistant app to look up information about passenger accommodations or regulatory requirements, they receive instant, token-by-token responses instead of waiting several seconds for a complete answer – improving both employee efficiency and passenger service quality during time-sensitive inflight situations."
      },
      {
        question: "How would you automate hallucination tests in a CI pipeline for an LLM used in maintenance documentation?",
        answer: "For American Airlines' maintenance documentation LLM, I would implement automated hallucination testing in CI as follows:\n\n• **Reference dataset creation**:\n  - Curate a test set of maintenance queries with verified ground-truth answers from AA's maintenance manuals\n  - Include specific aircraft part numbers, maintenance procedures, and safety-critical information\n  - Categorize questions by aircraft type, maintenance domain, and criticality level\n\n• **Automated testing framework**:\n  - Implement pre-commit and nightly test suites in our CI pipeline\n  - For each model update or documentation change, run the test suite against our RAG system\n  - Compare generated answers against ground truth using multiple evaluation methods\n\n• **Multi-faceted evaluation metrics**:\n  - Factual consistency scoring using NLI (Natural Language Inference) models\n  - Named entity verification to ensure correct part numbers and procedure codes\n  - Citation accuracy verification to confirm information is properly sourced\n  - Contradictions detection between response and referenced documentation\n\n• **Threshold-based gates**:\n  - Establish minimum performance thresholds for different risk levels\n  - Implement stricter thresholds for safety-critical maintenance procedures\n  - Block deployment if hallucination rates exceed acceptable thresholds\n\n• **Continuous improvement loop**:\n  - Log failed examples to a feedback database\n  - Use failures to improve retrieval components and prompt engineering\n  - Expand test set based on identified weak points\n\n• **Human-in-the-loop validation**:\n  - Route edge cases and borderline examples to maintenance experts for review\n  - Incorporate their feedback into future test iterations\n\nThis approach ensures that our LLM system provides accurate maintenance information, which is essential for aircraft safety and regulatory compliance at American Airlines."
      },
      {
        question: "Pick a vector database for a low-latency rebooking bot and defend your choice.",
        answer: "For American Airlines' low-latency rebooking bot, I would select **Pinecone** as the vector database due to these advantages:\n\n• **Fully managed service**: Minimizes operational overhead for AA's IT team, allowing them to focus on improving the rebooking algorithm rather than database maintenance\n\n• **Low and consistent latency**: Provides sub-100ms query times even at scale, essential when rebooking hundreds of passengers during irregular operations (IROPS) events like weather delays\n\n• **Auto-scaling**: Can rapidly scale during disruption events when query volume spikes, such as when storms affect hub airports like DFW or CLT\n\n• **High availability**: Offers 99.9%+ uptime SLA, critical for a rebooking system that must function reliably during operational disruptions\n\n• **Metadata filtering**: Allows filtering by important rebooking constraints like fare class, AAdvantage status, and connection requirements before performing similarity search\n\n• **Azure integration**: Aligns with AA's Azure-centric infrastructure strategy, enabling seamless connectivity with Azure AI Foundry services\n\n• **Production-proven**: Already used in high-stakes applications in other industries, providing confidence in its reliability\n\nAlternatives considered:\n\n• **FAISS**: While computationally efficient, would require self-hosting and managing scaling, creating unnecessary operational burden during critical rebooking events\n\n• **Weaviate**: Has excellent hybrid search but introduces complexity not needed for our specific rebooking use case focused on finding similar itineraries\n\n• **Chroma**: Simpler to implement but lacks the enterprise-grade features needed for mission-critical airline operations\n\nPinecone's combination of managed infrastructure, low latency, and production readiness makes it the optimal choice for ensuring passengers can be rebooked quickly during disruptions, improving both operational recovery and customer satisfaction."
      }
    ],
    mlops: [
      {
        question: "Compare Terraform vs. Bicep to spin up a GPU-enabled AKS node pool for LLM inference. Which is more portable and why?",
        answer: "**Terraform vs. Bicep for GPU-enabled AKS at American Airlines:**\n\n**Terraform approach:**\n```hcl\nresource \"azurerm_kubernetes_cluster_node_pool\" \"gpu\" {\n  name                  = \"gpunodepool\"\n  kubernetes_cluster_id = azurerm_kubernetes_cluster.aa_cluster.id\n  vm_size               = \"Standard_NC6s_v3\"  # NVIDIA GPU-enabled VM\n  node_count            = 3\n  mode                  = \"User\"\n  \n  node_labels = {\n    \"accelerator\" = \"nvidia\"\n    \"workload\" = \"llm-inference\"\n  }\n  \n  node_taints = [\"nvidia.com/gpu=present:NoSchedule\"]\n  \n  tags = {\n    Environment = \"Production\"\n    Application = \"LLMInference\"\n  }\n}\n```\n\n**Bicep approach:**\n```bicep\nresource gpuNodePool 'Microsoft.ContainerService/managedClusters/agentPools@2021-05-01' = {\n  parent: aaCluster\n  name: 'gpunodepool'\n  properties: {\n    count: 3\n    vmSize: 'Standard_NC6s_v3'\n    mode: 'User'\n    nodeLabels: {\n      accelerator: 'nvidia'\n      workload: 'llm-inference'\n    }\n    nodeTaints: [\n      'nvidia.com/gpu=present:NoSchedule'\n    ]\n    tags: {\n      Environment: 'Production'\n      Application: 'LLMInference'\n    }\n  }\n}\n```\n\n**Terraform is more portable** for American Airlines' multi-cloud strategy because:\n\n• **Provider ecosystem**: Terraform supports multiple cloud providers (Azure, AWS, GCP) with a consistent syntax, enabling AA to potentially deploy similar LLM infrastructure across different clouds if needed\n\n• **State management flexibility**: Terraform's state can be stored in various backends (Azure Blob Storage, S3, etc.), facilitating easier migration scenarios\n\n• **Abstraction layers**: Terraform modules can abstract provider-specific details, making it easier to adapt deployment scripts for different environments\n\n• **Multi-resource orchestration**: Can coordinate resources across different services and providers in a single deployment, useful for complex AI workloads\n\n• **Community adoption**: Wider adoption means more examples and patterns for GPU workloads across different platforms\n\nWhile Bicep offers tighter Azure integration and better Azure-specific validation, Terraform's cross-platform capabilities make it more suitable for American Airlines' needs, especially if considering potential AWS deployments in the future or wanting to standardize IaC across different departments with varying cloud strategies."
      },
      {
        question: "Describe a Blue-Green rollout for an LLM endpoint in Azure ML; which metrics would trigger a rollback?",
        answer: "**Blue-Green Deployment for LLM Endpoint at American Airlines**\n\n**Implementation Steps:**\n\n1. **Environment Preparation:**\n   • Create identical \"blue\" (current) and \"green\" (new) AzureML inference endpoints\n   • Configure both with identical compute resources optimized for our LLM workloads\n   • Set up shared Azure Front Door as traffic manager with weighted routing\n\n2. **Deployment Process:**\n   • Deploy the new LLM version (e.g., updated RAG system for gate assignments) to the green environment\n   • Run comprehensive validation tests with synthetic American Airlines operational data\n   • Start with 10% traffic to green, 90% to blue using Azure ML's traffic control\n   • Gradually increase traffic to green while monitoring metrics\n   • Once validated, switch 100% traffic to green\n   • Keep blue environment available for quick rollback\n\n3. **Post-Deployment:**\n   • Monitor both environments for 24-48 hours covering peak operational periods\n   • After successful confirmation, decommission blue environment or prepare it for the next release\n\n**Rollback Metrics & Thresholds:**\n\n• **Latency**: Trigger rollback if p95 response time exceeds 300ms (critical for real-time gate operations)\n\n• **Error Rate**: Rollback if errors exceed 0.5% of requests (vs. established baseline of 0.1%)\n\n• **Hallucination Score**: Using our AA-specific factual consistency metric, rollback if score drops below 0.92\n\n• **Business Metrics**: Rollback if gate assignment optimality score decreases by more than 3% compared to baseline\n\n• **User Feedback**: Automated rollback if negative feedback from gate agents exceeds 5% of interactions\n\n• **System Resource Utilization**: Trigger alert if GPU utilization exceeds 85% sustained for >15 minutes\n\n• **Cost Efficiency**: Rollback if token utilization increases by >15% for equivalent operations\n\nThis approach ensures we can deploy LLM updates that improve American Airlines operations without risking disruption to critical systems like gate assignments, rebooking tools, or crew recovery applications."
      },
      {
        question: "How do you monitor data drift in live chat logs for an LLM-based customer service system?",
        answer: "**Monitoring Data Drift in AA's Live Chat Logs**\n\n**Data Collection Pipeline:**\n\n• Store historical chat logs with metadata (customer issue type, resolution path, timestamps)\n• Establish a baseline distribution from training data (passenger queries by category, sentiment distribution, topic frequency)\n• Sample production chat logs in real-time through Azure Event Hub\n• Process logs through a feature extraction pipeline to normalize formats\n\n**Key Drift Metrics to Monitor:**\n\n1. **Statistical Distribution Shifts:**\n   • Track KL divergence between baseline and production distributions of:  \n     - Query types (rebooking, baggage, loyalty program questions)\n     - Sentiment scores\n     - Message length distribution\n     - Booking class distribution\n   • Alert if divergence exceeds predetermined thresholds\n\n2. **Semantic Drift:**\n   • Monitor embedding space with UMAP visualization\n   • Track cosine similarity between centroids of historical vs. current chat clusters\n   • Detect new clusters emerging (potentially new customer concerns)\n\n3. **Performance Indicators:**\n   • Track resolution rate and time by issue category\n   • Monitor customer satisfaction scores over time\n   • Track escalation rates to human agents\n\n**Implementation at American Airlines:**\n\n• Deploy an Azure ML drift detection pipeline running hourly checks\n• Create a real-time dashboard for operations team showing:\n  - Current drift metrics vs. thresholds\n  - Trend lines for key distribution parameters\n  - Automated anomaly detection for sudden changes\n\n**Response Mechanisms:**\n\n• Set up tiered alerts based on drift magnitude:\n  - Low: Log and review weekly\n  - Medium: Notify team for investigation\n  - High: Trigger investigation and potential model update\n\n• Implement automated topic discovery for emerging issues:\n  - Automatically identify new topics in drifting data\n  - Surface examples for data science team review\n  - Create fast-track for incorporating new patterns\n\n• Connect drift detection with MLflow to link performance changes to specific data patterns\n\nThis comprehensive monitoring system ensures our LLM customer service system stays aligned with evolving passenger needs, especially during seasonal travel changes, irregular operations, or sudden policy updates."
      }
    ],
    airline: [
      {
        question: "Smart Gating already saves 17 hrs taxi time/day—what Gen AI layer would you add and how would you A/B test it?",
        answer: "**Enhanced Smart Gating with Gen AI Layer**\n\n**Proposed Gen AI Enhancement:**\n\nI would implement a \"Dynamic Context-Aware Gate Assignment LLM\" that:\n\n• **Incorporates real-time conversational context** from ground operations, tower communications, and crew messaging to anticipate gate needs beyond structured data\n\n• **Processes multimodal inputs** including ramp camera feeds, weather radar, and baggage system status to better predict potential delays and congestion\n\n• **Generates explanatory narratives** alongside gate assignments, providing ground staff with reasoning and suggested contingency options\n\n• **Enables natural language interaction** for operations staff to query the system about assignment decisions or request specific scenarios\n\n**Implementation Architecture:**\n\n• Fine-tuned LLM with aviation operations context connected to the existing Smart Gating optimization engine\n• Real-time embedding and indexing of operational communications\n• Vector database storing historical resolution patterns for similar situations\n• RAG system grounding recommendations in AA operational procedures\n\n**A/B Testing Methodology:**\n\n1. **Test Design:**\n   • Split test across 2-4 major AA hubs (e.g., DFW and CLT as test airports, ORD and MIA as controls)\n   • Run 4-week test with crossover design to control for airport-specific factors\n   • Balanced allocation of similar operational days (avoiding major weather events for initial testing)\n\n2. **Primary Metrics:**\n   • Taxi time reduction beyond current 17 hrs/day baseline\n   • Fuel savings compared to current 1.4M gal baseline\n   • Gate conflict resolution time (how quickly issues are addressed)\n   • Unnecessary gate changes (measuring decision stability)\n   • Recovery time after irregular operations\n\n3. **Secondary Metrics:**\n   • Operations staff satisfaction scores\n   • Time spent interacting with gate assignment system\n   • Quality of explanations (rated by operations staff)\n   • Adaptation speed to unexpected events\n\n4. **Statistical Analysis:**\n   • CUPED (Controlled-experiment Using Pre-Experiment Data) to reduce variance\n   • Difference-in-differences analysis comparing test vs. control airports\n   • Segmentation analysis by aircraft type, time of day, and weather conditions\n\n5. **Rollout Strategy:**\n   • Progressive deployment starting with lowest-risk airports\n   • Shadowing period where Gen AI suggestions are logged but not automatically implemented\n   • Full deployment with continuous monitoring and feedback loop\n\nThis enhancement could potentially increase Smart Gating's taxi time savings from 17 to 20+ hours per day while providing valuable operational intelligence during irregular operations when optimization is most critical."
      },
      {
        question: "Propose a crew recovery optimizer that pairs MILP with an LLM; how do you prove fairness?",
        answer: "**MILP-LLM Hybrid Crew Recovery System for American Airlines**\n\n**System Architecture:**\n\n1. **MILP Optimization Core:**\n   • Formulates crew recovery as a constrained optimization problem\n   • Objectives: minimize disruption cost, recovery time, and crew dissatisfaction\n   • Hard constraints: FAA duty time limits, required rest periods, qualifications\n   • Soft constraints: crew preferences, seniority considerations, commuting patterns\n\n2. **LLM Enhancement Layer:**\n   • **Context Enrichment**: Interprets crew communications, historical preferences, and exceptions\n   • **Constraint Generation**: Converts natural language crew policies into formal constraints\n   • **Solution Explanation**: Generates personalized explanations for affected crew members\n   • **Feedback Processing**: Analyzes crew feedback to improve future recovery scenarios\n\n3. **Integration Approach:**\n   • LLM preprocesses inputs to identify implicit constraints and preferences\n   • MILP solver generates mathematically optimal recovery solutions\n   • LLM post-processes solutions to enhance explainability and fairness evaluation\n\n**Fairness Validation Framework:**\n\n1. **Multi-dimensional Fairness Metrics:**\n   • **Distributional Fairness**: Equal distribution of disruption across demographic groups\n   • **Workload Equity**: Balanced distribution of recovery assignments over time\n   • **Preference Satisfaction**: Similar rates of honoring preferences across groups\n   • **Seniority Respect**: Appropriate weighting of seniority in decisions\n   • **Commuter Consideration**: Balanced accommodation of commuting crews\n\n2. **Quantitative Validation:**\n   • Calculate Gini coefficient for disruption distribution across crew segments\n   • Implement fairness constraints directly in the MILP formulation\n   • Use counterfactual testing to detect disparate impact\n   • Maintain auditable decision logs with fairness metrics for each recovery event\n\n3. **Qualitative Validation:**\n   • LLM-generated explanations highlighting fairness considerations\n   • Structured crew feedback collection specific to fairness perceptions\n   • Regular review with crew representatives and union stakeholders\n\n4. **Continuous Improvement:**\n   • Feed fairness metrics back into model tuning\n   • Update constraint weights based on validated fairness outcomes\n   • A/B test different fairness approaches during simulated disruptions\n\n**Implementation Strategy:**\n\n• Start with shadow testing during actual irregular operations\n• Conduct retrospective analysis comparing system recommendations to actual decisions\n• Phase in gradually with increasing levels of automation\n• Maintain human oversight for edge cases and major disruptions\n\nThis hybrid approach leverages MILP's optimality guarantees while using LLM capabilities to enhance context understanding, policy interpretation, and explanation generation—leading to recovery solutions that are not only operationally sound but demonstrably fair to American Airlines' diverse crew population."
      },
      {
        question: "Outline a retrieval schema for maintenance log triage; how do you surface anomaly explanations to Tech Ops?",
        answer: "**Maintenance Log Retrieval & Anomaly Detection System for AA Tech Ops**\n\n**Retrieval Schema Architecture:**\n\n1. **Data Ingestion & Processing Pipeline:**\n   • Ingest structured maintenance logs from AA's maintenance systems across fleet types\n   • Process unstructured mechanic narratives using aviation-specific NLP models\n   • Extract entities (part numbers, procedures, aircraft identifiers, stations)\n   • Normalize technical terminology across Boeing and Airbus documentation\n\n2. **Multi-Vector Embedding Strategy:**\n   • Create specialized embeddings for different aspects:\n     - Component-focused vectors capturing part relationships\n     - Procedure-focused vectors for maintenance actions\n     - Temporal vectors capturing sequence patterns\n     - Anomaly-specific vectors highlighting deviations\n   • Implement composite indexing to balance retrieval precision and recall\n\n3. **Hierarchical Indexing Structure:**\n   • Primary index by aircraft type (777, 787, A321, etc.)\n   • Secondary indices by ATA chapter (standard aircraft system classification)\n   • Tertiary indices by component and maintenance action type\n   • Specialized index for recurring/anomalous patterns\n\n4. **Query Processing:**\n   • Accept natural language queries from Tech Ops personnel\n   • Support structured parameter searches (tail number, date ranges, stations)\n   • Enable similarity search for \"find similar maintenance events\"\n   • Implement boolean filters for regulatory and safety-critical documentation\n\n**Anomaly Explanation System:**\n\n1. **Multi-Modal Anomaly Detection:**\n   • Statistical pattern detection across sensor data\n   • NLP-based identification of unusual maintenance narratives\n   • Sequence anomalies in maintenance actions\n   • Correlation anomalies between systems/components\n\n2. **Explanation Generation Interface:**\n   • **Context Panel**: Shows the detected anomaly in relation to historical patterns\n   • **Similar Case Retrieval**: Displays relevant past occurrences with resolutions\n   • **Contributing Factors Analysis**: LLM-generated explanation of potential causes\n   • **Risk Assessment**: Operational impact prediction with confidence levels\n\n3. **Tech Ops-Optimized Presentation:**\n   • Role-based views (Line Maintenance vs. MRO Engineers vs. QA)\n   • Mobile-friendly interface for line mechanics\n   • Integration with technical documentation and IPD/IPC\n   • Direct links to relevant maintenance procedures and airworthiness directives\n\n4. **Continuous Learning:**\n   • Feedback loop from resolution outcomes\n   • Expert validation of explanations\n   • Tracking of explanation utility and accuracy\n\n**Implementation Benefits for American Airlines:**\n\n• **Reduced AOG time**: Faster troubleshooting through relevant case retrieval\n• **Enhanced safety**: Better identification of fleet-wide issues before they escalate\n• **Knowledge retention**: Preservation of expert knowledge as experienced mechanics retire\n• **Regulatory compliance**: Improved tracking of recurring discrepancies for regulatory reporting\n• **Training enhancement**: Real-world cases for mechanic training programs\n\nThis system would help American Airlines move from reactive to predictive maintenance practices, reducing operational disruptions while improving safety and compliance."
      }
    ],
    governance: [
      {
        question: "Summarize AA's Gen AI governance framework and give a P0 safety-gate example.",
        answer: "**American Airlines' Gen AI Governance Framework**\n\n Based on AA's governance-first culture and aviation industry best practices, American Airlines likely implements a comprehensive Gen AI governance framework with these key components:\n\n• **Tiered Risk Classification System**:\n  - P0: Safety-critical applications (maintenance, flight operations)\n  - P1: Operational applications (crew scheduling, gate management)\n  - P2: Customer-facing applications (rebooking bots, itinerary management)\n  - P3: Internal tools (document summarization, knowledge management)\n\n• **Stage-Gate Approval Process**:\n  - Initial concept risk assessment\n  - Design review with ethics and compliance\n  - Pre-deployment technical validation\n  - Controlled rollout with monitoring\n  - Regular reassessment and audit\n\n• **Cross-Functional Governance Committee**:\n  - Technical AI experts\n  - Legal and compliance representatives\n  - Business unit stakeholders\n  - Safety and operations officers\n  - Third-party auditors when needed\n\n• **Documentation Requirements**:\n  - Model cards with limitations and assumptions\n  - Data provenance tracking\n  - Bias assessment reports\n  - Performance benchmarks against safety criteria\n  - Human oversight protocols\n\n**P0 Safety-Gate Example: Maintenance Advisory LLM**\n\nFor a P0 system like an LLM that assists maintenance technicians with procedural guidance, a critical safety gate would be the **Factual Accuracy Verification Gate**:\n\n• **Gate Description**: Automated and human verification that all LLM outputs match approved maintenance documentation\n\n• **Implementation**:\n  - Automated testing against 10,000+ verified maintenance procedures\n  - Zero tolerance for procedural hallucinations\n  - Specialized test suite focused on safety-critical components (engines, flight controls, etc.)\n  - Dual human SME review of model outputs prior to deployment\n  - Continuous monitoring comparing outputs to source documentation\n\n• **Threshold Criteria**:\n  - 100% accuracy on safety-critical procedures\n  - Automatic fallback to cited documentation when confidence below 99.9%\n  - Mandatory explicit uncertainty acknowledgment\n  - No deployment without successful completion of all test cases\n\n• **Governance Actions**:\n  - Signed attestation by Chief Maintenance Officer\n  - Full model retraining if any safety discrepancy identified\n  - Quarterly recertification process\n  - Immutable audit trail of all test results and approvals\n\nThis example illustrates AA's commitment to safety-first AI development, ensuring that even advanced Gen AI technologies adhere to the same rigorous standards that have made commercial aviation the safest form of transportation."
      },
      {
        question: "How would you detect bias in a crew-pairing LLM?",
        answer: "**Detecting Bias in American Airlines' Crew-Pairing LLM**\n\n**Comprehensive Bias Detection Framework:**\n\n1. **Demographic Parity Analysis:**\n   • Measure assignment distribution across protected attributes (gender, age, race)\n   • Compare desirability scores of assignments (premium routes, holidays, weekends) across groups\n   • Calculate statistical significance of differences using chi-square tests\n   • Benchmark against seniority-normalized expectations\n\n2. **Counterfactual Testing:**\n   • Generate paired test cases that differ only in protected attributes\n   • Submit identical qualifications with varied demographic indicators\n   • Measure divergence in model recommendations between counterparts\n   • Automate continuous testing with synthetic crew profiles\n\n3. **Representation Testing:**\n   • Analyze embedding space to detect demographic clustering\n   • Measure distance between different group representations in the model's latent space\n   • Test for unintended correlations between operational factors and protected attributes\n   • Apply UMAP visualization for interpretable bias detection\n\n4. **Temporal Fairness Evaluation:**\n   • Track assignment quality over time across different groups\n   • Detect compounding disadvantages in sequential pairing assignments\n   • Measure recovery from disruptions by demographic segment\n   • Analyze \"career path\" simulations for long-term impact\n\n5. **Operational Context Assessment:**\n   • Evaluate differential treatment during irregular operations\n   • Measure bias in accommodation of crew preferences\n   • Analyze cancellation impact distribution\n   • Test for contextual fairness in base assignments\n\n**Implementation for American Airlines:**\n\n• **Metrics Dashboard:**\n  - Real-time bias metrics visualization\n  - Drill-down capability for segment analysis\n  - Historical trend tracking\n  - Correlation with operational factors\n\n• **Testing Integration:**\n  - Automated bias test suite in CI/CD pipeline\n  - Pre-deployment bias audit requirements\n  - Continuous monitoring in production\n\n• **Stakeholder Input:**\n  - Feedback mechanisms for crew members\n  - Regular review with union representation\n  - Transparent reporting and remediation process\n\n• **Mitigation Strategy:**\n  - Bias-aware constraints in the optimization model\n  - Fairness-enhancing fine-tuning\n  - Human review process for edge cases\n  - Regular retraining with debiased datasets\n\n**Example Detection Scenario:**\n\nFor an LLM that assists in crew recovery during irregular operations, we would identify potential bias by:  \n\n1. Analyzing if certain demographic groups are disproportionately assigned disrupted trips  \n2. Measuring recovery quality (time to normal schedule) across groups  \n3. Examining preference satisfaction rates during reassignments  \n4. Testing if explanations and communication differ by group\n\nThis framework ensures that as American Airlines leverages AI for crew scheduling, we maintain our commitment to fairness and equal opportunity while optimizing operational efficiency."
      }
    ],
    fundamentals: [
      {
        question: "Compare Ridge vs. Lasso regression for predicting flight delays, including when you would choose each approach.",
        answer: "**Ridge vs. Lasso for Flight Delay Prediction at American Airlines**\n\n**Ridge Regression (L2 Regularization)**\n\n• **How it works**: Adds penalty proportional to the square of coefficient magnitudes (β²)\n• **Effect on coefficients**: Shrinks all coefficients but rarely to exactly zero\n• **Flight delay application**: Maintains influence from all delay factors while reducing overfitting\n\n**Lasso Regression (L1 Regularization)**\n\n• **How it works**: Adds penalty proportional to absolute value of coefficients (|β|)\n• **Effect on coefficients**: Drives some coefficients exactly to zero, performing feature selection\n• **Flight delay application**: Automatically identifies and removes irrelevant delay factors\n\n**When I'd Choose Ridge for AA Delay Prediction:**\n\n• When all potential delay factors likely contribute to some degree (weather, traffic, maintenance, crew availability)\n• When interpretability requires considering all variables, even with small effects\n• When we need to account for complex interactions between many predictors (e.g., network effects across our hub system)\n• When multicollinearity exists among predictors (e.g., related weather metrics or congestion measures)\n• When historical data shows that most factors have some predictive value\n\n**When I'd Choose Lasso for AA Delay Prediction:**\n\n• When building a simplified, interpretable model for operational decision-making\n• When we suspect many factors are noise with no real impact on delays\n• When computationally efficient predictions are needed for real-time applications\n• When creating distinct models for different airports or routes to identify location-specific factors\n• When model output will feed into downstream optimization systems that benefit from sparsity\n\n**Practical Approach at American Airlines:**\n\nI would implement an Elastic Net solution (combining both L1 and L2 penalties) with these steps:\n\n1. Split data by airport hub class (large hubs vs. spoke airports)\n2. Use cross-validation to optimize alpha (mixing parameter) and lambda (regularization strength)\n3. For operational dashboards, favor Lasso-dominant models (higher alpha) for interpretability\n4. For predictive accuracy in irregular operations planning, use Ridge-dominant models (lower alpha)\n5. Validate against historical performance during both normal operations and extreme events (severe weather, ATC disruptions)\n\nThis balanced approach would deliver the most actionable insights for improving American Airlines' on-time performance while maintaining model interpretability for operational stakeholders."
      },
      {
        question: "Compare SHAP values versus built-in feature importance for GBDT models when analyzing customer satisfaction predictors.",
        answer: "**SHAP vs. Built-in Feature Importance for AA Customer Satisfaction Analysis**\n\n**SHAP (SHapley Additive exPlanations) Values**\n\n• **Methodology**: Based on cooperative game theory, distributes prediction credit among features\n• **Mathematical foundation**: Calculates marginal contribution of each feature across all possible feature combinations\n• **Output type**: Local (per-prediction) and global (aggregated) importance values\n\n**Built-in GBDT Feature Importance**\n\n• **Methodology**: Typically measures how often a feature is used in trees and its average gain\n• **Mathematical foundation**: Based on reduction in loss function when splitting on the feature\n• **Output type**: Global importance scores only (no local explanations)\n\n**Advantages of SHAP for American Airlines' Customer Satisfaction Analysis:**\n\n• **Consistency across models**: SHAP provides consistent interpretation regardless of model type, valuable when comparing results across different AA divisions\n\n• **Direction of impact**: Shows whether a feature positively or negatively affects satisfaction, critical for understanding if on-time performance is helping or hurting NPS\n\n• **Interaction detection**: Reveals how features interact (e.g., how delays interact with elite status or rebooking options)\n\n• **Local explanations**: Provides passenger-specific insights (why was THIS customer dissatisfied?), enabling personalized service recovery\n\n• **Regulatory alignment**: More defensible for compliance with transparency requirements\n\n• **Trust building**: More intuitive explanations for frontline staff and executives\n\n**Advantages of Built-in Importance for American Airlines:**\n\n• **Computational efficiency**: Faster to calculate when analyzing millions of satisfaction surveys\n\n• **Implementation simplicity**: Already included in most GBDT packages (XGBoost, LightGBM)\n\n• **Training insight**: Better reflects model training process\n\n• **Consistent with feature selection**: Aligns with how the model determines splits\n\n**Practical Application at American Airlines:**\n\nI would implement a hybrid approach where:\n\n1. **Use built-in importance for initial exploration and model iteration**\n   • Quick feedback during model development\n   • Faster processing for weekly satisfaction trends\n\n2. **Use SHAP for deeper operational insights and action planning**\n   • Detailed analysis of how service disruptions affect different customer segments\n   • Personalized explanation generation for customer service agents\n   • Executive dashboards showing satisfaction drivers by route and aircraft type\n   • Root cause analysis for satisfaction outliers\n\n3. **Specific AA use cases for SHAP:**\n   • Understanding how elite status moderates negative impact of delays\n   • Quantifying the effect of proactive communication during IRROPs\n   • Analyzing how different aircraft configurations affect comfort ratings\n   • Determining optimal compensation offers for service recovery\n\nThis approach maximizes computational efficiency while providing the deeper insights needed to systematically improve American Airlines' customer satisfaction metrics across different passenger segments and service aspects."
      },
      {
        question: "Explain how you would implement K-fold CV for time-series forecasting of departure delays.",
        answer: "**Time-Series K-fold CV for AA Departure Delay Forecasting**\n\n**Challenge with Standard K-fold CV:**\n\nStandard K-fold CV randomly assigns observations to folds, which would cause data leakage in departure delay forecasting due to temporal dependencies. Using future data to predict past events would create unrealistically optimistic performance estimates.\n\n**Proper Implementation for American Airlines:**\n\n1. **Temporal Block Structure:**\n   • Organize AA flight delay data chronologically by departure timestamp\n   • Create consecutive time blocks respecting temporal order\n   • Ensure each block contains sufficient operational cycles (day/week patterns)\n\n2. **Forward-Chaining CV Approach:**\n   • **Initial Training Period**: Start with at least 3 months of historical delay data\n   • **Progressive Testing Windows**: Test on subsequent 2-week periods\n   • **Expanding Window Option**: Incrementally add previous test periods to training set\n   • **Fixed Window Option**: Maintain consistent training window size, sliding forward\n\n**Practical Implementation Example:**\n\n```python\ndef time_aware_cv(delay_data, n_splits=5, test_period='2W', initial_train='3M'):\n    # Sort American Airlines departure data by timestamp\n    delay_data = delay_data.sort_values('departure_datetime')\n    \n    # Create time-based splits considering AA's operational patterns\n    splits = []\n    \n    # Define initial training period (e.g., first 3 months)\n    train_end = delay_data['departure_datetime'].min() + pd.Timedelta(initial_train)\n    \n    for i in range(n_splits):\n        # Define train/test indices respecting time order\n        train_idx = delay_data[delay_data['departure_datetime'] < train_end].index\n        \n        test_start = train_end\n        test_end = test_start + pd.Timedelta(test_period)\n        test_idx = delay_data[(delay_data['departure_datetime'] >= test_start) & \n                             (delay_data['departure_datetime'] < test_end)].index\n        \n        splits.append((train_idx, test_idx))\n        train_end = test_end\n    \n    return splits\n```\n\n**AA-Specific Considerations:**\n\n1. **Seasonality Handling:**\n   • Ensure test periods span different seasonal patterns (summer peak, holiday travel)\n   • Include fold-specific performance metrics to identify seasonal model weaknesses\n\n2. **Operational Changes:**\n   • Account for schedule changes, new routes, or aircraft swaps\n   • Consider adding metadata indicating major operational changes\n\n3. **External Events:**\n   • Flag periods with extreme weather, ATC ground stops, or system outages\n   • Option to exclude or specially weight unusual operational periods\n\n4. **Hub-Specific Variations:**\n   • Stratify validation across different hub types (DFW, CLT, PHX, etc.)\n   • Ensure representation of both hub and spoke airports in each fold\n\n5. **Performance Metrics:**\n   • Track fold-specific RMSE, MAE for each location/time period\n   • Consider operational impact metrics (misclassification costs vary by delay magnitude)\n   • Monitor prediction stability across folds\n\n**Implementation Benefits:**\n\n• Realistic performance estimates that respect the forward-flowing nature of time\n• Better preparation for model degradation over time\n• Insight into how far ahead reliable predictions can be made\n• Identification of temporal patterns affecting model performance\n• Proper foundation for operational implementation in American's delay prediction systems\n\nThis approach ensures our delay forecasting models are evaluated in a way that matches their actual operational use, leading to more reliable predictions that operations controllers and customers can depend on."
      },
      {
        question: "Compare LSTM vs. Prophet for passenger demand forecasting at American Airlines.",
        answer: "**LSTM vs. Prophet for AA Passenger Demand Forecasting**\n\n**LSTM (Long Short-Term Memory)**\n\n• **Architecture**: Deep learning approach using recurrent neural networks with specialized memory cells\n• **Strengths for AA**: Captures complex non-linear patterns and long-term dependencies in booking curves\n• **Input handling**: Can incorporate multiple features (pricing, competitor actions, events, historical load factors)\n\n**Prophet**\n\n• **Architecture**: Decomposition-based approach using additive model with trend, seasonality, and holiday components\n• **Strengths for AA**: Explicitly models multiple seasonality patterns (daily, weekly, yearly) and holiday effects\n• **Input handling**: Primarily univariate but can incorporate known future events as regressors\n\n**Comparative Analysis for American Airlines**\n\n| Factor | LSTM Advantage | Prophet Advantage |\n|--------|----------------|-------------------|\n| **Seasonality** | Learns patterns indirectly from data | Explicitly models multiple seasonal patterns (daily, weekly, yearly) critical for airline demand |\n| **Irregular Events** | Requires substantial examples to learn patterns | Built-in capability to handle holidays and special events like Super Bowl or conventions |\n| **Data Requirements** | Needs large amounts of historical data to perform well | Works reasonably well with limited history (e.g., new routes) |\n| **Multivariate Inputs** | Natively handles multiple input features (pricing tiers, competitive data, macroeconomic indicators) | Primarily designed for univariate forecasting with limited regressor support |\n| **Computational Resources** | Requires significant training time and GPU resources | Lightweight, can be deployed across multiple markets quickly |\n| **Interpretability** | Black-box model, difficult to explain predictions | Decomposable model with clear visualization of trend, seasonality components |\n| **Uncertainty** | Requires custom approaches for prediction intervals | Built-in Bayesian approach for uncertainty estimation |\n| **Maintenance** | Requires regular retraining as patterns evolve | Easier to update and maintain for business analysts |\n\n**Optimal Implementation for American Airlines:**\n\nI would recommend a hybrid, use-case specific approach:\n\n**Use Prophet for:**\n\n• **Network-wide capacity planning** (3-6 month horizon) where seasonality and holidays dominate\n• **New route forecasting** where limited historical data is available\n• **Explicit impact analysis** of schedule changes, holidays, and special events\n• **Executive dashboard reporting** requiring transparent explanations of forecast drivers\n\n**Use LSTM for:**\n\n• **Short-term revenue management** optimization (0-90 days) incorporating multiple pricing and competitor signals\n• **Hub-specific forecasting** where complex connections and traffic flows create non-linear patterns\n• **Dynamic pricing applications** requiring real-time response to market conditions\n• **Integrated forecasting** combining demand, pricing, and operational constraints\n\n**Implementation Strategy:**\n\n1. **Segmented deployment**: Apply Prophet for baseline long-range planning, LSTM for short-term revenue optimization\n\n2. **Ensemble approach**: Weight models differently by market, horizon, and data availability\n\n3. **Transfer learning**: Pre-train LSTM on network-wide patterns, fine-tune for specific markets\n\n4. **Continuous evaluation**: Maintain both models in production, shifting weights based on performance\n\nThis approach maximizes forecast accuracy while maintaining operational feasibility across American Airlines' diverse network of routes and markets."
      }
    ]
  };

  const cramList = [
    "Azure MLflow example \"deploy to AKS\" end-to-end",
    "RAG demo practice (chunking, embedding, vector search, retrieval)",
    "Smart Gating stats (17 hrs/day saved, 1.4M gallons fuel)",
    "FlashAttention, KV cache, LoRA concepts",
    "Azure model management and governance implementation",
    "Airline industry-specific AI benchmarks and KPIs",
    "IT Operations Research applications in airline context"
  ];

  const techFlashcards = [
    {
      question: "What is Retrieval-Augmented Generation (RAG) and why is it useful for enterprise applications?",
      answer: "RAG combines information retrieval with text generation to ground LLM outputs in factual, up-to-date information. It's especially valuable for enterprises because it allows LLMs to access proprietary information, reduces hallucinations, improves factual accuracy, and doesn't require full model retraining when information changes. In an airline context, RAG can help ground responses in specific maintenance manuals, company policies, or operational procedures."
    },
    {
      question: "What are the key components of Azure AI Foundry?",
      answer: "Azure AI Foundry is Microsoft's unified platform for generative AI development. Key components include: 1) Access to foundation models (via Azure OpenAI Service), 2) Tools for prompt engineering and evaluation, 3) Fine-tuning capabilities, 4) RAG implementation tools, 5) Deployment options (serverless APIs, managed compute), and 6) MLOps features for monitoring and governance. American Airlines appears to be standardizing on this platform for its GenAI initiatives."
    },
    {
      question: "What is FlashAttention and why does it matter for edge deployment?",
      answer: "FlashAttention is an optimization algorithm that makes attention computations in transformer models more memory-efficient and faster by reducing memory access costs and using block-wise computation. This is critical for edge deployment (like at airport gates or on mobile devices) because it enables running larger models on memory-constrained devices, improves inference speed for real-time applications, and reduces power consumption—all essential for operational AI systems at American Airlines."
    },
    {
      question: "How does KV-cache optimization improve LLM response times?",
      answer: "KV-cache (Key-Value cache) stores previously computed key and value tensors during transformer inference to avoid redundant computations when generating text autoregressively. This significantly reduces token generation latency by eliminating the need to recompute attention for already processed tokens. For American Airlines' customer-facing or operational applications, this means faster, more responsive AI systems, better user experience, and reduced compute costs."
    },
    {
      question: "What is the Smart Gating initiative at American Airlines?",
      answer: "Smart Gating is American Airlines' AI-driven gate assignment system that optimizes airport operations. It currently saves approximately 17 hours of taxi time per day and reduces fuel consumption by 1.4 million gallons. The system optimizes gate assignments based on multiple factors including arrival/departure times, aircraft type, and ground operations constraints. It represents a successful AI implementation with measurable operational benefits that could be further enhanced with generative AI capabilities."
    },
    {
      question: "What is the difference between blue-green deployment and canary releases?",
      answer: "Blue-green deployment involves maintaining two identical environments (blue=current, green=new) and switching traffic entirely from one to the other after validation. Canary releases gradually route increasing percentages of traffic to the new version. For American Airlines' LLM deployments, blue-green offers safer cutover for critical systems like crew scheduling, while canary releases might be better for customer-facing chatbots where gradual testing with real users is valuable."
    },
    {
      question: "How do vector databases support Retrieval-Augmented Generation?",
      answer: "Vector databases store text as high-dimensional vectors (embeddings) that represent semantic meaning rather than just keywords. In RAG systems, they enable efficient similarity search to find the most relevant documents or passages given a query. For American Airlines, vector databases would store embeddings of maintenance manuals, operational procedures, or customer service policies, allowing LLMs to quickly retrieve and ground their responses in the most relevant company information."
    }
  ];

  const toggleAnswer = (id) => {
    setShowAnswers({
      ...showAnswers,
      [id]: !showAnswers[id]
    });
  };

  const markCompleted = (topic, subtopic) => {
    const key = `${topic}-${subtopic}`;
    setCompletedTopics({
      ...completedTopics,
      [key]: !completedTopics[key]
    });
  };

  const toggleCheckedItem = (key) => {
    setCheckedItems({
      ...checkedItems,
      [key]: !checkedItems[key]
    });
  };

  const getNextFlashcard = () => {
    setFlashcardIndex((prevIndex) => (prevIndex + 1) % techFlashcards.length);
    setShowFlashcardAnswer(false);
  };

  const getPrevFlashcard = () => {
    setFlashcardIndex((prevIndex) => (prevIndex - 1 + techFlashcards.length) % techFlashcards.length);
    setShowFlashcardAnswer(false);
  };

  const toggleFlashcardAnswer = () => {
    setShowFlashcardAnswer(!showFlashcardAnswer);
  };

  const filteredTopics = searchQuery.length > 2 
    ? Object.entries(techTopics).filter(([topicKey, topic]) => {
        const lowerQuery = searchQuery.toLowerCase();
        if (topic.title.toLowerCase().includes(lowerQuery)) return true;
        if (topic.description.toLowerCase().includes(lowerQuery)) return true;
        
        return topic.subtopics.some(subtopic => 
          subtopic.name.toLowerCase().includes(lowerQuery) || 
          subtopic.description.toLowerCase().includes(lowerQuery) ||
          subtopic.keyPoints.some(point => point.toLowerCase().includes(lowerQuery)) ||
          subtopic.airlineContext.toLowerCase().includes(lowerQuery)
        );
      })
    : Object.entries(techTopics);

  const filteredQuestions = searchQuery.length > 2
    ? Object.entries(practiceQuestions).reduce((acc, [category, questions]) => {
        const filtered = questions.filter(q => 
          q.question.toLowerCase().includes(searchQuery.toLowerCase()) || 
          q.answer.toLowerCase().includes(searchQuery.toLowerCase())
        );
        if (filtered.length > 0) acc[category] = filtered;
        return acc;
      }, {})
    : practiceQuestions;

  const TabButton = ({ id, value, activeTab, onChange, children, className }) => (
    <button 
      id={id}
      className={`px-4 py-2 text-sm font-medium ${activeTab === value 
        ? 'bg-blue-600 text-white' 
        : 'bg-gray-200 text-gray-700 hover:bg-gray-300'} ${className || ''}`}
      onClick={() => onChange(value)}
    >
      {children}
    </button>
  );

  const TopicTabs = ({ activeTopic, setActiveTopic }) => (
    <div className="flex mb-4 border-b border-gray-200">
      {Object.entries(techTopics).map(([key, topic]) => (
        <button
          key={key}
          className={`py-2 px-4 text-sm font-medium ${activeTopic === key 
            ? 'border-b-2 border-blue-500 text-blue-600' 
            : 'text-gray-500 hover:text-gray-700'}`}
          onClick={() => setActiveTopic(key)}
        >
          {topic.title}
        </button>
      ))}
    </div>
  );

  return (
    <div className="container mx-auto px-4 py-8 bg-gray-50">
      <div className="flex flex-col md:flex-row justify-between items-start mb-6">
        <div className="bg-blue-600 text-white p-4 rounded-lg shadow-lg transform -rotate-1">
          <h1 className="text-3xl font-bold mb-1">American Airlines Gen AI Interview Prep</h1>
          <p className="text-blue-100">Data Scientist/Sr. Data Scientist position in IT Operations Research and Advanced Analytics</p>
          <div className="flex mt-2">
            <div className="bg-blue-500 rounded-full p-1 mr-2">
              <Award size={18} />
            </div>
            <p className="text-sm">Optimized for visual learners</p>
          </div>
        </div>
        <div className="mt-4 md:mt-0">
          <div className="bg-white rounded-lg shadow-lg p-4 w-full border-t-4 border-blue-600 transform rotate-1">
            <div className="flex items-center mb-2">
              <Award size={20} className="text-blue-600 mr-2" />
              <h5 className="font-medium">Interview Readiness</h5>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-6 mb-3">
              <div 
                className="bg-gradient-to-r from-blue-500 to-purple-600 h-6 rounded-full flex items-center justify-end pr-2 transition-all duration-500"
                style={{ width: `${progress}%` }}
              >
                <span className="text-white text-xs font-bold">{progress}%</span>
              </div>
            </div>
            <div className="flex items-start">
              <BookmarkCheck size={16} className="text-green-500 mr-1 mt-1" />
              <p className="text-gray-500 text-sm">
                Track your preparation across all interview areas
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="mb-6">
        <div className="relative">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <Search size={20} className="text-blue-500" />
          </div>
          <input
            type="text"
            className="block w-full pl-10 pr-3 py-3 border-2 border-blue-300 rounded-lg leading-5 bg-white placeholder-blue-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 text-sm shadow-md"
            placeholder="Search topics, concepts, or questions..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      </div>

      <div className="flex justify-center mb-8 overflow-x-auto">
        <div className="inline-flex rounded-xl shadow-lg p-1 bg-white" role="group">
          <TabButton 
            id="overview-tab" 
            value="overview" 
            activeTab={activeTab} 
            onChange={setActiveTab} 
            className={`rounded-l-lg flex items-center ${activeTab === 'overview' ? 'bg-gradient-to-r from-blue-500 to-blue-600' : ''}`}
          >
            <div className="mr-2">🔍</div> Overview
          </TabButton>
          <TabButton 
            id="topics-tab" 
            value="topics" 
            activeTab={activeTab} 
            onChange={setActiveTab}
            className={`flex items-center ${activeTab === 'topics' ? 'bg-gradient-to-r from-green-500 to-green-600' : ''}`}
          >
            <div className="mr-2">📚</div> Tech Topics
          </TabButton>
          <TabButton 
            id="questions-tab" 
            value="questions" 
            activeTab={activeTab} 
            onChange={setActiveTab}
            className={`flex items-center ${activeTab === 'questions' ? 'bg-gradient-to-r from-purple-500 to-purple-600' : ''}`}
          >
            <div className="mr-2">❓</div> Practice Q&A
          </TabButton>
          <TabButton 
            id="flashcards-tab" 
            value="flashcards" 
            activeTab={activeTab} 
            onChange={setActiveTab}
            className={`flex items-center ${activeTab === 'flashcards' ? 'bg-gradient-to-r from-yellow-500 to-yellow-600' : ''}`}
          >
            <div className="mr-2">🔄</div> Flashcards
          </TabButton>
          <TabButton 
            id="cram-tab" 
            value="cram" 
            activeTab={activeTab} 
            onChange={setActiveTab} 
            className={`rounded-r-lg flex items-center ${activeTab === 'cram' ? 'bg-gradient-to-r from-red-500 to-red-600' : ''}`}
          >
            <div className="mr-2">🚀</div> Cram List
          </TabButton>
        </div>
      </div>

      {activeTab === 'overview' && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="md:col-span-2">
            <div className="bg-white rounded-lg shadow-lg mb-6 border-t-4 border-blue-500 overflow-hidden transform hover:scale-101 transition-transform duration-300">
              <div className="bg-blue-500 text-white px-6 py-3 flex items-center">
                <div className="bg-white p-2 rounded-full mr-3">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 3L20 7.5V16.5L12 21L4 16.5V7.5L12 3Z" fill="#0062cc" />
                    <path d="M12 8L16 10.5V15.5L12 18L8 15.5V10.5L12 8Z" fill="white" />
                  </svg>
                </div>
                <h3 className="text-xl font-bold">American Airlines Gen AI Context</h3>
              </div>
              <div className="p-6">
                <div className="flex items-center bg-blue-50 p-3 rounded-lg mb-4">
                  <svg className="w-12 h-12 mr-3" viewBox="0 0 24 24" fill="none">
                    <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="#0062cc" strokeWidth="2"/>
                    <path d="M12 16V12M12 8H12.01" stroke="#0062cc" strokeWidth="2" strokeLinecap="round"/>
                  </svg>
                  <p>American Airlines is expanding its Gen AI capabilities, with multiple open positions in this area. The Data Scientist/Sr Data Scientist role (Req 78278) is part of this initiative, focusing on applying Gen AI techniques to IT Operations Research and Advanced Analytics.</p>
                </div>
                
                <div className="mb-6">
                  <h5 className="font-semibold mb-3 flex items-center">
                    <svg className="w-5 h-5 mr-2 text-blue-500" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                    Organizational Context
                  </h5>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-blue-50 p-3 rounded-lg flex items-start">
                      <div className="bg-blue-100 p-2 rounded-full mr-2 mt-1">
                        <svg className="w-4 h-4 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
                        </svg>
                      </div>
                      <div>
                        <span className="text-sm font-medium text-blue-800">Team Structure</span>
                        <p className="text-sm">IT Operations Research and Advanced Analytics (OR&AA) Gen AI team</p>
                      </div>
                    </div>
                    <div className="bg-blue-50 p-3 rounded-lg flex items-start">
                      <div className="bg-blue-100 p-2 rounded-full mr-2 mt-1">
                        <svg className="w-4 h-4 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                        </svg>
                      </div>
                      <div>
                        <span className="text-sm font-medium text-blue-800">AI Approach</span>
                        <p className="text-sm">"Governance-first" stance toward AI deployment</p>
                      </div>
                    </div>
                    <div className="bg-blue-50 p-3 rounded-lg flex items-start">
                      <div className="bg-blue-100 p-2 rounded-full mr-2 mt-1">
                        <svg className="w-4 h-4 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                        </svg>
                      </div>
                      <div>
                        <span className="text-sm font-medium text-blue-800">Priority Projects</span>
                        <p className="text-sm">Smart Gating, crew recovery, and predictive maintenance</p>
                      </div>
                    </div>
                    <div className="bg-blue-50 p-3 rounded-lg flex items-start">
                      <div className="bg-blue-100 p-2 rounded-full mr-2 mt-1">
                        <svg className="w-4 h-4 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 14v3m4-3v3m4-3v3M3 21h18M3 10h18M3 7l9-4 9 4M4 10h16v11H4V10z" />
                        </svg>
                      </div>
                      <div>
                        <span className="text-sm font-medium text-blue-800">Technology</span>
                        <p className="text-sm">Microsoft Azure, particularly Azure AI Foundry</p>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="bg-gradient-to-r from-blue-50 to-purple-50 p-4 rounded-lg">
                  <h5 className="font-semibold mb-3 text-blue-800 flex items-center">
                    <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                    </svg>
                    Technical Focus Areas
                  </h5>
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-white rounded-lg p-3 shadow-sm flex items-center">
                      <div className="bg-blue-100 text-blue-700 p-2 rounded-full mr-3 flex-shrink-0">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M12 18h.01M8 21h8a2 2 0 002-2V5a2 2 0 00-2-2H8a2 2 0 00-2 2v14a2 2 0 002 2z" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                      </div>
                      <span>LLM development (OpenAI GPT, Google Gemini, Llama)</span>
                    </div>
                    <div className="bg-white rounded-lg p-3 shadow-sm flex items-center">
                      <div className="bg-green-100 text-green-700 p-2 rounded-full mr-3 flex-shrink-0">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                      </div>
                      <span>RAG and Prompt Engineering</span>
                    </div>
                    <div className="bg-white rounded-lg p-3 shadow-sm flex items-center">
                      <div className="bg-purple-100 text-purple-700 p-2 rounded-full mr-3 flex-shrink-0">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2z" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                      </div>
                      <span>MLOps and model deployment on Azure</span>
                    </div>
                    <div className="bg-white rounded-lg p-3 shadow-sm flex items-center">
                      <div className="bg-red-100 text-red-700 p-2 rounded-full mr-3 flex-shrink-0">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                      </div>
                      <span>Responsible AI governance and bias mitigation</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="bg-white rounded-lg shadow-lg border-t-4 border-purple-500">
              <div className="bg-purple-500 text-white px-6 py-3 flex items-center">
                <div className="bg-white p-2 rounded-full mr-3">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M9 5H7C5.89543 5 5 5.89543 5 7V19C5 20.1046 5.89543 21 7 21H17C18.1046 21 19 20.1046 19 19V7C19 5.89543 18.1046 5 17 5H15M9 5C9 6.10457 9.89543 7 11 7H13C14.1046 7 15 6.10457 15 5M9 5C9 3.89543 9.89543 3 11 3H13C14.1046 3 15 3.89543 15 5" stroke="#a855f7" strokeWidth="2"/>
                    <path d="M9 12H15M9 16H15" stroke="#a855f7" strokeWidth="2" strokeLinecap="round"/>
                  </svg>
                </div>
                <h3 className="text-xl font-bold">Interview Structure</h3>
              </div>
              <div className="p-6">
                <div className="flex mb-6">
                  <div className="w-full">
                    <div className="relative mb-8">
                      <div className="absolute top-0 h-full border-r-2 border-purple-300 left-6"></div>
                      <div className="flex items-center mb-4">
                        <div className="z-10 flex items-center justify-center w-12 h-12 bg-purple-200 rounded-full">
                          <span className="text-purple-700 text-xl font-bold">1</span>
                        </div>
                        <div className="flex-grow pl-4">
                          <h5 className="font-bold text-lg text-purple-700">Technical Panel <span className="text-purple-500 text-sm font-normal">(60 minutes)</span></h5>
                        </div>
                      </div>
                      
                      <div className="ml-16 mb-6">
                        <div className="grid grid-cols-3 gap-2">
                          <div className="bg-purple-50 p-3 rounded-lg text-center border border-purple-100">
                            <div className="bg-purple-100 rounded-full w-8 h-8 flex items-center justify-center mx-auto mb-2">
                              <svg className="w-4 h-4 text-purple-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                              </svg>
                            </div>
                            <p className="text-xs font-medium">LLM & Gen AI</p>
                          </div>
                          <div className="bg-purple-50 p-3 rounded-lg text-center border border-purple-100">
                            <div className="bg-purple-100 rounded-full w-8 h-8 flex items-center justify-center mx-auto mb-2">
                              <svg className="w-4 h-4 text-purple-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                              </svg>
                            </div>
                            <p className="text-xs font-medium">Cloud & MLOps</p>
                          </div>
                          <div className="bg-purple-50 p-3 rounded-lg text-center border border-purple-100">
                            <div className="bg-purple-100 rounded-full w-8 h-8 flex items-center justify-center mx-auto mb-2">
                              <svg className="w-4 h-4 text-purple-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 8v8m-4-5v5m-4-2v2m-2 4h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                              </svg>
                            </div>
                            <p className="text-xs font-medium">Airline Ops Analytics</p>
                          </div>
                          <div className="bg-purple-50 p-3 rounded-lg text-center border border-purple-100">
                            <div className="bg-purple-100 rounded-full w-8 h-8 flex items-center justify-center mx-auto mb-2">
                              <svg className="w-4 h-4 text-purple-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                              </svg>
                            </div>
                            <p className="text-xs font-medium">Responsible AI</p>
                          </div>
                          <div className="bg-purple-50 p-3 rounded-lg text-center border border-purple-100">
                            <div className="bg-purple-100 rounded-full w-8 h-8 flex items-center justify-center mx-auto mb-2">
                              <svg className="w-4 h-4 text-purple-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
                              </svg>
                            </div>
                            <p className="text-xs font-medium">DS Fundamentals</p>
                          </div>
                        </div>
                      </div>
                      
                      <div className="flex items-center mb-4">
                        <div className="z-10 flex items-center justify-center w-12 h-12 bg-purple-200 rounded-full">
                          <span className="text-purple-700 text-xl font-bold">2</span>
                        </div>
                        <div className="flex-grow pl-4">
                          <h5 className="font-bold text-lg text-purple-700">Coding Assessment <span className="text-purple-500 text-sm font-normal">(30 minutes)</span></h5>
                        </div>
                      </div>
                      
                      <div className="ml-16 mb-6">
                        <div className="grid grid-cols-2 gap-2">
                          <div className="bg-purple-50 p-3 rounded-lg text-center border border-purple-100">
                            <div className="bg-purple-100 rounded-full w-8 h-8 flex items-center justify-center mx-auto mb-2">
                              <svg className="w-4 h-4 text-purple-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
                              </svg>
                            </div>
                            <p className="text-xs font-medium">Data Wrangling & SQL</p>
                          </div>
                          <div className="bg-purple-50 p-3 rounded-lg text-center border border-purple-100">
                            <div className="bg-purple-100 rounded-full w-8 h-8 flex items-center justify-center mx-auto mb-2">
                              <svg className="w-4 h-4 text-purple-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                              </svg>
                            </div>
                            <p className="text-xs font-medium">LLM/RAG Implementation</p>
                          </div>
                          <div className="bg-purple-50 p-3 rounded-lg text-center border border-purple-100">
                            <div className="bg-purple-100 rounded-full w-8 h-8 flex items-center justify-center mx-auto mb-2">
                              <svg className="w-4 h-4 text-purple-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                              </svg>
                            </div>
                            <p className="text-xs font-medium">MLOps Deployment Scripts</p>
                          </div>
                          <div className="bg-purple-50 p-3 rounded-lg text-center border border-purple-100">
                            <div className="bg-purple-100 rounded-full w-8 h-8 flex items-center justify-center mx-auto mb-2">
                              <svg className="w-4 h-4 text-purple-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                              </svg>
                            </div>
                            <p className="text-xs font-medium">Optimization Algorithms</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="bg-gradient-to-r from-purple-100 to-indigo-100 p-4 rounded-lg flex items-center">
                  <div className="bg-purple-200 p-2 rounded-full mr-3">
                    <svg className="w-6 h-6 text-purple-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                  <div>
                    <strong className="text-purple-800">Interview Approach:</strong> AA panels typically prefer realistic, aviation-focused scenarios over abstract brain teasers. Be prepared to connect technical concepts directly to airline operations challenges.
                  </div>
                </div>
              </div>
            </div>
          </div>
          <div className="md:col-span-1">
            <div className="bg-white rounded-lg shadow-lg mb-6 border-t-4 border-green-500 transform hover:scale-101 transition-transform duration-300">
              <div className="bg-green-500 text-white px-6 py-3 flex items-center">
                <div className="bg-white p-2 rounded-full mr-3">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M16 7C16 9.20914 14.2091 11 12 11C9.79086 11 8 9.20914 8 7C8 4.79086 9.79086 3 12 3C14.2091 3 16 4.79086 16 7Z" stroke="#16a34a" strokeWidth="2"/>
                    <path d="M12 14C8.13401 14 5 17.134 5 21H19C19 17.134 15.866 14 12 14Z" stroke="#16a34a" strokeWidth="2"/>
                  </svg>
                </div>
                <h3 className="text-xl font-bold">About the Interviewer</h3>
              </div>
              <div className="p-6">
                <div className="flex mb-4">
                  <div className="flex-shrink-0 bg-green-100 rounded-full p-3 mr-4">
                    <svg className="w-14 h-14 text-green-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                      <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" />
                    </svg>
                  </div>
                  <div>
                    <h4 className="font-bold text-xl text-green-800">Varun Khemani, PhD</h4>
                    <p className="text-green-700">Senior Data Scientist in American Airlines' OR&AA group</p>
                  </div>
                </div>
                
                <div className="bg-green-50 rounded-lg p-4 mb-4">
                  <h5 className="font-semibold text-green-800 mb-2">Background & Expertise</h5>
                  <div className="grid grid-cols-1 gap-3">
                    <div className="flex items-start">
                      <div className="bg-green-200 p-1.5 rounded-full mr-2 mt-0.5">
                        <svg className="w-3.5 h-3.5 text-green-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                      </div>
                      <div>
                        <span className="text-sm font-medium text-green-900">PhD in Reliability Engineering</span>
                      </div>
                    </div>
                    <div className="flex items-start">
                      <div className="bg-green-200 p-1.5 rounded-full mr-2 mt-0.5">
                        <svg className="w-3.5 h-3.5 text-green-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                      </div>
                      <div>
                        <span className="text-sm font-medium text-green-900">Master's in Industrial Engineering</span>
                      </div>
                    </div>
                    <div className="flex items-start">
                      <div className="bg-green-200 p-1.5 rounded-full mr-2 mt-0.5">
                        <svg className="w-3.5 h-3.5 text-green-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                      </div>
                      <div>
                        <span className="text-sm font-medium text-green-900">Data Science, AI/ML, Operations Research</span>
                      </div>
                    </div>
                    <div className="flex items-start">
                      <div className="bg-green-200 p-1.5 rounded-full mr-2 mt-0.5">
                        <svg className="w-3.5 h-3.5 text-green-700" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                      </div>
                      <div>
                        <span className="text-sm font-medium text-green-900">Previously focused on Predictive Maintenance</span>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="bg-yellow-50 rounded-lg p-4 border-l-4 border-yellow-400">
                  <p className="text-sm text-yellow-800">
                    <span className="font-bold">What to expect:</span> Questions that test fundamental understanding and mathematical intuition, with emphasis on robust, reliable, and maintainable AI solutions.
                  </p>
                </div>
              </div>
            </div>
            
            <div className="bg-white rounded-lg shadow-lg mb-6 border-t-4 border-red-500 transform hover:scale-101 transition-transform duration-300">
              <div className="bg-red-500 text-white px-6 py-3 flex items-center">
                <div className="bg-white p-2 rounded-full mr-3">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M9 12L11 14L15 10M20.618 5.984C17.45 5.984 14.895 5.984 13.417 5.984C10.76 5.984 9.21 3.25 6.54 3.25C5.241 3.25 3.25 3.677 3.25 6.596C3.25 9.516 3.25 14.443 3.25 16.667C3.25 19.12 4.928 20.75 7.375 20.75C10.3 20.75 15.85 20.75 18.042 20.75C19.834 20.75 20.75 18.334 20.75 16.667C20.75 15 20.75 9.35 20.75 5.984" stroke="#dc2626" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </div>
                <h3 className="text-xl font-bold">Key Success Factors</h3>
              </div>
              <div className="p-6">
                <ul className="space-y-3">
                  <li className="flex items-center bg-gray-50 p-3 rounded-lg shadow-sm">
                    <div className="bg-red-100 p-2 rounded-full mr-3">
                      <svg className="w-5 h-5 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                      </svg>
                    </div>
                    <span>Demonstrate hands-on LLM development experience</span>
                  </li>
                  <li className="flex items-center bg-gray-50 p-3 rounded-lg shadow-sm">
                    <div className="bg-blue-100 p-2 rounded-full mr-3">
                      <svg className="w-5 h-5 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                      </svg>
                    </div>
                    <span>Show proficiency with Azure AI services</span>
                  </li>
                  <li className="flex items-center bg-gray-50 p-3 rounded-lg shadow-sm">
                    <div className="bg-green-100 p-2 rounded-full mr-3">
                      <svg className="w-5 h-5 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                      </svg>
                    </div>
                    <span>Connect technical concepts to airline operations</span>
                  </li>
                  <li className="flex items-center bg-gray-50 p-3 rounded-lg shadow-sm">
                    <div className="bg-yellow-100 p-2 rounded-full mr-3">
                      <svg className="w-5 h-5 text-yellow-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                      </svg>
                    </div>
                    <span>Emphasize model reliability and governance</span>
                  </li>
                  <li className="flex items-center bg-gray-50 p-3 rounded-lg shadow-sm">
                    <div className="bg-purple-100 p-2 rounded-full mr-3">
                      <svg className="w-5 h-5 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                      </svg>
                    </div>
                    <span>Showcase both ML and operations research thinking</span>
                  </li>
                  <li className="flex items-center bg-gray-50 p-3 rounded-lg shadow-sm">
                    <div className="bg-indigo-100 p-2 rounded-full mr-3">
                      <svg className="w-5 h-5 text-indigo-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </div>
                    <span>Quantify business impact of AI solutions</span>
                  </li>
                </ul>
              </div>
            </div>
            
            <div className="bg-gradient-to-r from-blue-500 to-blue-700 rounded-lg shadow-lg border border-blue-300 transform rotate-1">
              <div className="p-6 text-white">
                <h3 className="text-xl font-bold mb-4 flex items-center">
                  <svg className="w-6 h-6 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  Bottom Line
                </h3>
                <p className="text-blue-100 italic font-medium">
                  "Expect questions that prove you can design, govern and deploy LLM systems on Azure while tying them to concrete airline ROI. Practice explaining each choice (algorithm, infra, metric) in both tech depth and business English."
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'topics' && (
        <div>
          <div className="bg-white rounded-lg shadow-lg p-4 mb-6">
            <div className="flex overflow-x-auto space-x-2 pb-2">
              {Object.entries(techTopics).map(([key, topic]) => {
                const isActive = activeTopic === key;
                let bgColor, textColor, borderColor, iconBg;
                
                switch(key) {
                  case 'llm':
                    bgColor = isActive ? 'bg-blue-500' : 'bg-blue-100';
                    textColor = isActive ? 'text-white' : 'text-blue-700';
                    borderColor = isActive ? 'border-blue-600' : 'border-blue-200';
                    iconBg = 'bg-blue-100';
                    break;
                  case 'mlops':
                    bgColor = isActive ? 'bg-green-500' : 'bg-green-100';
                    textColor = isActive ? 'text-white' : 'text-green-700';
                    borderColor = isActive ? 'border-green-600' : 'border-green-200';
                    iconBg = 'bg-green-100';
                    break;
                  case 'airline':
                    bgColor = isActive ? 'bg-yellow-500' : 'bg-yellow-100';
                    textColor = isActive ? 'text-white' : 'text-yellow-700';
                    borderColor = isActive ? 'border-yellow-600' : 'border-yellow-200';
                    iconBg = 'bg-yellow-100';
                    break;
                  case 'governance':
                    bgColor = isActive ? 'bg-red-500' : 'bg-red-100';
                    textColor = isActive ? 'text-white' : 'text-red-700';
                    borderColor = isActive ? 'border-red-600' : 'border-red-200';
                    iconBg = 'bg-red-100';
                    break;
                  case 'fundamentals':
                    bgColor = isActive ? 'bg-purple-500' : 'bg-purple-100';
                    textColor = isActive ? 'text-white' : 'text-purple-700';
                    borderColor = isActive ? 'border-purple-600' : 'border-purple-200';
                    iconBg = 'bg-purple-100';
                    break;
                  default:
                    bgColor = isActive ? 'bg-gray-500' : 'bg-gray-100';
                    textColor = isActive ? 'text-white' : 'text-gray-700';
                    borderColor = isActive ? 'border-gray-600' : 'border-gray-200';
                    iconBg = 'bg-gray-100';
                }
                
                let icon;
                switch(key) {
                  case 'llm':
                    icon = (
                      <svg className={`w-5 h-5 ${isActive ? 'text-white' : 'text-blue-700'}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    );
                    break;
                  case 'mlops':
                    icon = (
                      <svg className={`w-5 h-5 ${isActive ? 'text-white' : 'text-green-700'}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    );
                    break;
                  case 'airline':
                    icon = (
                      <svg className={`w-5 h-5 ${isActive ? 'text-white' : 'text-yellow-700'}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    );
                    break;
                  case 'governance':
                    icon = (
                      <svg className={`w-5 h-5 ${isActive ? 'text-white' : 'text-red-700'}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    );
                    break;
                  case 'fundamentals':
                    icon = (
                      <svg className={`w-5 h-5 ${isActive ? 'text-white' : 'text-purple-700'}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    );
                    break;
                  default:
                    icon = <div className="w-5 h-5"></div>;
                }
                
                return (
                  <button
                    key={key}
                    className={`min-w-max flex-shrink-0 px-5 py-3 rounded-lg flex items-center space-x-2 font-medium ${bgColor} ${textColor} border-2 ${borderColor} transition-all transform ${isActive ? 'scale-105 shadow-md' : 'hover:scale-102'}`}
                    onClick={() => setActiveTopic(key)}
                  >
                    <div className={`flex-shrink-0 ${isActive ? 'text-white' : ''}`}>
                      {icon}
                    </div>
                    <span>{topic.title}</span>
                  </button>
                );
              })}
            </div>
          </div>
          
          <div className="mt-6">
            <div className="flex items-center mb-6">
              {/* Topic icon and title with visual styling based on topic */}
              {(() => {
                let color, bgGradient, icon;
                switch(activeTopic) {
                  case 'llm':
                    color = 'blue';
                    bgGradient = 'from-blue-500 to-blue-700';
                    icon = (
                      <svg className="w-8 h-8 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    );
                    break;
                  case 'mlops':
                    color = 'green';
                    bgGradient = 'from-green-500 to-green-700';
                    icon = (
                      <svg className="w-8 h-8 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    );
                    break;
                  case 'airline':
                    color = 'yellow';
                    bgGradient = 'from-yellow-500 to-yellow-600';
                    icon = (
                      <svg className="w-8 h-8 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    );
                    break;
                  case 'governance':
                    color = 'red';
                    bgGradient = 'from-red-500 to-red-700';
                    icon = (
                      <svg className="w-8 h-8 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    );
                    break;
                  case 'fundamentals':
                    color = 'purple';
                    bgGradient = 'from-purple-500 to-purple-700';
                    icon = (
                      <svg className="w-8 h-8 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    );
                    break;
                  default:
                    color = 'gray';
                    bgGradient = 'from-gray-500 to-gray-700';
                    icon = <div className="w-8 h-8"></div>;
                }
                
                return (
                  <div className={`bg-gradient-to-r ${bgGradient} px-6 py-3 rounded-xl shadow-lg flex items-center space-x-3 transform -rotate-1`}>
                    <div className={`bg-${color}-400 p-2 rounded-full`}>
                      {icon}
                    </div>
                    <div>
                      <h3 className="text-2xl font-bold text-white">{techTopics[activeTopic].title}</h3>
                      <p className={`text-${color}-100`}>{techTopics[activeTopic].description}</p>
                    </div>
                  </div>
                );
              })()}
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {techTopics[activeTopic].subtopics.map((subtopic, idx) => {
                // Determine styling based on active topic
                let headerBg, iconBg, iconColor, borderColor;
                switch(activeTopic) {
                  case 'llm':
                    headerBg = 'bg-blue-50';
                    iconBg = 'bg-blue-100';
                    iconColor = 'text-blue-600';
                    borderColor = 'border-blue-200';
                    break;
                  case 'mlops':
                    headerBg = 'bg-green-50';
                    iconBg = 'bg-green-100';
                    iconColor = 'text-green-600';
                    borderColor = 'border-green-200';
                    break;
                  case 'airline':
                    headerBg = 'bg-yellow-50';
                    iconBg = 'bg-yellow-100';
                    iconColor = 'text-yellow-600';
                    borderColor = 'border-yellow-200';
                    break;
                  case 'governance':
                    headerBg = 'bg-red-50';
                    iconBg = 'bg-red-100';
                    iconColor = 'text-red-600';
                    borderColor = 'border-red-200';
                    break;
                  case 'fundamentals':
                    headerBg = 'bg-purple-50';
                    iconBg = 'bg-purple-100';
                    iconColor = 'text-purple-600';
                    borderColor = 'border-purple-200';
                    break;
                  default:
                    headerBg = 'bg-gray-50';
                    iconBg = 'bg-gray-100';
                    iconColor = 'text-gray-600';
                    borderColor = 'border-gray-200';
                }
                
                return (
                  <div key={idx} className={`bg-white rounded-lg shadow-lg border ${borderColor} overflow-hidden transform hover:shadow-xl transition-all duration-300`}>
                    <div className={`${headerBg} px-4 py-3 flex justify-between items-center`}>
                      <div className="flex items-center">
                        <div className={`${iconBg} p-2 rounded-full mr-3`}>
                          <svg className={`w-5 h-5 ${iconColor}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                        </div>
                        <h5 className="font-bold">{subtopic.name}</h5>
                      </div>
                      <button 
                        className={`p-1.5 rounded-full ${completedTopics[`${activeTopic}-${idx}`] 
                          ? iconColor + " bg-white shadow" 
                          : "text-gray-400 hover:bg-gray-100"}`}
                        onClick={() => markCompleted(activeTopic, idx)}
                      >
                        <CheckCircle size={20} />
                      </button>
                    </div>
                    <div className="p-5">
                      <div className="bg-gray-50 rounded-lg p-4 mb-4">
                        <p>{subtopic.description}</p>
                      </div>
                      
                      <h6 className="font-semibold mb-3 flex items-center">
                        <svg className={`w-5 h-5 mr-2 ${iconColor}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                        Key Points
                      </h6>
                      <div className="space-y-2 mb-5">
                        {subtopic.keyPoints.map((point, pidx) => (
                          <div key={pidx} className="flex items-start">
                            <div className={`${iconBg} p-1 rounded-full mr-2 mt-1`}>
                              <svg className={`w-3 h-3 ${iconColor}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                              </svg>
                            </div>
                            <p>{point}</p>
                          </div>
                        ))}
                      </div>
                      
                      <div className={`${headerBg} rounded-lg p-4 border ${borderColor}`}>
                        <h6 className="font-semibold mb-2 flex items-center">
                          <svg className={`w-5 h-5 mr-2 ${iconColor}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                          </svg>
                          American Airlines Context
                        </h6>
                        <div className="flex items-start">
                          <div className={`flex-shrink-0 ${iconBg} mt-1 p-2 rounded-full mr-3`}>
                            <svg className={`w-5 h-5 ${iconColor}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                            </svg>
                          </div>
                          <p className="italic">{subtopic.airlineContext}</p>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {activeTab === 'questions' && (
        <div>
          <TopicTabs activeTopic={activeTopic} setActiveTopic={setActiveTopic} />
          
          <div className="mt-6">
            <h3 className="text-xl font-bold mb-2">{techTopics[activeTopic].title} Practice Questions</h3>
            <p className="text-gray-600 mb-6">Practice answering these questions to prepare for your American Airlines interview.</p>
            
            {practiceQuestions[activeTopic].map((q, idx) => (
              <div key={idx} className="bg-white rounded-lg shadow mb-6">
                <div className="bg-gray-50 px-6 py-4 border-b">
                  <h5 className="font-bold">{q.question}</h5>
                </div>
                <div className="p-6">
                  <div className="mb-4">
                    <button 
                      className={`w-full py-2 px-4 border rounded-md ${showAnswers[`${activeTopic}-${idx}`] 
                        ? "border-blue-300 text-blue-600" 
                        : "border-gray-300 bg-blue-600 text-white"}`}
                      onClick={() => toggleAnswer(`${activeTopic}-${idx}`)}
                    >
                      {showAnswers[`${activeTopic}-${idx}`] ? "Hide Answer" : "Show Answer"}
                    </button>
                  </div>
                  
                  {showAnswers[`${activeTopic}-${idx}`] && (
                    <div className="mt-4">
                      <h6 className="font-semibold mb-2">Model Answer:</h6>
                      <div className="border rounded-md p-4 bg-gray-50" style={{ whiteSpace: 'pre-wrap' }}>
                        {q.answer}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {activeTab === 'flashcards' && (
        <div className="flex justify-center">
          <div className="w-full max-w-4xl">
            {/* Progress bar showing position in flashcard deck */}
            <div className="mb-6 bg-white rounded-lg shadow-lg p-4">
              <div className="flex justify-between text-sm text-gray-600 mb-2">
                <div>First Card</div>
                <div>Progress: {Math.round(((flashcardIndex + 1) / techFlashcards.length) * 100)}%</div>
                <div>Last Card</div>
              </div>
              <div className="relative w-full h-3 bg-gray-200 rounded-full overflow-hidden">
                <div 
                  className="absolute top-0 left-0 h-full bg-gradient-to-r from-blue-400 to-purple-500 rounded-full transition-all duration-500 ease-in-out"
                  style={{ width: `${((flashcardIndex + 1) / techFlashcards.length) * 100}%` }}
                ></div>
                {techFlashcards.map((_, idx) => (
                  <div 
                    key={idx} 
                    className={`absolute top-0.5 w-2 h-2 rounded-full transition-all duration-300 cursor-pointer ${flashcardIndex === idx ? 'bg-white' : 'bg-gray-400'}`}
                    style={{ left: `calc(${(idx / (techFlashcards.length - 1)) * 100}% - 4px)` }}
                    onClick={() => {
                      setFlashcardIndex(idx);
                      setShowFlashcardAnswer(false);
                    }}
                  ></div>
                ))}
              </div>
            </div>
            
            {/* Main flashcard with 3D effect */}
            <div className="perspective-1000">
              <div 
                className={`relative transform-style-3d transition-transform duration-1000 w-full ${showFlashcardAnswer ? 'rotate-y-180' : ''}`}
                style={{ height: '450px' }}
              >
                {/* Front of card (question) */}
                <div 
                  className={`absolute w-full h-full backface-hidden bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl shadow-2xl p-8 flex flex-col justify-between transition-all transform ${showFlashcardAnswer ? 'opacity-0' : 'opacity-100'}`}
                  onClick={toggleFlashcardAnswer}
                  style={{ cursor: 'pointer' }}
                >
                  <div className="absolute top-4 left-4 bg-white bg-opacity-20 rounded-full px-3 py-1 text-sm text-white">
                    Card {flashcardIndex + 1} of {techFlashcards.length}
                  </div>
                  
                  <div className="absolute top-4 right-4">
                    <svg className="w-10 h-10 text-white opacity-30" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                  
                  <div className="flex-1 flex items-center justify-center px-4">
                    <h4 className="text-2xl md:text-3xl font-bold text-center text-white">{techFlashcards[flashcardIndex].question}</h4>
                  </div>
                  
                  <div className="text-center text-white opacity-70 flex items-center justify-center">
                    <svg className="w-5 h-5 mr-2 animate-bounce" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                    </svg>
                    Tap to see answer
                  </div>
                </div>
                
                {/* Back of card (answer) */}
                <div 
                  className={`absolute w-full h-full backface-hidden bg-white rotate-y-180 rounded-xl shadow-2xl overflow-hidden transition-all transform ${showFlashcardAnswer ? 'opacity-100' : 'opacity-0'}`}
                  onClick={toggleFlashcardAnswer}
                  style={{ cursor: 'pointer' }}
                >
                  <div className="absolute top-0 left-0 right-0 h-12 bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-between px-4">
                    <div className="text-white font-medium">Answer</div>
                    <div className="text-white text-sm">Card {flashcardIndex + 1} of {techFlashcards.length}</div>
                  </div>
                  
                  <div className="h-full pt-16 pb-12 px-6 overflow-auto">
                    <div className="bg-blue-50 border-l-4 border-blue-500 p-4 mb-4">
                      <p className="text-blue-800 font-medium text-lg">{techFlashcards[flashcardIndex].question}</p>
                    </div>
                    
                    <div className="prose max-w-none">
                      <p>{techFlashcards[flashcardIndex].answer}</p>
                    </div>
                  </div>
                  
                  <div className="absolute bottom-0 left-0 right-0 h-12 bg-gray-100 flex items-center justify-center">
                    <span className="text-gray-600 flex items-center">
                      <svg className="w-5 h-5 mr-2 animate-bounce transform rotate-180" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                      </svg>
                      Tap to see question
                    </span>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Navigation controls with visual design */}
            <div className="flex justify-between mt-8">
              <button 
                className="flex items-center px-6 py-3 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg shadow-md transform transition-transform hover:scale-105 hover:shadow-lg"
                onClick={getPrevFlashcard}
              >
                <ChevronLeft size={20} className="mr-2" /> Previous Card
              </button>
              <button 
                className="flex items-center px-6 py-3 bg-gradient-to-r from-purple-500 to-purple-600 text-white rounded-lg shadow-md transform transition-transform hover:scale-105 hover:shadow-lg"
                onClick={() => {
                  setShowFlashcardAnswer(!showFlashcardAnswer);
                }}
              >
                {showFlashcardAnswer ? "Show Question" : "Show Answer"}
              </button>
              <button 
                className="flex items-center px-6 py-3 bg-gradient-to-r from-blue-600 to-blue-500 text-white rounded-lg shadow-md transform transition-transform hover:scale-105 hover:shadow-lg"
                onClick={getNextFlashcard}
              >
                Next Card <ChevronRight size={20} className="ml-2" />
              </button>
            </div>
            
            {/* Keyboard shortcuts hint */}
            <div className="text-center mt-6 text-gray-500 text-sm">
              Tip: Use the flashcards to memorize key concepts like RAG, Azure AI Foundry, and Smart Gating stats
            </div>
          </div>
        </div>
      )}

      {activeTab === 'cram' && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="md:col-span-2">
            <div className="bg-white rounded-lg shadow">
              <div className="bg-yellow-100 text-yellow-800 px-6 py-3 rounded-t-lg">
                <h4 className="font-bold">Last-Minute Cram List</h4>
              </div>
              <div className="p-6">
                <p className="text-lg mb-4">Focus on these high-priority items before your interview:</p>
                <div className="divide-y">
                  {cramList.map((item, idx) => (
                    <div key={idx} className="py-3 flex items-center">
                      <div 
                        className="mr-3 cursor-pointer" 
                        onClick={() => toggleCheckedItem(`cram-${idx}`)}
                      >
                        {checkedItems[`cram-${idx}`] ? (
                          <CheckSquare size={22} className="text-green-600" />
                        ) : (
                          <Square size={22} />
                        )}
                      </div>
                      <div className={checkedItems[`cram-${idx}`] ? "line-through text-gray-400" : ""}>
                        {item}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              <div className="px-6 py-3 bg-gray-50 rounded-b-lg">
                <p className="font-bold">Bottom Line: "Expect questions that prove you can design, govern and deploy LLM systems on Azure while tying them to concrete airline ROI."</p>
              </div>
            </div>
          </div>
          <div className="md:col-span-1">
            <div className="bg-white rounded-lg shadow mb-6">
              <div className="bg-blue-100 text-blue-800 px-6 py-3 rounded-t-lg">
                <h5 className="font-bold">Key Stats to Memorize</h5>
              </div>
              <div className="p-4">
                <ul className="divide-y">
                  <li className="py-2">
                    <strong>Smart Gating:</strong> Saves 17 hours of taxi time per day
                  </li>
                  <li className="py-2">
                    <strong>Fuel Savings:</strong> 1.4 million gallons from Smart Gating
                  </li>
                  <li className="py-2">
                    <strong>IT Operations Team:</strong> Part of OR&AA (Operations Research & Advanced Analytics)
                  </li>
                  <li className="py-2">
                    <strong>Cloud Preference:</strong> Azure AI Foundry services and Azure ML
                  </li>
                  <li className="py-2">
                    <strong>AI Approach:</strong> Governance-first culture with emphasis on safety
                  </li>
                </ul>
              </div>
            </div>
            <div className="bg-white rounded-lg shadow">
              <div className="bg-green-100 text-green-800 px-6 py-3 rounded-t-lg">
                <h5 className="font-bold">Interview Day Checklist</h5>
              </div>
              <div className="p-4">
                <div className="flex items-center mb-3">
                  <input type="checkbox" id="check1" className="mr-2" />
                  <label htmlFor="check1">
                    Review your RAG implementation examples
                  </label>
                </div>
                <div className="flex items-center mb-3">
                  <input type="checkbox" id="check2" className="mr-2" />
                  <label htmlFor="check2">
                    Practice explaining Azure MLflow deployment
                  </label>
                </div>
                <div className="flex items-center mb-3">
                  <input type="checkbox" id="check3" className="mr-2" />
                  <label htmlFor="check3">
                    Refresh on airline operations terminology
                  </label>
                </div>
                <div className="flex items-center mb-3">
                  <input type="checkbox" id="check4" className="mr-2" />
                  <label htmlFor="check4">
                    Prepare 2-3 questions about the team and projects
                  </label>
                </div>
                <div className="flex items-center">
                  <input type="checkbox" id="check5" className="mr-2" />
                  <label htmlFor="check5">
                    Practice connecting AI concepts to airline ROI
                  </label>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
