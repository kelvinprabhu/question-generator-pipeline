# Agentic Flow Architecture

This document outlines the multi-agent architecture and execution flow for the Intent-Based Question Generator.

## High-Level System Architecture

The system operates as a **Multi-Agent Orchestration** where `main.py` coordinates specialized agents and managers to produce high-quality, diverse questions.

```mermaid
graph TD
    subgraph Initialization
        Config[Configuration .env] --> IM[Intent Manager]
        Config --> EM[Embedding Model]
        Config --> SC[Similarity Checker]
    end

    subgraph "Agent 1: Context Generator"
        PM[Persona Manager] -->|Prompt| PA[Persona Agent]
        PA -->|LLM Call| P_LLM[Provider: Groq/Gemini/etc]
        P_LLM -->|Returns Persona| PM
    end

    subgraph "Agent 2: Question Designer"
        IM -->|Sample Intents| QG[Question Generator]
        SC -->|Retrieve Similars| QG
        PM -->|Inject Persona| QG
        
        QG -->|Build Prompt| QA[Question Agent]
        QA -->|Generate| Fallback[Fallback Model Chain]
    end
    
    subgraph Data Persistence
        Fallback -->|JSON Output| Valid[Validation & Dedup]
        Valid -->|Save| CSV[CSV File]
        Valid -->|Save| Mongo[MongoDB]
        Valid -->|Feedback| IM
    end

    style PA fill:#f9f,stroke:#333
    style QA fill:#bbf,stroke:#333
    style Fallback fill:#ff9,stroke:#333
```

---

## Detailed Agent Execution Flow

The core logic revolves around two distinct agentic workflows:

### 1. Persona Generation Flow (Agent 1)
This agent's sole responsibility is to create realistic "User Profiles" (Personas) to ensure diversity.

```mermaid
sequenceDiagram
    participant Main as Main Evaluator
    participant PM as PersonaManager
    participant Agent as Persona Agent (LLM)
    
    Main->>PM: request_persona(context="Angry Farmer")
    PM->>Agent: "Generate detailed Indian farmer persona..."
    Note over Agent: Uses Fallback Model (Groq -> Gemini -> HF)
    Agent-->>PM: Persona Object (Name, Age, Region, Issues)
    PM-->>Main: Return Persona
```

### 2. Question Generation Flow (Agent 2)
This is the primary agent that designs the questions based on the context provided by Agent 1 and the Intent Logic.

```mermaid
sequenceDiagram
    participant Main as Main Loop
    participant IM as IntentManager
    participant SC as SimilarityChecker
    participant QG as QuestionGenerator
    participant Agent as Question Agent (LLM)
    participant Mongo as MongoDB

    loop For Each Batch
        Main->>IM: Sample Intent Mix (e.g., [CropDisease, Weather])
        IM-->>Main: Mix: {CropDisease: 0.7, Weather: 0.3}
        
        Main->>SC: Find Similar Questions("Crop Disease Weather")
        SC-->>Main: Returns Top-5 similar Qs (for few-shot context)
        
        Main->>QG: generate_batch(Persona, Intents, Similars)
        
        rect rgb(240, 248, 255)
            Note right of QG: Prompt Construction
            QG->>QG: Build Prompt (Inject Persona + Intents + Similars)
            QG->>Agent: "Generate 10 confusing questions..."
            
            Note right of Agent: Multi-Provider Fallback Logic
            alt Primary Provider (Groq)
                Agent->>Groq: Request
                Groq-->>Agent: Success
            else Groq Fails (Rate Limit)
                Agent->>Gemini: Request (Fallback 1)
                Gemini-->>Agent: Success
            else Gemini Fails
                Agent->>HuggingFace: Request (Fallback 2)
            end
            
            Agent-->>QG: JSON List of Questions
        end
        
        QG->>SC: check_duplicate(New Question)
        alt Is Duplicate?
            SC-->>QG: True (Reject)
        else Is Unique
            QG->>Mongo: Save Question
            QG->>IM: Update Intent Weights (Evolution)
        end
    end
```

## Multi-Provider Fallback Mechanism

The generic "Agent" wrapper implements a robust fallback mechanism to ensure high availability and rate-limit handling.

```mermaid
stateDiagram-v2
    [*] --> TryGroq
    
    state TryGroq {
        [*] --> CallGroq
        CallGroq --> Success: 200 OK
        CallGroq --> FailGroq: 429/500 Error
    }

    state TryGemini {
        FailGroq --> CallGemini
        CallGemini --> Success: 200 OK
        CallGemini --> FailGemini: 429/500 Error
    }

    state TryHuggingFace {
        FailGemini --> CallHF
        CallHF --> Success: 200 OK
        CallHF --> FailHF: 429/500 Error
    }

    state TryOpenRouter {
        FailHF --> CallOR
        CallOR --> Success: 200 OK
        CallOR --> FailOR: Error
    }

    FailOR --> [*]: Raise Exception (Batch Failed)
    Success --> [*]: Return Result
```
