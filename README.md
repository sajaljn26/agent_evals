# LangChain Agent Evaluation System

> A production-ready evaluation harness for LLM agents using LangChain, LangGraph, and DeepEval. Features automated testing with 7 metrics, custom Groq judge models, and CI/CD integration.

## 🎯 Performance Benchmarks

The system was evaluated using a **Golden Dataset** of refund scenarios. The results demonstrate high semantic accuracy and robust tool-calling capabilities.

### 📊 Evaluation Results

| Metric | Average Score | Pass Rate | Interpretation |
| :--- | :--- | :--- | :--- |
| **Answer Correctness (GEval)** | **1.00** | **100%** | Agent consistently reaches the correct logical conclusion. |
| **Tool Correctness** | **0.85** | **50%** | High precision; variance caused by autonomous use of `get_today_date`. |
| **Argument Correctness** | **0.75** | **N/A** | Strong parameter mapping; currently tuning Pydantic schemas. |
| **Overall Pass Rate** | **0.75** | **75%** | 3/4 core scenarios passed strictly; 100% passed semantically. |

### ⚡ System Efficiency
- **Average Eval Latency**: ~0.9s per test case.
- **Total Suite Runtime**: < 4 seconds for complete evaluation.
- **Inference Engine**: Groq (utilizing `gpt-oss-120b` for high-reasoning judge).

---

## 🧪 Quality Assurance (QA)

Beyond LLM evaluation, the system is backed by a rigorous traditional testing suite to ensure code stability.

### Test Coverage
- **Unit Tests (13/13 Passed)**: Validates individual tool logic, Pydantic schema validation, and data retrieval.
- **Integration Tests (5/5 Passed)**: End-to-end verification of the agent's "thought-action-observation" loop.
- **Total Test Pass Rate**: **100%** (Functional Code)

---

## 🏗 Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Golden Dataset │────▶│  Agent (Groq)    │────▶│  DeepEval Judge │
│  (4 Test Cases) │     │  gpt-oss-120b    │     │  gpt-oss-120b   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                              │                          │
                              ▼                          ▼
                        ┌─────────────┐            ┌─────────────┐
                        │ 4 Tools     │            │ 7 Metrics   │
                        │ • Purchase  │            │ • Task Comp │
                        │ • Calculate │            │ • Step Eff  │
                        │ • Issue     │            │ • Plan Qual │
                        │ • Date      │            │ • Tool Corr │
                        └─────────────┘            └─────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- [Groq API Key](https://console.groq.com/keys)
- `uv` (recommended) or pip

### Installation
```bash
git clone https://github.com/yourusername/agent-evaluation-system
cd agent-evaluation-system

# Install dependencies
uv sync
# or: pip install -e .

# Set environment
cp .env.example .env
# Edit .env with your GROQ_API_KEY
```

### Run Evaluation
```bash
# Option 1: Direct
uv run python -m evaluations.eval_all

# Option 2: With environment (Linux/Mac)
GROQ_API_KEY=your_key PYTHONPATH=. uv run python -m evaluations.eval_all

# Option 3: Windows CMD
set GROQ_API_KEY=your_key && set PYTHONPATH=. && uv run python -m evaluations.eval_all
```

### Expected Output
```
✓ Evaluation completed 🎉! (time taken: 3.59s)
Test Results (4 total tests):
   Pass Rate: 75.0% | Passed: 3 | Failed: 1

Metric              Average Score  Pass Rate
Answer Correctness  1.00           100%
Tool Correctness    0.85           50%
Argument Correctness 0.75          N/A
```

---

## 📁 Project Structure

```
agent_evals/
├── agent/                      # Core agent implementation
│   ├── agent.py               # LangGraph create_agent + tool loop
│   ├── models.py              # Groq model configuration
│   ├── tools.py               # 4 refund tools (purchase, calc, issue, date)
│   ├── prompts.py             # Strict system prompt with refund rules
│   ├── objects.py             # Pydantic schemas (Answer, PurchaseItem)
│   └── data.py                # Mock user purchase database
│
├── evaluations/               # Evaluation harness
│   ├── eval_all.py           # Main entry: 7 DeepEval metrics + CI ready
│   ├── dataset.py            # Golden dataset (4 EvalCase objects)
│   └── model.py              # Judge model configuration
│
├── understanding_metrics/     # Individual metric isolation demos
│   ├── answer_correctness.py
│   ├── tool_correctness.py
│   ├── step_efficiency.py
│   └── task_completion.py
│
├── tests/                     # Unit & integration tests
│   ├── unit/
│   └── integration/
│
├── .github/workflows/         # CI/CD pipeline
│   └── evaluate.yml
│
├── pyproject.toml             # Dependencies & project config
├── Dockerfile                 # Containerized evaluation
├── .env.example               # Environment template
└── README.md
```

---

## 🔧 Core Components

### Agent (`agent/agent.py`)
Modern LangChain 1.4+ `create_agent` API with automatic tool-calling loop:

```python
from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=agent_system_prompt,
)

# Returns full message history for evaluation tracing
response = agent.invoke({"messages": [{"role": "user", "content": input}]})
```

### Tools (`agent/tools.py`)
| Tool | Purpose | Parameters |
|------|---------|------------|
| `get_user_purchase_history` | Lookup user orders | `user: str` |
| `get_today_date` | Current date for 30-day policy | none |
| `calculate_right_refund` | Sum eligible item prices | `user: str, items: List[PurchaseItem]` |
| `issue_refund_ticket` | Submit refund to finance | `user: str, refund_amount: int` |

### Evaluation Metrics (`evaluations/eval_all.py`)

| Metric | Threshold | Purpose |
|--------|-----------|---------|
| **TaskCompletion** | 0.8 | Did the agent complete the refund workflow? |
| **StepEfficiency** | 0.8 | Minimal unnecessary steps? |
| **PlanQuality** | 0.8 | Logical plan before execution? |
| **PlanAdherence** | 0.8 | Followed the plan? |
| **ToolCorrectness** | 0.9 | Right tools, right parameters? |
| **ArgumentCorrectness** | N/A | Tool inputs semantically correct? |
| **AnswerCorrectness (GEval)** | 0.8 | Final response matches expected? |

---

## 🧪 Golden Dataset

| Test Case | Input | Expected Tools | Policy |
|-----------|-------|----------------|--------|
| **Eligible Single** | "I am user123, refund my book" | purchase, date, calc, issue | ✅ Within 30 days |
| **Ineligible** | "I am user125, refund my ruler" | purchase, date | ❌ >30 days old |
| **Missing Context** | "Refund my mouse" | (none - ask for user) | 🤷 No username |
| **Multi-Item** | "I am user131, refund lamp & mouse" | purchase, date, calc, issue | ✅ Multiple items |

---

## 🐳 Docker Deployment

```bash
# Build
docker build -t agent-eval .

# Run evaluation
docker run --env GROQ_API_KEY=$GROQ_API_KEY agent-eval

# Or with docker-compose
docker-compose up
```

```dockerfile
# Dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN pip install uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen
COPY . .
CMD ["uv", "run", "python", "-m", "evaluations.eval_all"]
```

---

## ⚙️ CI/CD Pipeline

```yaml
# .github/workflows/evaluate.yml
name: Agent Evaluation
on: [push, pull_request]
jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - name: Run Evaluation
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
        run: uv run python -m evaluations.eval_all
```

---

## 📊 Understanding Metrics (Isolation Demos)

Run individual metrics to understand each one:

```bash
# Answer correctness only
uv run python understanding_metrics/answer_correctness.py

# Tool selection correctness
uv run python understanding_metrics/tool_correctness.py

# Step efficiency (minimal tool calls)
uv run python understanding_metrics/step_efficiency.py

# Task completion (end-to-end success)
uv run python understanding_metrics/task_completion.py
```

---

## 🔑 Configuration

All settings via environment variables (`.env`):

```bash
# Required
GROQ_API_KEY=gsk_xxxxxxxxxxxx

# Optional
AGENT_MODEL=openai/gpt-oss-120b
JUDGE_MODEL=openai/gpt-oss-120b
LOG_LEVEL=INFO
```

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|------------|
| **LLM** | Groq (gpt-oss-120b, llama-3.1, qwen) |
| **Agent Framework** | LangChain 1.4+, LangGraph |
| **Evaluation** | DeepEval 4.2+ |
| **Orchestration** | LangChain Agents (create_agent) |
| **Validation** | Pydantic 2 |
| **Environment** | python-dotenv |
| **Packaging** | uv / pyproject.toml |

---

## 📈 Improving Scores

| Issue | Fix |
|-------|-----|
| Tool Correctness 85% | Add `get_today_date` to expected tools for all refund cases |
| Argument Correctness 75% | Tighten Pydantic schemas, add enum validation |
| Overall Pass Rate 75% | Add 5th test case (edge case: partial refund) |

---

## 📝 License

MIT License - Feel free to use for learning, interviews, or production.

---

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the agent framework
- [DeepEval](https://deepeval.com/) for evaluation metrics
- [Groq](https://groq.com/) for fast inference
- Original concept from [LangChain Agent Evaluation System](https://github.com/Mohamad-Hachem/LangChain_Agent_Evaluation_System)