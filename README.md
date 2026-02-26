# Query-COI: Credibility of Information in Ranking Systems

A simple, focused framework implementing three core algorithms for analyzing credibility in ranking systems.

## 🎯 What This Does

This framework implements three algorithms from your research paper:

1. **Algorithm 1**: Credibility Detection - Finds credible ranking pairs
2. **Algorithm 2**: Base Query Construction (q_base) - Builds base query structure  
3. **Algorithm 4**: Maximally Informative Query Construction (q★) - Optimizes query structure

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Demo
```bash
python demo.py
```

### Run Experiments
```bash
# Basic experiment
python main.py basic --n-items 100

# Ablation study
python main.py ablation --n-items 100

# Scalability study
python main.py scalability

# Full experiment with all combinations
python main.py full --n-items 50
```

## 📊 Features

### Algorithms
- **Algorithm 1**: Credibility detection with intent-aware tie-breaking
- **Algorithm 2**: Base query construction with boundary merging
- **Algorithm 4**: Dynamic programming for optimal query structure

### Priors
- `uniform`: Uniform distribution
- `beta`: Beta distribution
- `exponential`: Exponential kernel

### Bias Types
- `constant`: Fixed bias
- `linear_high`: Higher values get higher bias
- `linear_low`: Lower values get higher bias  
- `gaussian`: Gaussian bias distribution

### Receiver Models
- `threshold`: Binary keep/drop decisions
- `quadratic`: Continuous actions

## 📁 Project Structure

```
Querying-COI/
├── src/
│   ├── algorithms.py      # Three core algorithms
│   └── experiments.py     # Experiment runner
├── main.py               # Command-line interface
├── demo.py               # Simple demo
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## 🔧 Usage Examples

### Basic Usage
```python
from src.algorithms import run_pipeline
from src.experiments import create_synthetic_dataset

# Create dataset
dataset = create_synthetic_dataset(n_items=100)

# Run pipeline
result = run_pipeline(
    dataset, "id", "quality",
    prior="uniform",
    bias_type="constant", 
    bias_degree=0.3,
    receiver_model="threshold",
    threshold=0.5
)

print(f"Credibility fraction: {result['credibility_fraction']:.3f}")
print(f"Time: {result['total_time']:.3f}s")
```

### Command Line
```bash
# Test with real dataset
python main.py basic --dataset data/real/census.csv --order-by education_num

# Run ablation study
python main.py ablation --n-items 200 --output ablation_results.csv

# Scalability test
python main.py scalability --output scalability_results.csv
```

## 📈 Output

The framework outputs:
- **Credibility fraction**: Fraction of credible ranking pairs
- **Execution time**: Time for each algorithm
- **Group counts**: Number of groups at each stage
- **Credible pairs**: Count of credible pairs found

Example output:
```
Items: 100
Initial groups: 15
Q-base groups: 8
Q-star groups: 5
Credible pairs: 42
Credibility fraction: 0.847
Total time: 1.234s
```

## 🎯 Research Applications

This framework is perfect for:
- **Credibility Analysis**: Understanding ranking credibility
- **Parameter Studies**: Testing different priors and biases
- **Scalability Analysis**: Performance across dataset sizes
- **Algorithm Comparison**: Comparing different approaches

## 🔧 Customization

### Add New Priors
Edit `compute_posteriors()` function in `algorithms.py`

### Add New Bias Types  
Edit `compute_bias()` function in `algorithms.py`

### Add New Receiver Models
Edit `system_response()` function in `algorithms.py`

## 📊 Example Results

Running the demo shows:
- Different configurations produce different credibility scores
- Algorithm execution times vary by parameters
- Credible pair counts depend on bias and prior settings

## 🎉 That's It!

This is a focused, working implementation of your three core algorithms. No unnecessary complexity - just the algorithms that matter for your research.