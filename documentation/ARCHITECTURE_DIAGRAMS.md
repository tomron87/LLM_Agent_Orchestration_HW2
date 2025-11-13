# Architecture Diagrams

**System:** LSTM Frequency Extraction System
**Version:** 1.0
**Date:** November 13, 2025

This document contains formal C4 and UML diagrams following industry standards.

**Rendering:** These diagrams use Mermaid and PlantUML syntax. View on GitHub or use:
- [Mermaid Live Editor](https://mermaid.live/)
- [PlantUML Online](http://www.plantuml.com/plantuml/uml/)

---

## Table of Contents

1. [C4 Model Diagrams](#c4-model-diagrams)
2. [UML Class Diagrams](#uml-class-diagrams)
3. [UML Sequence Diagrams](#uml-sequence-diagrams)
4. [Component Diagrams](#component-diagrams)
5. [Deployment Diagram](#deployment-diagram)

---

## C4 Model Diagrams

### Level 1: System Context Diagram

```mermaid
C4Context
    title System Context Diagram - LSTM Frequency Extraction System

    Person(researcher, "Data Scientist", "Trains models to extract<br/>frequencies from signals")

    System(lstm_system, "LSTM Frequency<br/>Extraction System", "Trains LSTM models to extract<br/>pure frequency components from<br/>noisy mixed signals")

    System_Ext(pytorch, "PyTorch", "Deep learning framework")
    System_Ext(filesystem, "File System", "Stores models, figures,<br/>and logs")

    Rel(researcher, lstm_system, "Configures and runs", "CLI/Config Files")
    Rel(lstm_system, pytorch, "Uses for", "Model training")
    Rel(lstm_system, filesystem, "Reads/Writes", "Models, Data, Logs")
```

### Level 2: Container Diagram

```mermaid
C4Container
    title Container Diagram - LSTM Frequency Extraction System

    Person(researcher, "Data Scientist")

    Container_Boundary(system, "LSTM Frequency Extraction System") {
        Container(cli, "CLI Application", "Python", "Command-line interface<br/>for training and evaluation")
        Container(data_pipeline, "Data Pipeline", "Python/NumPy", "Generates synthetic<br/>signal data")
        Container(model_engine, "Model Engine", "PyTorch", "LSTM model architectures<br/>and training logic")
        Container(eval_engine, "Evaluation Engine", "Python/Matplotlib", "Computes metrics and<br/>generates visualizations")
        ContainerDb(storage, "File Storage", "YAML/PKL/PNG", "Configuration, models,<br/>figures, logs")
    }

    System_Ext(pytorch, "PyTorch Framework")

    Rel(researcher, cli, "Executes commands", "Terminal")
    Rel(cli, data_pipeline, "Requests data", "Function calls")
    Rel(cli, model_engine, "Trains model", "Function calls")
    Rel(cli, eval_engine, "Evaluates", "Function calls")
    Rel(data_pipeline, storage, "Reads config", "YAML")
    Rel(model_engine, pytorch, "Uses", "API calls")
    Rel(model_engine, storage, "Saves checkpoints", "PKL files")
    Rel(eval_engine, storage, "Saves figures", "PNG files")
```

### Level 3: Component Diagram

```mermaid
C4Component
    title Component Diagram - Model Engine Container

    Container_Boundary(model_engine, "Model Engine") {
        Component(stateful_lstm, "StatefulLSTM", "PyTorch Module", "L=1 model with<br/>explicit state management")
        Component(sequence_lstm, "SequenceLSTM", "PyTorch Module", "L>1 model for<br/>sequence processing")
        Component(conditioned_lstm, "ConditionedLSTM", "PyTorch Module", "FiLM-conditioned<br/>frequency extraction")
        Component(trainer, "Trainer", "Python Class", "Training loop,<br/>early stopping,<br/>checkpointing")
        Component(config, "TrainingConfig", "Dataclass", "Hyperparameters and<br/>training settings")
    }

    Container_Ext(data_loader, "Data Loader", "PyTorch DataLoader")
    Container_Ext(optimizer, "Optimizer", "PyTorch Optim")

    Rel(trainer, stateful_lstm, "Trains")
    Rel(trainer, sequence_lstm, "Trains")
    Rel(trainer, conditioned_lstm, "Trains")
    Rel(trainer, config, "Reads settings")
    Rel(trainer, data_loader, "Fetches batches")
    Rel(trainer, optimizer, "Updates parameters")
```

---

## UML Class Diagrams

### Core Model Architecture

```mermaid
classDiagram
    class Module {
        <<PyTorch>>
        +forward()
        +parameters()
    }

    class StatefulLSTM {
        -input_size: int
        -hidden_size: int
        -num_layers: int
        -lstm: LSTM
        -fc: Linear
        -hidden_state: Tensor
        -cell_state: Tensor
        +__init__(input_size, hidden_size, num_layers, dropout)
        +forward(x, reset_state) Tensor
        +reset_state()
        +get_state() Tuple
        +set_state(state)
        +count_parameters() int
    }

    class SequenceLSTM {
        -input_size: int
        -hidden_size: int
        -num_layers: int
        -sequence_length: int
        -lstm: LSTM
        -fc: Linear
        +__init__(input_size, hidden_size, num_layers, sequence_length, dropout)
        +forward(x) Tensor
        +count_parameters() int
    }

    class ConditionedLSTM {
        -input_size: int
        -hidden_size: int
        -num_layers: int
        -condition_dim: int
        -lstm: LSTM
        -film_gamma: Linear
        -film_beta: Linear
        -fc: Linear
        +__init__(input_size, hidden_size, num_layers, condition_dim)
        +forward(x, condition) Tensor
        +apply_film(features, condition) Tensor
    }

    Module <|-- StatefulLSTM
    Module <|-- SequenceLSTM
    Module <|-- ConditionedLSTM
```

### Data Pipeline Classes

```mermaid
classDiagram
    class SignalGenerator {
        -frequencies: List[float]
        -fs: int
        -duration: float
        -seed: int
        -phase_scale: float
        -num_samples: int
        -rng: RandomState
        +__init__(frequencies, fs, duration, seed, phase_scale)
        +generate_time_array() ndarray
        +generate_noisy_component(freq_idx, t) ndarray
        +generate_mixed_signal() Tuple
        +generate_ground_truth(freq_idx, t) ndarray
        +generate_complete_dataset() Dict
        +save_dataset(filepath)
        +load_dataset(filepath) Dict
    }

    class Dataset {
        <<PyTorch>>
        +__len__() int
        +__getitem__(idx) Tuple
    }

    class FrequencyExtractionDataset {
        -signal_data: ndarray
        -target_data: ndarray
        -time_data: ndarray
        -frequencies: List
        +__init__(signal_data, target_data, time_data, frequencies)
        +__len__() int
        +__getitem__(idx) Tuple
        +get_frequency_slice(freq_idx) Tuple
    }

    class SequenceDataset {
        -signal_data: ndarray
        -target_data: ndarray
        -sequence_length: int
        -stride: int
        +__init__(signal_data, target_data, sequence_length, stride)
        +__len__() int
        +__getitem__(idx) Tuple
    }

    Dataset <|-- FrequencyExtractionDataset
    Dataset <|-- SequenceDataset
    SignalGenerator --> FrequencyExtractionDataset : creates data for
```

### Training Components

```mermaid
classDiagram
    class TrainingConfig {
        <<dataclass>>
        +model_type: str
        +hidden_size: int
        +num_layers: int
        +sequence_length: int
        +dropout: float
        +batch_size: int
        +learning_rate: float
        +num_epochs: int
        +val_split: float
        +patience: int
        +device: str
        +to_dict() Dict
        +from_dict(config_dict) TrainingConfig
    }

    class EarlyStopping {
        -patience: int
        -min_delta: float
        -counter: int
        -best_loss: float
        -early_stop: bool
        +__init__(patience, min_delta, verbose)
        +__call__(val_loss) bool
        +reset()
    }

    class Trainer {
        -model: Module
        -config: TrainingConfig
        -train_loader: DataLoader
        -val_loader: DataLoader
        -optimizer: Optimizer
        -criterion: Loss
        -early_stopping: EarlyStopping
        -history: Dict
        +__init__(model, config, train_loader, val_loader)
        +train() Dict
        +train_epoch(epoch) Dict
        +validate() float
        +save_checkpoint(filepath)
        +load_checkpoint(filepath)
        -_train_step_stateful(batch) float
        -_train_step_sequence(batch) float
    }

    Trainer --> TrainingConfig : uses
    Trainer --> EarlyStopping : uses
    Trainer --> Module : trains
```

### Evaluation System

```mermaid
classDiagram
    class Evaluator {
        -model: Module
        -test_loader: DataLoader
        -device: str
        +__init__(model, test_loader, device)
        +evaluate() Dict
        +compute_per_frequency_metrics() Dict
        -_evaluate_stateful() Dict
        -_evaluate_sequence() Dict
    }

    class Metrics {
        <<utility>>
        +compute_mse(predictions, targets) float
        +compute_correlation(predictions, targets) float
        +check_generalization(train_mse, test_mse, threshold) bool
    }

    class Visualizer {
        -output_dir: Path
        -dpi: int
        +__init__(output_dir, dpi)
        +plot_training_curves(history, output_path)
        +plot_single_frequency(time, target, prediction, freq, output_path)
        +plot_all_frequencies(results, output_path)
        +plot_fft_analysis(signal, targets, predictions, output_path)
        +plot_error_distribution(errors, output_path)
    }

    Evaluator --> Metrics : uses
    Evaluator ..> Visualizer : creates plots with
```

---

## UML Sequence Diagrams

### Training Sequence (L=1 Stateful)

```mermaid
sequenceDiagram
    actor User
    participant CLI as Main CLI
    participant Generator as SignalGenerator
    participant Dataset as FrequencyExtractionDataset
    participant Trainer as Trainer
    participant Model as StatefulLSTM
    participant Optimizer as Optimizer

    User->>CLI: python main.py --model stateful
    CLI->>Generator: generate_complete_dataset()
    Generator-->>CLI: signal_data, target_data

    CLI->>Dataset: FrequencyExtractionDataset(data)
    CLI->>Model: StatefulLSTM(hidden_size=64)
    CLI->>Trainer: Trainer(model, config, dataloader)

    loop For each epoch
        Trainer->>Trainer: train_epoch(epoch)
        loop For each batch
            Trainer->>Dataset: Get batch
            Dataset-->>Trainer: inputs, targets, metadata

            alt Frequency changed or sample_idx == 0
                Trainer->>Model: forward(inputs, reset_state=True)
            else
                Trainer->>Model: forward(inputs, reset_state=False)
            end

            Model-->>Trainer: predictions
            Trainer->>Trainer: compute_loss(predictions, targets)
            Trainer->>Optimizer: zero_grad()
            Trainer->>Trainer: loss.backward()
            Trainer->>Trainer: clip_gradients()
            Trainer->>Optimizer: step()
            Trainer->>Model: detach states (prevent graph buildup)
        end

        Trainer->>Trainer: validate()

        alt Validation improved
            Trainer->>Trainer: save_checkpoint()
        end

        alt Early stopping triggered
            Trainer-->>CLI: Training stopped early
        end
    end

    Trainer-->>CLI: training_history
    CLI->>User: Display results
```

### Evaluation Sequence

```mermaid
sequenceDiagram
    actor User
    participant CLI as Main CLI
    participant Evaluator as Evaluator
    participant Model as LSTM Model
    participant Metrics as Metrics
    participant Viz as Visualizer

    User->>CLI: Evaluate trained model
    CLI->>CLI: load_checkpoint()
    CLI->>Evaluator: Evaluator(model, test_loader)

    Evaluator->>Model: model.eval()

    loop For each test batch
        Evaluator->>Model: forward(inputs)
        Model-->>Evaluator: predictions
        Evaluator->>Evaluator: accumulate results
    end

    Evaluator->>Metrics: compute_mse(pred, target)
    Metrics-->>Evaluator: mse_value

    Evaluator->>Metrics: compute_correlation(pred, target)
    Metrics-->>Evaluator: correlation_value

    loop For each frequency
        Evaluator->>Metrics: compute_per_frequency_metrics()
        Metrics-->>Evaluator: freq_metrics
    end

    Evaluator-->>CLI: evaluation_results

    CLI->>Viz: plot_training_curves()
    CLI->>Viz: plot_single_frequency()
    CLI->>Viz: plot_all_frequencies()
    CLI->>Viz: plot_fft_analysis()

    Viz-->>CLI: Figures saved
    CLI->>User: Display summary and figures
```

### Configuration Loading Sequence

```mermaid
sequenceDiagram
    participant User
    participant CLI as Main CLI
    participant ConfigLoader as ConfigLoader
    participant YAML as config.yaml
    participant ENV as .env file
    participant Config as TrainingConfig

    User->>CLI: python main.py
    CLI->>ConfigLoader: ConfigLoader('config.yaml')

    ConfigLoader->>ENV: load_dotenv()
    ENV-->>ConfigLoader: Environment variables

    ConfigLoader->>YAML: open and parse
    YAML-->>ConfigLoader: config_dict

    loop For each config key
        ConfigLoader->>ConfigLoader: Check env variable override
        alt Env var exists
            ConfigLoader->>ConfigLoader: Use env var value
        else
            ConfigLoader->>ConfigLoader: Use YAML value
        end
    end

    ConfigLoader-->>CLI: merged_config
    CLI->>Config: TrainingConfig.from_dict(config)
    Config-->>CLI: config_object

    CLI->>CLI: Initialize components with config
```

---

## Component Diagrams

### System Component Architecture

```mermaid
graph TB
    subgraph "User Interface Layer"
        CLI[CLI Application<br/>main.py]
    end

    subgraph "Configuration Layer"
        ConfigLoader[Config Loader<br/>config_loader.py]
        YAML[config.yaml]
        ENV[.env]
    end

    subgraph "Data Layer"
        SigGen[Signal Generator<br/>signal_generator.py]
        Dataset[PyTorch Datasets<br/>dataset.py]
        DataLoader[Data Loaders<br/>data_loader.py]
    end

    subgraph "Model Layer"
        Stateful[Stateful LSTM<br/>lstm_stateful.py]
        Sequence[Sequence LSTM<br/>lstm_sequence.py]
        Conditioned[Conditioned LSTM<br/>lstm_conditioned.py]
    end

    subgraph "Training Layer"
        TrainConfig[Training Config<br/>config.py]
        Trainer[Trainer<br/>trainer.py]
        EarlyStopping[Early Stopping<br/>callbacks]
    end

    subgraph "Evaluation Layer"
        Evaluator[Evaluator<br/>evaluator.py]
        Metrics[Metrics<br/>metrics.py]
        Visualizer[Visualizer<br/>visualization.py]
    end

    subgraph "Utility Layer"
        Logger[Logger<br/>logger.py]
        CostAnalyzer[Cost Analyzer<br/>cost_analysis.py]
    end

    subgraph "Storage Layer"
        FileSystem[(File System<br/>Models/Figures/Logs)]
    end

    CLI --> ConfigLoader
    ConfigLoader --> YAML
    ConfigLoader --> ENV

    CLI --> SigGen
    SigGen --> Dataset
    Dataset --> DataLoader

    CLI --> Trainer
    Trainer --> TrainConfig
    Trainer --> Stateful
    Trainer --> Sequence
    Trainer --> Conditioned
    Trainer --> EarlyStopping
    Trainer --> DataLoader

    CLI --> Evaluator
    Evaluator --> Metrics
    Evaluator --> Visualizer

    Trainer --> Logger
    Trainer --> CostAnalyzer
    Evaluator --> Logger

    Trainer --> FileSystem
    Evaluator --> FileSystem
    Visualizer --> FileSystem
    Logger --> FileSystem
```

---

## Deployment Diagram

### Local Development/Training Deployment

```mermaid
graph TB
    subgraph "Development Machine"
        subgraph "Python Environment"
            Main[main.py]
            SrcCode[Source Code<br/>src/]
            Tests[Test Suite<br/>tests/]
        end

        subgraph "Dependencies"
            PyTorch[PyTorch]
            NumPy[NumPy]
            Matplotlib[Matplotlib]
        end

        subgraph "Local Storage"
            Config[config.yaml<br/>.env]
            Data[Data Files<br/>*.pkl]
            Models[Model Checkpoints<br/>*.pth]
            Figures[Figures<br/>*.png]
            Logs[Log Files<br/>*.log]
        end

        subgraph "Compute Resources"
            CPU[CPU<br/>Training]
            GPU[GPU/MPS<br/>Accelerated Training]
        end
    end

    Main --> SrcCode
    Main --> Config
    SrcCode --> PyTorch
    SrcCode --> NumPy
    SrcCode --> Matplotlib

    Main --> CPU
    Main --> GPU

    Main --> Data
    Main --> Models
    Main --> Figures
    Main --> Logs
```

---

## Data Flow Diagram

```mermaid
flowchart LR
    A[Configuration<br/>YAML + ENV] --> B[Signal Generator]
    B --> C[Synthetic Data<br/>S t + Targets]
    C --> D[PyTorch Dataset]
    D --> E[DataLoader<br/>Batching]
    E --> F{Model Type}

    F -->|L=1| G[Stateful LSTM]
    F -->|L>1| H[Sequence LSTM]

    G --> I[Training Loop]
    H --> I

    I --> J{Validation}
    J -->|Improved| K[Save Checkpoint]
    J -->|No Improvement| L{Early Stop?}
    L -->|Yes| M[End Training]
    L -->|No| I

    K --> N[Best Model]
    M --> O[Final Evaluation]
    N --> O

    O --> P[Compute Metrics]
    O --> Q[Generate Plots]
    P --> R[Results Summary]
    Q --> R
```

---

## Legend

### C4 Model Elements

- **Person**: External user or actor
- **System**: Software system boundary
- **Container**: Deployable unit (application, service, database)
- **Component**: Grouping of related functionality

### UML Elements

- **Class**: Object-oriented class definition
- **Interface**: Contract specification
- **Association**: Relationship between classes
- **Inheritance**: Parent-child relationship
- **Dependency**: Uses relationship

### Diagram Conventions

- **Solid Line**: Direct dependency
- **Dashed Line**: Indirect dependency
- **Arrow**: Direction of dependency
- **Diamond**: Composition/Aggregation

---

**Note**: These diagrams can be rendered using:
- GitHub (automatic Mermaid rendering)
- Mermaid Live Editor: https://mermaid.live/
- VS Code with Mermaid extension
- PlantUML Online: http://www.plantuml.com/plantuml/uml/

**Last Updated:** November 13, 2025
