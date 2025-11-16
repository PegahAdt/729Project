

##  Configs/

Contains YAML configuration files that describe:

### **Agent settings**
- network architecture  
- optimizer settings  
- discount factor γ  
- batch size  
- learning rates  
- entropy regularization  

### **Environment settings**
- environment name 
- time limit
- rendering flag (`visualize: true/false`)

---

##  envs/

Environment wrappers and builders.

### **env_builder.py**
Builds environments from YAML configuration.  

### **gym_env.py**
**You might need to change this. This is just a place holder for our env.**

Unified Gymnasium environment wrapper:
- handles `reset()` and `step()`  
- enforces a custom time limit  
- manages truncated vs terminated flags  
- exposes observation/action spaces for the agent  

---

##  learning/

Core learning logic (distributions, models, normalization, optimizers, training loops).

### **distribution_gaussian_diag.py**
Diagonal Gaussian policy for **continuous action spaces**.
Supports:
- fixed σ  
- learnable σ  
- state-dependent σ (neural network)

Provides:
- action sampling  
- log-probabilities  
- entropy  
- KL divergence  

### **distribution_categorical.py**
Categorical (softmax) distribution for **discrete action spaces**.  
Provides:
- sampling  
- log-probabilities  
- entropy  

> Use whichever matches the environment. Both are included for flexibility.

---

### **agent_builder.py**
Loads an agent YAML file, verifies that the agent type is PG, and constructs a `PGAgent`.

---

### **base_agent.py**
Base class for all RL agents. Provides:
- environment management  
- sample/iteration tracking  
- normalizers  
- experience buffer setup  
- saving/loading  
- high-level training & testing loops  

Key methods:
- `train_model()` – collect rollouts, update actor/critic, log progress  
- `test_model()` – run evaluation episodes  

---

### **pg_model.py**
Defines the actor–critic model.

- Builds actor and critic networks via `net_builder.build_net()`
- `eval_actor()` → returns an action distribution  
- `eval_critic()` → returns a scalar value estimate  

Architectures are controlled fully via YAML.

---

### **normalizer.py**
Tracks running mean and variance for observations/actions.  
Provides:
- `normalize(x)`  
- `unnormalize(x)`  


---

### **return_tracker.py**
Tracks episode returns and lengths.  
Used for logging and reporting progress.

---

### **mp_optimizer.py**
Wraps a PyTorch optimizer and adds **multi-process gradient synchronization**.  
Provides:
- backward pass  
- gradient averaging  
- optimizer step  

---

##  nets/

Contains network builders.

Example:  
- `fc_2layers_128units.py`

> **To add a costumized architecture**, create a file here and reference it in `configs/pg_agent_config.yaml` under `actor_net` and `critic_net`.

---

##  pg/

Contains the **PGAgent** implementation.

### **pg_agent.py**
Implements policy-gradient learning:
- computes reward-to-go  
- computes advantages  
- builds actor & critic optimizers  
- runs multiple updates per iteration  
- samples actions during training  
- uses greedy actions during testing  

THese functions must implement similar to the assignment:
- `_calc_return()`  
- `_calc_adv()`  
- `_critic_loss()`  
- `_actor_loss()`  


---

##  tools/

Post-processing, plotting, and analysis utilities.

### **plot_log/plot_log.py**
Loads log files and produces learning-curve plots using Matplotlib.  
Calls helper functions from `tools/util/plot_util.py`.

### **tools/util/plot_util.py**
Utility functions for smoothing and plotting learning curves.

---

##  util/

General-purpose utilities:
- logging  
- tensor helpers  
- multiprocessing helpers  
- seeding  
- device selection  
---

## Running Examples

---
**Training:**
```bash
python run.py \
  --mode train \
  --env_config configs/pg_project_config.yaml \
  --agent_config configs/pg_project_config.yaml \
  --device cpu \
  --max_samples 200000 \
  --log_file output/project_train.log \

  --out_model_file output/pg_project_model.pt
```
**Testing**
```bash
python run.py \
  --mode test \
  --env_config configs/pg_project_config.yaml \
  --agent_config configs/pg_project_config.yaml \
  --device cpu \
  --model_file output/pg_project_model.pt \
  --test_episodes 20

```


