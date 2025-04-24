# mixture_of_search_agents
Combining different AI models to improve search and reasoning

mcts_llm.py: the skeleton version that other versions build on
mcts_api.py: a version using agent integregation from four different models
mcts_local.py: a version using only models running locally, using ollama
mcts_reward.py: a fully implemented version that contains a reward function based on which answers get aggregated most, a separate llm to aggregate the answers from different sub-agents, and another model to determine the descrepency between the MOSA answer and the labeled answer.

Inspired by the MOSA paper: https://arxiv.org/pdf/2502.18873 
