# Methods

Method implementations are independent from model and API adapters. The first
method is `agentic`, which performs a low-cost full-video pass and then requests
bounded high-detail frame intervals before producing the final answer.

The controller is separate from the regular inference runner but reuses its API
clients. Local Transformers backends are intentionally outside this method.
