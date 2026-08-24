# Noryx AI model runtime

This directory contains the `llama.cpp` source imported from the supplied Qwen archive. It provides the local model runtime used by Noryx AI; model weights are intentionally not included in Git.

Build `llama.cpp` using its upstream instructions, then configure Noryx AI to connect to the local OpenAI-compatible model server. Keep downloaded weights and generated build output outside Git.
