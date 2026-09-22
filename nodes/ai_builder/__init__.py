# (c) Geekatplay Studio - Vladimir Chopine
# ComfyUI-Blender-Toolbox - AI Scene Builder
"""
AI Scene Builder engine.

Everything in this package is plain Python you can read end to end:
  config.py         defaults, paths, model recommendations, the sandbox warning text
  session.py        on-disk session state (JSON) that makes multi-pass building possible
  llm.py            thin HTTP clients for Ollama / Anthropic / OpenAI-compatible APIs
  rag.py            retrieval over docs/ai_builder/reference/*.md (BM25 + optional embeddings)
  prompts.py        every prompt the system sends to a model
  safety.py         static scan of generated code for obviously dangerous calls (NOT a sandbox)
  blender_runner.py headless (subprocess) and live (addon socket) execution of generated code
  validation.py     turns the Blender-side validation JSON into pass/fail + readable report
  agent.py          the plan -> generate -> execute -> validate -> retry loop
"""
