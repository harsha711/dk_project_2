# 19 December 2025

**Platform**: Claude

**Prompt**: i am doing project 2. i need to know the structure of flow on how to approach it

**Prompt**: Which tech stack is preferred for this usecase given the time constraints and deliverables.

**Prompt**: How is Gradio better? How is it better than other tech stacks?

**Prompt**:

Build a Gradio app with 2 tabs for a dental AI platform:

TAB 1 - "Wisdom Tooth Detection":

-   User uploads dental X-ray image
-   Send image to GPT-4V or Gemini Vision API with prompt to identify wisdom tooth location
-   Draw bounding box on the image using PIL/OpenCV based on model response
-   Display original and annotated image side by side
-   Show detailed analysis with tooth descriptions and coordinates

TAB 2 - "Multi-Model Chatbot":

-   Query 3 AI models simultaneously in parallel:
    -   OpenAI GPT-4o
    -   Google Gemini 1.5 Flash
    -   Groq Llama3 70B
-   Display responses side by side for comparison
-   Include example questions
-   Use async execution for optimal performance


