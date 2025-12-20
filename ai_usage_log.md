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

TAB 2 - "Multi-Model Chatbot":

-   User enters a text query
-   Send query in parallel to 3 LLM APIs: OpenAI GPT-4o, Google Gemini, Groq Llama3
-   Display all 3 responses side by side in separate columns
-   Use asyncio or threading for parallel requests

Requirements:

-   Use Gradio for UI
-   Use python-dotenv for API keys (.env file)
-   Add error handling for API failures
-   Create requirements.txt
-   Keep code modular (separate files for api calls if needed)

Start with a working skeleton I can build upon.

**Prompt**: Create relevant documentation of features, architecture and readme

# 20 December 2025
