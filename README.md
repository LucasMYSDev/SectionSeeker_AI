
# 🤖 SectionSeekerAI

## 📌 Overview
The **SectionSeekerAIt** is an innovative and advanced search tool designed to pinpoint relevant data sections within uploaded DOCX files. It responds to user-provided keywords or entire sentences. Unlike traditional search functionalities, this tool employs cutting-edge AI models to comprehend and address user queries with unparalleled precision.

## ✨ Core Features
- **DOCX File Support**: Instantly render your `DOCX` files searchable.
- **Dual Search Mode**: Whether it's a specific keyword or a full-fledged sentence, we've got you covered.
- **Multiple Document Selection**: Why limit to one? Search across several uploaded documents simultaneously.

## 🔧 Technical Details
At its core, the AI Search Bot adopts an `embedding search technique`, utilizing the `text-embedding-ada-002` model from OpenAI.
 It further integrates the `gpt-3.5-turbo-16k` model for `AI-driven filtering` during its learning journey. Such an approach empowers users to converse in natural language, 
 breaking free from traditional tool constraints, like the `"find"` feature in standard browsers.


## 📂 SectionSeekerAI Website Pages
1. **Contents Page**
- Organize and manage with folders.
    - Ability to delete folders and remove uploaded documents.
    - Uploaded documents remain permanent in the database unless deleted.
    - Upload documents for the search bot to learn from.
    - View how the search bot interprets the uploaded documents.

2. **Chat Page**
   - Dive into a direct conversation with the search bot post-document upload.


## How to Use SectionSeekerAI

SectionSeekerAI leverages the "heading styles" feature in Microsoft Word. This feature is found in the Styles Pane of the Microsoft Word interface. Specifically, SectionSeekerAI identifies sections that have titles styled with any variation containing the word "heading." It detects the beginning of a section based on this style and recognizes the end of that section when another styled title appears.

Users don't need to style every subsection; SectionSeekerAI employs GPT filtering to break down extensive sections into smaller subsections. Therefore, users can merely style prominent sections, such as "Section 1" or "Section 2." However, it's essential to be aware that exceptionally long sections might be truncated, leading to potential information loss. Typically, keeping each section limited to the content of one page is recommended.

## Limitations of SectionSeekerAI for Documents

Please note the following constraints when using SectionSeekerAI:

- **Development Stage**: As SectionSeekerAI is still in its early development stages, it currently supports only `.docx` file types.
  
- **Document Complexity**: The tool may not function optimally with `.docx` files containing images or intricate structures. It is best suited for plain text documents. This is ideal for business-related documents like company handbooks or business agreements, which usually have a straightforward format.
  
- **Styles Dependency**: SectionSeekerAI primarily uses heading styles to demarcate sections. If a `.docx` file has pre-existing styles that aren't correctly assigned, the tool might still process the document, but there might be unintended segmentations. For instance, what should be a single section might be split into two. As a result, when users search for specific sections based on queries, the returned results might not be entirely accurate.

To achieve the best results, ensure your document adheres to the guidelines and is compatible with the tool's current capabilities.


## 📂 Project Structure
- **Main Directory**: `SectionSeeker_AI`
    - `requirements.txt`: Enumerates the essential libraries.
    - `wsgi.py`: The central code initiating the app.
    - `config.py`: Orchestrates the API key and other pivotal configurations.
- **Subdirectory**: `flask_APP` (Nestled within `SectionSeeker_AI`)
    - `static`: Home to CSS styling, the app emblem, and other static commodities.
    - `templates`: Plays host to Jinja2 templates for the app's web interface, spanning from login to dashboard and more.
    - `__init__.py`: Spells out the `create_app` function leveraged in `wsgi.py`.
    - `assets.py`: Entrusted with asset compilation, be it JS or CSS.
    - `auth.py`: Oversees user registration and login dynamics.
    - `documents.py`: Encompasses primary codes for the search bot web endpoints.
    - `forms.py`: Dedicated to the user sign-up interface.
    - `SectonSeekerAI_DataBase.db`: Leans on SQLite, safeguarding both uploaded documents and user credentials.
    - `helper_function.py`: The backbone of the AI search bot's modus operandi.
    - `models.py`: Paves the path for the database's operations.
    - `routes.py`: Maps out the routes for authenticated pages.
