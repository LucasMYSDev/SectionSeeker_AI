# 🤖 SectionSeekerAI

## 📌 Overview
The **SectionSeekerAI** is an innovative and advanced search tool designed to pinpoint relevant data sections within uploaded DOCX files. It responds to user-provided keywords or entire sentences. Unlike traditional search functionalities, this tool employs cutting-edge AI models to comprehend and address user queries with unparalleled precision.

## ✨ Core Features
- **DOCX File Support**: Instantly render your `DOCX` files searchable.
- **Dual Search Mode**: Whether it's a specific keyword or a full-fledged sentence, we've got you covered.
- **Multiple Document Selection**: Why limit to one? Search across several uploaded documents simultaneously.

## 🔧 Technical Details
SectionSeekerAI adopts an `embedding search technique` utilizing the `text-embedding-ada-002` model from OpenAI. It further integrates the `gpt-3.5-turbo-16k` model for `AI-driven filtering` during its learning journey. Such an approach empowers users to converse in natural language, breaking free from traditional tool constraints, like the `"find"` feature in standard browsers.

## 📂 SectionSeekerAI Website Pages
1. **Contents Page**
   - Organize and manage with folders.
   - Ability to delete folders and remove uploaded documents.
   - Uploaded documents remain permanent in the database unless deleted.
   - Upload documents for the search bot to learn from.
   - View how the search bot interprets the uploaded documents.
2. **Chat Page**
   - Dive into a direct conversation with the search bot post-document upload.

## ⚡ Quick Start
To test and try SectionSeekerAI:
- **Hosted Site**: [https://lucasmys.pythonanywhere.com/login](https://lucasmys.pythonanywhere.com/login)
- **Login Details**:
  - **Email**: sectionseekerai@gmail.com
  - **Password**: 123456
- **Sample Content**: 
  - The platform comes preloaded with 4 sample policies/handbooks. Each of these samples contains approximately 4,000 words, all generated by ChatGPT.
  - 📄 [Sample Docx](https://github.com/LucasMYS/SectionSeeker_AI/tree/main/sample_docx)

## 📖 How to Use

### **Step 1**: Quick Start
- Log in using the account details provided in the Quick Start section.

### **Step 2**: Accessing Contents
- Navigate to the contents page by clicking on "Contents" in the top navigation bar.

### **Step 3**: Managing Documents
1. Create a new folder.
2. Select a folder and upload a single document.
3. After uploading, view your processed documents by selecting "Documents" on the same Contents page.

### **Step 4**: Using Chat
1. Navigate to the chat page by clicking on "Chat" in the top navigation bar.
2. Select one or multiple documents to pose questions.
3. Click on "Ask" to submit your question. (Note: Pressing 'Enter' is not currently supported.)
4. SectionSeekerAI will return the 5 most relevant sections. Each result will include:
   - The section name (indicating the document it originates from and its respective section name).
   - A relatedness score (in percentage).
   - The actual text from the section identified by SectionSeekerAI.

https://github.com/LucasMYS/SectionSeeker_AI/assets/71440362/77576985-1c3e-44d9-94ec-31ed0e0452f6







## Limitations of SectionSeekerAI for Documents

Please note the following constraints when using SectionSeekerAI:

- **Development Stage**: As SectionSeekerAI is still in its early development stages, it currently supports only `.docx` file types.
  
- **Document Complexity**: The tool may not function optimally with `.docx` files containing images or intricate structures. It is best suited for plain text documents. This is ideal for business-related documents like company handbooks or business agreements, which usually have a straightforward format.
  
- **Styles Dependency**: SectionSeekerAI primarily uses heading styles to demarcate sections. If a `.docx` file has pre-existing styles that aren't correctly assigned, the tool might still process the document, but there might be unintended segmentations. For instance, what should be a single section might be split into two. As a result, when users search for specific sections based on queries, the returned results might not be entirely accurate.

To achieve the best results, ensure your document adheres to the guidelines and is compatible with the tool's current capabilities.

## 📂 Project Structure
- **Main Directory**: `SectionSeeker_AI`
    - Essential files: `requirements.txt`, `wsgi.py`, `config.py`.
- **Subdirectory**: `flask_APP` (Within `SectionSeeker_AI`)
    - Contains files and folders for styling, web interface templates, user registration, search bot web endpoints, and more.

## 🔗 Extend or Clone the Project
To expand upon or set up SectionSeekerAI on your local machine, check out the detailed installation guide on GitHub:
- [SectionSeekerAI Installation Guide](https://github.com/LucasMYS/SectionSeeker_AI/blob/main/SectionSeekerAI_Installation_Guide.md)
