
# 🤖 GPA AI Search Bot

## 📌 Overview
The **GPA AI Search Bot** is an innovative and advanced search tool designed to pinpoint relevant data sections within uploaded DOCX files. It responds to user-provided keywords or entire sentences. Unlike traditional search functionalities, this tool employs cutting-edge AI models to comprehend and address user queries with unparalleled precision.

## ✨ Core Features
- **DOCX File Support**: Instantly render your `DOCX` files searchable.
- **Dual Search Mode**: Whether it's a specific keyword or a full-fledged sentence, we've got you covered.
- **Multiple Document Selection**: Why limit to one? Search across several uploaded documents simultaneously.

## 🔧 Technical Details
At its core, the AI Search Bot adopts an `embedding search technique`, utilizing the `text-embedding-ada-002` model from OpenAI.
 It further integrates the `gpt-3.5-turbo-16k` model for `AI-driven filtering` during its learning journey. Such an approach empowers users to converse in natural language, 
 breaking free from traditional tool constraints, like the `"find"` feature in standard browsers.

## 📂 Functional Pages
1. **Contents Page**
- Organize and manage with folders.
    - Ability to delete folders and remove uploaded documents.
    - Uploaded documents remain permanent in the database unless deleted.
    - Upload documents for the search bot to learn from.
    - View how the search bot interprets the uploaded documents.

2. **Chat Page**
   - Dive into a direct conversation with the search bot post-document upload.


## 🛠 Prerequisites
1. **OpenAI API Key**: Secure an OpenAI API key to operate this bot. Once in hand, feed this into the `.env` file.
2. **Virtual Environment**: It's recommended to set up and use a virtual environment to run the project. Follow the steps below to set it up.

### Setting Up Virtual Environment

#### macOS/Linux:

1. **Creation**:
   To create a virtual environment, use the built-in `venv` module:
   ```bash
   python3 -m venv myenv
   ```
   Replace `myenv` with your desired environment name.

2. **Activation**:
   Activate the virtual environment:
   ```bash
   source myenv/bin/activate
   ```



#### Windows:

1. **Creation**:
   Open the command prompt and run:
   ```bash
   python -m venv myenv
   ```
   Again, replace `myenv` with your desired environment name.

2. **Activation**:
   Activate the virtual environment:
   ```bash
   myenv\Scripts\activate
   ```

## pip version error
To avoid possible error caused due to pip version, please upgrade pip before installing 
    library requirements below <br>
    
   **pip upgrade command**:
   ```bash
   pip install --upgrade pip
   ```

### 📥 Installation (After Creation of Virtual Environment and Activated)
1. **Library Setup (Mac)**: 
    ```bash
    pip install -r requirements_mac.txt
    ```

2. **Library Setup (Windows)**: 
    ```bash
    pip install -r requirements_windows.txt
    ```

3. **OpenAI API Key Configuration**:
    Open the `.env` file and configure as follows:
    ```bash
    OPENAI_API_KEY=your_openai_api_key_here
    ```

## 🚀 Launching the App

To get started with the app, follow the instructions below:

### 1. Running the App

Execute the following command in your terminal:

```bash
python wsgi.py
```

### 2. Accessing the Web Interface

After executing the command, a link will be generated pointing to the website running locally. Copy this link to your browser to access the website.

### 3. Account Creation

When accessing the website, you'll land on a login page. If it's your first time, click on the **Sign Up** button to create a new account.


### Exiting Virtual Environment:

To deactivate the virtual environment and return to your system's Python:
```bash
deactivate
```



## 📂 Project Structure
- **Main Directory**: `GPA_Dict_AI_Search_Bot`
    - `requirements.txt`: Enumerates the essential libraries.
    - `wsgi.py`: The central code initiating the app.
    - `config.py`: Orchestrates the API key and other pivotal configurations.
- **Subdirectory**: `flask_APP` (Nestled within `GPA_Dict_AI_Search_Bot`)
    - `static`: Home to CSS styling, the app emblem, and other static commodities.
    - `templates`: Plays host to Jinja2 templates for the app's web interface, spanning from login to dashboard and more.
    - `__init__.py`: Spells out the `create_app` function leveraged in `wsgi.py`.
    - `assets.py`: Entrusted with asset compilation, be it JS or CSS.
    - `auth.py`: Oversees user registration and login dynamics.
    - `documents.py`: Encompasses primary codes for the search bot web endpoints.
    - `forms.py`: Dedicated to the user sign-up interface.
    - `GPA_DataBase.db`: Leans on SQLite, safeguarding both uploaded documents and user credentials.
    - `helper_function.py`: The backbone of the AI search bot's modus operandi.
    - `models.py`: Paves the path for the database's operations.
    - `routes.py`: Maps out the routes for authenticated pages.
