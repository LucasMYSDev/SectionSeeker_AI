## Guide of Setting up SectionSeekerAI in local machine

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

## 🚀 Launching the SectionSeekerAI

To get started with the app, follow the instructions below:

### 1. Running the SectionSeekerAI

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