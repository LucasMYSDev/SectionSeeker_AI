function updateFolderSelect() {
    // Disable the folder select and upload button while loading
    const folderSelect = document.getElementById('folder-select');
    const uploadButton = document.querySelector('#upload-form button');
    if (folderSelect && uploadButton) { // Check if the elements exist
        folderSelect.disabled = true;
        uploadButton.disabled = true;

        fetch('/folders')
            .then(response => response.json())
            .then(folders => {
                // Clear all existing options
                if (folders.length > 0) {
                    document.getElementById('folder-id').value = folders[0].id;
                }
                while (folderSelect.firstChild) {
                    folderSelect.firstChild.remove();
                }
                // Add an option for each folder
                folders.forEach(folder => {
                    const option = document.createElement('option');
                    option.value = folder.id;
                    option.textContent = folder.name;
                    folderSelect.appendChild(option);
                });

                // Enable the folder select and upload button after loading
                folderSelect.disabled = false;
                uploadButton.disabled = false;
            })
            .then(fetchFolders); // Fetch folders and populate the folder list table
    }
}


function updateDocumentSelect() {
    const documentSelect = document.getElementById('document-select');
    if (documentSelect) {
        documentSelect.innerHTML = ''; // Clear existing options

        fetch('/documents') // You may need to adjust this endpoint to match your API
            .then(response => response.json())
            .then(data => {
                const folders = data.folders; // Adjust this based on the structure of your response

                // Loop through each folder
                folders.forEach(folder => {
                    // Create a disabled option for the folder name
                    const folderOption = document.createElement('option');
                    folderOption.value = folder.folder_id;
                    folderOption.textContent = folder.folder_name;
                    folderOption.disabled = true; // Disable the option so it can't be selected
                    documentSelect.appendChild(folderOption);

                    // Loop through each document in the folder
                    folder.documents.forEach(doc => { 
                        const option = document.createElement('option');
                        option.value = doc.id;
                        option.textContent = doc.filename; 
                        documentSelect.appendChild(option);
                    });
                });
            })
            .catch(error => console.error('Error fetching documents:', error));
    }
}

// function deleteFolder(folderId) {
//     fetch(`/delete-folder/${folderId}`, {
//         method: 'DELETE',
//     })
//     .then((response) => response.json())
//     .then((data) => {
//         if (data.success) {
//             updateFolderSelect();
//             fetchFolders();
//             // fetchDocuments();
//             fetchDocumentsForContents
//             updateDocumentSelect();
//         } else {
//             console.error('Error deleting folder:', data.message);
//         }
//     })
//     .catch((error) => {
//         console.error('Error:', error);
//     });
// }

async function deleteFolder(folderId) {
    try {
        const response = await fetch(`/delete-folder/${folderId}`, {
            method: 'DELETE',
        });
        const data = await response.json();

        if (data.success) {
            await updateFolderSelect();
            await fetchFolders();
            await fetchDocumentsForContents();
            await updateDocumentSelect();
        } else {
            console.error('Error deleting folder:', data.message);
        }
    } catch (error) {
        console.error('Error:', error);
    }
}


function fetchFolders() {
    fetch('/folders')
        .then(response => response.json())
        .then(folders => {
            const table = document.querySelector('.table');

            // Clear the table
            table.innerHTML = `
                <tr>
                    <th>Folder Name</th>
                    <th>Delete</th>
                </tr>`;

            folders.forEach(folder => {
                let row = table.insertRow();
                let cell = row.insertCell();
                cell.textContent = folder.name;

                cell = row.insertCell();
                let button = document.createElement('button');
                button.textContent = 'Delete Folder';
                button.className = 'delete-folder-button'; // Add this line
                button.dataset.folderId = folder.id; // Add this line
                cell.appendChild(button);
            });
        });
}


function fetchDocumentsForContents() {
    fetch('/documents')
        .then(response => response.json())
        .then(data => {
            const folders = data.folders;
            const documentsListDiv = document.getElementById('documents-list');

            if (documentsListDiv){
                documentsListDiv.innerHTML = ''; // Clear existing content

                folders.forEach(folder => {
                    // Create a container for the folder
                    const folderDiv = document.createElement('div');
                    folderDiv.classList.add('folder-box');

                    // Add folder name as a title
                    const folderTitle = document.createElement('h3');
                    folderTitle.textContent = folder.folder_name;
                    folderDiv.appendChild(folderTitle);

                    // Create table for documents
                    const table = document.createElement('table');
                    table.classList.add('table', 'table-bordered'); // Add table-bordered class for styling
                    table.style.width = '100%'; // Adjust the width as needed
                    table.innerHTML = `
                        <tr>
                            <th>Document Name</th>
                            <th>View Processed Document</th>
                            <th>Delete</th>
                        </tr>`;

                    folder.documents.forEach(doc => {
                        let row = table.insertRow();
                        row.id = `document-${doc.id}`;

                        // Add document name
                        let cell = row.insertCell();
                        cell.textContent = doc.filename;

                        // View Processed Document button
                        cell = row.insertCell();
                        let button = document.createElement('button');
                        button.textContent = 'View Processed Document';
                        button.addEventListener('click', () => {
                            window.location.href = '/view-learnt-documents/' + doc.id;
                        });
                        cell.appendChild(button);

                        // Delete Document button
                        cell = row.insertCell();
                        button = document.createElement('button');
                        button.textContent = 'Delete';
                        button.addEventListener('click', () => {
                            fetch(`/delete-document/${doc.id}`, {
                                method: 'DELETE',
                            })
                            .then((response) => response.json())
                            .then((data) => {
                                if (data.success) {
                                    updateFolderSelect();
                                    fetchFolders();
                                    fetchDocumentsForDashboard();
                                } else {
                                    console.error('Error deleting document:', data.message);
                                }
                            })
                            .catch((error) => {
                                console.error('Error:', error);
                            });
                        });
                        cell.appendChild(button);
                    });

                    // Append table to folder div
                    folderDiv.appendChild(table);

                    // Append folder div to documents list
                    documentsListDiv.appendChild(folderDiv);
                });
            }
        });
}

const selectedDocumentIds = new Set();

function fetchDocuments_AskPage() {
    fetch('/documents')
        .then(response => response.json())
        .then(data => {
            const folders = data.folders;
            const documentsContainer = document.querySelector('.documents-container');
            documentsContainer.innerHTML = ''; // Clear existing content

            folders.forEach(folder => {
                // Create a separator for the folder name
                const folderSeparator = document.createElement('div');
                folderSeparator.className = 'folder-separator';
                folderSeparator.textContent = folder.folder_name;
                documentsContainer.appendChild(folderSeparator);

                folder.documents.forEach(doc => {
                    const docDiv = document.createElement('div');
                    docDiv.className = 'document-item';
                    docDiv.textContent = doc.filename;
                    docDiv.dataset.docId = doc.id;
                    documentsContainer.appendChild(docDiv);
                });
            });

            // This code populates the documentsContainer with clickable divs representing each document.
            // When a div is clicked, it toggles its selected state and updates the selectedDocumentIds set.
            documentsContainer.addEventListener('click', (event) => {
                if (event.target.classList.contains('document-item')) {
                    const docId = event.target.dataset.docId;
                    if (selectedDocumentIds.has(docId)) {
                        selectedDocumentIds.delete(docId);
                        event.target.classList.remove('selected');
                    } else {
                        selectedDocumentIds.add(docId);
                        event.target.classList.add('selected');
                    }
                    
                    // Update the selected document count after each click
                    updateSelectedCount();
                }
            });
        });
}


function updateSelectedCount() {
    const selectedCountEl = document.querySelector('.selected-count');
    if (selectedCountEl) { // Only update if the element exists
        selectedCountEl.textContent = `${selectedDocumentIds.size} documents selected`;
    }
}



window.onload = function() {
        // Attach the event listener to the table for event delegation
        const table = document.querySelector('.table');
        if (table){
            table.addEventListener('click', (event) => {
                let targetElement = event.target;
                while (targetElement !== null && !targetElement.classList.contains('delete-folder-button')) {
                    targetElement = targetElement.parentElement;
                }
                
                if (targetElement !== null && targetElement.classList.contains('delete-folder-button')) {
                    const folderId = targetElement.dataset.folderId;
                    deleteFolder(folderId);
                }
            });
        }
    
    if (document.getElementById('dashboard-page')) { // Dashboard page
        updateFolderSelect();
        updateDocumentSelect();
    }
    else if (document.getElementById('ask-page')) { // Chat page
        // Code specific to the chat page
        updateFolderSelect();
        fetchDocuments_AskPage();
        updateDocumentSelect();
    } else if (document.getElementById('contents-page')) { // Dashboard page
        // Code specific to the dashboard page
        updateFolderSelect();
        fetchDocumentsForContents();
        updateDocumentSelect();
    }



    const folderForm = document.getElementById('folder-form');
    const folderSelect = document.getElementById('folder-select');

    if (folderSelect) { // Check if the folder-select element exists
        folderSelect.addEventListener('change', (event) => {
            document.getElementById('folder-id').value = event.target.value;
        });
    }


    if (folderForm) {
        folderForm.addEventListener('submit', function(event) {
            event.preventDefault();
            const folderName = document.getElementById('folder-name').value;
            fetch('/create-folder', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 'name': folderName }),
            })
            .then((response) => {
                if (!response.ok) {
                    throw new Error('Failed to create folder');
                }
                return response.json();
            })
            .then((folder) => {
                updateFolderSelect();
                fetchDocumentsForContents()
                updateDocumentSelect();
                const folderSelect = document.getElementById('folder-select');
                const option = document.createElement('option');
                option.value = folder.folder_id;
                option.textContent = folder.folder_name;
                folderSelect.appendChild(option);
                document.getElementById('folder-id').value = folder.folder_id;
                const table = document.querySelector('.table');
                let row = table.insertRow();
                let cell = row.insertCell();
                cell.textContent = folder.folder_name;
                cell = row.insertCell();
                let button = document.createElement('button');
                button.textContent = 'Delete Folder';
                button.addEventListener('click', () => {
                    fetch(`/delete-folder/${folder.folder_id}`, {
                        method: 'DELETE',
                    })
                    .then((response) => response.json())
                    .then((data) => {
                        if (data.success) {
                            updateFolderSelect();
                            fetchFolders();
                            fetchDocumentsForContents()
                        } else {
                            console.error('Error deleting folder:', data.message);
                        }
                    })
                    .catch((error) => {
                        console.error('Error:', error);
                    });
                });
                cell.appendChild(button);
            })
            .catch((error) => {
                console.error('Error:', error);
                alert('Folder name already exists!');
            });
        });
    }
    

    if (folderSelect) {
        folderSelect.addEventListener('change', (event) => {
            document.getElementById('folder-id').value = event.target.value;
        });
    }

        
    document.addEventListener('click', (event) => {
        if (event.target.classList.contains('delete-button')) {
            const documentId = event.target.dataset.id;
        
            fetch(`/delete-document/${documentId}`, {
            method: 'DELETE',
            })
            .then((response) => response.json())
            .then((data) => {
                if (data.success) {
                    updateFolderSelect();
                    fetchFolders(); // Make sure this is defined, or replace with the appropriate function
                    fetchDocumentsForContents()
                // Remove the deleted document from the UI
                document.getElementById(`document-${documentId}`).remove();
                } else {
                console.error('Error deleting document:', data.message);
                }
            })
            .catch((error) => {
                console.error('Error:', error);
            });
        }
    });


    // const askButton = document.getElementById('ask-button');
    // if (askButton) {
    //     askButton.addEventListener('click', () => {
    //         // document.getElementById('question-response').innerHTML = '<p>Asking bot for response...<span class="loading-dots">...</span></p>';


    //         // const documentId = document.getElementById('document-select').value;

    //         // let selectedOptions = document.getElementById("document-select").selectedOptions;
    //         // let documentIds = Array.from(selectedOptions).map(option => option.value);

    //         let documentIds = Array.from(selectedDocumentIds);


    //         const question = document.getElementById('question-input').value;
    //         fetch('/ask', {
    //             method: 'POST',
    //             headers: { 'Content-Type': 'application/json' },
    //             // body: JSON.stringify({ document_id: documentId, question: question })
    //             body: JSON.stringify({ document_ids: documentIds, question: question })

    //         })
    //         .then(response => response.json())
    //         .then(data => {
    //             const dataframeDiv = document.getElementById('dataframe');
    //             dataframeDiv.innerHTML = data.df_html;
    //             // Iterate through the table cells, replacing newline characters with HTML line breaks
    //             const tableCells = dataframeDiv.querySelectorAll('td');
    //             tableCells.forEach(cell => {
    //                 cell.innerHTML = cell.textContent.replace(/\n/g, '<br>');
    //             });
    //             // Display the user's question
    //             document.getElementById('user-question-display').innerText = "Your Question : " + data.user_question;
    //             // Display the answer
    //             // document.getElementById('question-response').textContent = data.answer;
    //         });
    //     });
    // }
    const askButton = document.getElementById('ask-button');
if (askButton) {
    askButton.addEventListener('click', () => {
        // Set the 'Asking bot for response...' message in the 'user-question-display' div
        document.getElementById('user-question-display').innerHTML = '<p>Asking bot for response...<span class="loading-dots">...</span></p>';

        let documentIds = Array.from(selectedDocumentIds);
        const question = document.getElementById('question-input').value;

        fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ document_ids: documentIds, question: question })
        })
        .then(response => response.json())
        .then(data => {
            const dataframeDiv = document.getElementById('dataframe');
            dataframeDiv.innerHTML = data.df_html;
            // Iterate through the table cells, replacing newline characters with HTML line breaks
            const tableCells = dataframeDiv.querySelectorAll('td');
            tableCells.forEach(cell => {
                cell.innerHTML = cell.textContent.replace(/\n/g, '<br>');
            });
            // Display the user's question, overwriting the 'Asking bot for response...' message
            document.getElementById('user-question-display').innerText = "Your Question : " + data.user_question;
        });
    });
}

    // Existing code
    const fileInput = document.getElementById('file-input');
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            var fileName = this.files[0].name;
            document.getElementById('file-name').textContent = fileName;
        });
    }

    // Code to show spinner
    const btnUpload = document.querySelector('.btn-upload');
    if (btnUpload) {
        btnUpload.addEventListener('click', function() {
            const loadingElement = document.querySelector('.loading');
            if (loadingElement) {
                loadingElement.style.display = 'flex'; // Show the entire loading div
            }
        });
    }


    const uploadForm = document.getElementById('upload-form');
    if (uploadForm) {
    document.getElementById('upload-form').addEventListener('submit', function(e) {
        e.preventDefault();

        document.querySelector('.spinner').style.display = 'block';

        // Set a timeout to change the message after 2 minutes
        setTimeout(function() {
            document.getElementById('upload-time-message').innerHTML = "Please remain patient, as the upload process might require approximately 5 to 6 minutes to complete<br>Please also make sure your document has headings styles applied to reduce the learning time";

        }, 120000); // 120000 milliseconds = 2 minutes

    
        let formData = new FormData(this);
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Redirect or display success message
                window.location.href = '/upload_result';
                document.querySelector('.loading').style.display = 'none'; // Hide the entire loading div
                fetchDocumentsForContents(); // Fetch and display updated document list
            } else {
                // Display error message
                document.getElementById('upload_error_msg').innerText = data.message;
            }
        })
        .catch(error => console.error('Error uploading file:', error));
    });

    const fileInput = document.getElementById('file-input');
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            var fileName = this.files[0].name;
            const fileNameElement = document.getElementById('file-name');
            if (fileNameElement) {
                fileNameElement.textContent = fileName;
            }
        });
    }
    }
        
}

